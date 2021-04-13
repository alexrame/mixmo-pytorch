"""
General tensor manipulation utility functions (initializations, permutations, one hot)
"""

import math
import torch

import torch.nn as nn

from mixmo.utils.logger import get_logger

LOGGER = get_logger(__name__, level="DEBUG")


def onehot(size, target):
    """
    Translate scalar targets to one hot vectors
    """
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec


# permutation tools

def randperm_static(batch_size, proba_static=0):
    """
    Perform random permutation with a set percentage remaining fixed
    """
    if proba_static <= 0:
        return torch.randperm(batch_size)
    size_static = int(batch_size * proba_static)
    torch_static = torch.arange(0, size_static).long()
    size_shuffled = int(batch_size) - size_static
    torch_shuffled = size_static + torch.randperm(size_shuffled)
    return torch.cat([torch_static, torch_shuffled])



# initializations like in tensorflow

def truncated_normal_(tensor, mean=0, std=1):
    """
    Initialization function
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)



def _calculate_fan_in_and_fan_out(tensor):
    """
    Compute the minimal input and output sizes for the weight tensor
    """
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    if dimensions == 2:  # Linear
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def _calculate_correct_fan(tensor, mode):
    """
    Return the minimal input or output sizes for the weight tensor depending on which is needed
    """
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def kaiming_normal_truncated(tensor, a=0, mode='fan_in', nonlinearity='relu'):
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    normal distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std})` where

    .. math::
        \text{std} = \sqrt{\frac{2}{(1 + a^2) \times \text{fan\_in}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the negative slope of the rectifier used after this layer (0 for ReLU
            by default)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
    """
    fan = _calculate_correct_fan(tensor, mode)
    gain = nn.init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    std = std / .87962566103423978
    with torch.no_grad():
        return truncated_normal_(tensor, 0, std)


def weights_init_hetruncatednormal(m, dense_gaussian=False):
    """
    Simple init function
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        kaiming_normal_truncated(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant(m.bias, 0)
    elif classname.find('Linear') != -1:
        if dense_gaussian:
            nn.init.normal_(m.weight.data, mean=0, std=0.01)
        else:
            kaiming_normal_truncated(
                m.weight.data, a=0, mode='fan_in', nonlinearity='relu'
            )
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        if m.weight is not None:
            m.weight.data.fill_(1)
            m.bias.data.zero_()
