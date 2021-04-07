"""
Mixing blocks inspired from several standard mixing sample data augmentations
"""

import numpy as np
import torch
import torch.nn.functional as F
import numpy as np
import math

from mixmo.utils import misc


def mix_manifolds(list_lfeats, metadata):
    """
    Main function to mix manifolds in network

    :param list_lfeats: list of latent features: torch.tensors encodings in merged features space
    :param metadata: metadata information passed in network
    :type metadata: dict

    :return: merged representation
    :rtype: torch tensor
    """

    num_inputs = len(list_lfeats)
    if num_inputs == 1:
        return list_lfeats[0]

    if metadata["mode"] == "inference" and "mixmo_masks" not in metadata:
        # Standard sum mixing at inference
        tensor_aggreg = torch.stack(list_lfeats, dim=0).mean(dim=0)
    elif metadata["mode"] == "train" and "mixmo_masks" in metadata:
        # Custom mixing during training

        ## 1. Computing masks
        mixmo_masks = [
            mimo_mix_mask.to(list_lfeats[0].device).to(torch.float)
            for mimo_mix_mask in metadata["mixmo_masks"]
        ]

        ## 2. Combine latent features with given masks
        if len(mixmo_masks) == 1:
            ### Special case for M=2, we can directly sum
            mask_0 = mixmo_masks[0]
            tensor_aggreg = mask_0 * list_lfeats[0] + (1. - mask_0) * list_lfeats[1]
        else:
            ### General case
            assert num_inputs > 2
            list_tensor_aggreg = [mixmo_masks[i] * list_lfeats[i] for i in range(num_inputs)]
            tensor_aggreg = torch.stack(list_tensor_aggreg, dim=0).sum(dim=0)
    else:
        raise ValueError(metadata)

    return num_inputs * tensor_aggreg


######################################### Mask computing functions for mixing #########################################


def mix(method, lams, input_size, config_mix=None):
    """
    Front facing function that computes the masks/lams for any
    number of inputs
    """
    if len(lams) == 2:
        return _single_mix(method, lams[0], input_size, config_mix)
    return _n_mix(method, lams, input_size, config_mix=None)


def _single_mix(method, lam, input_size, config_mix):
    """
    Computes masks for two inputs (traditional MSDA methods)

    Inputs:
    -------

    method: string, mixing to use
    lam: float

    Returns:
    --------
    masks: list of torch.Tensor.
    The masks used for mixing.

    lams: list of torch.Tensor.
    The lams ratio on the final masks.
    """

    config_mix = config_mix or {}
    mask = MASK_MIX_DICT[method](input_size, lam, config_mix)
    lam = mask.mean()         # adjust for potential drift when computing masks
    return [mask, 1 - mask], [lam, 1 - lam]


def _n_mix(method, lams, input_size, config_mix):
    """
    Computes masks for M>2 inputs (requires lam tuples)

    Inputs:
    -------

    method: string, mixing to use
    lam: float

    Returns:
    --------
    masks: list of torch.Tensor.
    The masks used for mixing.

    lams: list of torch.Tensor.
    The lams ratio on the final masks.
    """
    config_mix = config_mix or {}
    masks = MASK_N_MIX_DICT[method](input_size, lams, config_mix)
    lams = [mask.mean() for mask in masks]  # adjust for potential drift
    return masks, lams


######################################### Mask generating functions #########################################


def _mixup_mask(input_size, lam, config_mix):
    """
    Compute masks for MixUp (constant masks)
    """
    mask = lam * torch.ones(input_size)
    return mask


def _cutmix_mask(input_size, lam, config_mix=None):
    """
    Compute masks for CutMix
    """
    # Get box
    [bbx1, bby1, bbx2, bby2] = _rand_bbox_of_area_lam(
        input_size, lam, seed=(config_mix or {}).get("seed"))

    # Build the mask
    mask = torch.zeros(input_size)
    mask[:, bbx1:bbx2, bby1:bby2] = 1

    return mask


def _cow_mask(input_size, lam, config_mix):
    """
    Compute masks for CowMix
    lam is overridden by Cowmask's parameters
    https://github.com/google-research/google-research/tree/master/milking_cowmask/masking
    """
    # Default CowMix config
    misc.ifnotfound_update(
        config_mix,
        {
            "cow_p_max": 0.8,
            "cow_p_min": 0.2,
            "cow_sigma_max": 16.0,
            "cow_sigma_min": 4.0,
        }
    )

    # Get lam ratio for the mask
    p_max = config_mix["cow_p_max"]
    p_min = config_mix["cow_p_min"]
    proba = torch.tensor(p_min + np.random.rand(1) * (p_max - p_min))

    # Get sigma for Gaussian kernel
    sigma_max = config_mix["cow_sigma_max"]
    sigma_min = config_mix["cow_sigma_min"]
    sigma = np.exp(math.log(sigma_min) + np.random.rand(1) * (math.log(sigma_max) - math.log(sigma_min)))

    # Compute Gaussian kernel
    gaussian_kernel = _gaussian_blur_kernel(sigma, sigma_max)
    gaussian_kernel = gaussian_kernel.unsqueeze(-1)
    gaussian_kernel = gaussian_kernel.T * gaussian_kernel

    # Shape it as a proper kernel
    gaussian_kernel = gaussian_kernel.unsqueeze(0).unsqueeze(0)
    gaussian_kernel = gaussian_kernel.repeat((input_size[0], 1, 1, 1)).float()

    noise = torch.randn(1, 1, input_size[1], input_size[2])

    blurred_noise = F.conv2d(noise, gaussian_kernel, padding = gaussian_kernel.size()[-1]//2)

    noise_mean = blurred_noise.mean()
    noise_std = blurred_noise.std()

    # Get thresholded cowmask
    threshold_stat = noise_mean + math.sqrt(2) * torch.erfinv(2*proba - 1) * noise_std

    mask = blurred_noise <= threshold_stat

    return mask.squeeze(0).float()


def _stack_mask(input_size, lam, config_mix):
    """
    Compute masks for Channel/Horizontal/Vertical concat
    (number of images) x channel x (image width) x (image height)
    """
    # Default config
    misc.ifnotfound_update(config_mix,
                           {
                               "stack_dim": 1,
                               "stack_rdflip": True,
                           })

    dim = config_mix["stack_dim"]
    random_flip = config_mix["stack_rdflip"]

    flip = random_flip and misc.random_lower_than(
        prob=0.5, seed=None, r=None)
    if flip:
        lam = 1 - lam

    # Split the dimension in two
    border = int(lam * input_size[dim])

    ones_size = list(input_size)
    ones_size[dim] = border

    zeros_size = list(input_size)
    zeros_size[dim] = input_size[dim] - border

    ones_mask = torch.ones(ones_size)
    zeros_mask = torch.zeros(zeros_size)

    # Merge the two split masks
    if flip:
        mask = torch.cat([zeros_mask, ones_mask], dim=dim)
    else:
        mask = torch.cat([ones_mask, zeros_mask], dim=dim)

    return mask


def _stack2_mask(input_size, lam, config_mix):
    """
    Wrapper function for vertical concat mixing
    """
    misc.ifnotfound_update(config_mix, {"stack_dim": 2, "stack_rdflip": True})

    return _stack_mask(input_size, lam, config_mix)


def _stack0_mask(input_size, lam, config_mix):
    """
    Wrapper function for channel concat mixing
    """
    misc.ifnotfound_update(config_mix, {"stack_dim": 0, "stack_rdflip": True})

    return _stack_mask(input_size, lam, config_mix)


def _noise_mask(input_size, lam, config_mix):
    """
    Random mask pixels drawn from uniform distribution
    """
    if config_mix["noise_2d"]:
        mask = torch.rand(input_size[1:])
    else:
        mask = torch.rand(input_size)

    # rescale the center with a piecewise linear function to have the proper lam
    mask_below = torch.min(mask - 0.5,torch.zeros_like(mask))
    mask_above = torch.max(mask - 0.5,torch.zeros_like(mask))

    mask = 2 * (lam * mask_below +
                (1-lam) * mask_above) + lam

    if config_mix["noise_2d"]:
        mask = mask.repeat((3,1,1))

    return mask

def _patchup_mask(input_size, lam, config_mix):
    """
    Compute masks for PatchUp mixing
    https://github.com/chandar-lab/PatchUp
    """
    # Default config
    misc.ifnotfound_update(
        config_mix,
        {
        "patchup_gamma": None,
        "patchup_block_size": 7,
        "patchup_soft": False,
        "patchup_2d": False
    })

    if config_mix.get("patchup_gamma", None) is not None:
        gamma = config_mix["patchup_gamma"]
    else:
        gamma = lam

    block_size = config_mix["patchup_block_size"]

    kernel_size = (block_size, block_size)
    padding = (block_size//2, block_size//2)
    stride = (1,1)

    # As per the official patchup_hard implementation
    gamma *= (input_size[-1] ** 2 / (
        block_size ** 2 * (input_size[-1] - block_size + 1) ** 2
        )
        )

    if config_mix["patchup_2d"]:
        p = gamma * torch.ones(input_size[1:])
    else:
        p = gamma * torch.ones(input_size)

    m_i_j = torch.bernoulli(p)

    if config_mix["patchup_2d"]:
        m_i_j = m_i_j.repeat((input_size[0], 1, 1))

    # following line provides the continuous blocks that should be altered with PatchUp denoted as holes here.
    mask = F.max_pool2d(m_i_j, kernel_size, stride, padding)

    if config_mix["patchup_soft"]:
        mask = mask + (1 - mask) * lam

    return mask


def _patchuphard2d_mask(input_size, lam, config_mix):
    """
    Wrapper function for PatchUp hard masking (2d variant)
    """
    misc.ifnotfound_update(config_mix, {
        "patchup_2d": True,
    })
    return _patchup_mask(input_size, lam, config_mix)


def _patchupsoft_mask(input_size, lam, config_mix):
    """
    Wrapper function for PatchUp soft masking
    """
    misc.ifnotfound_update(config_mix, {
        "patchup_soft": True,
    })

    assert config_mix["patchup_soft"]
    return _patchup_mask(input_size, lam, config_mix)


def _channel_mask(input_size, lam, config_mix):
    """
    Compute masks that toggle entire channels on and off
    """
    p = lam * torch.ones((input_size[0], 1, 1))  # 0 is the channel dimension
    mask = torch.bernoulli(p)
    mask = mask.expand(input_size)
    return mask


MASK_MIX_DICT = {
    "patchuphard": _patchup_mask,
    "patchupsoft": _patchupsoft_mask,
    "patchuphard2d": _patchuphard2d_mask,
    "mixup": _mixup_mask,
    "cutmix": _cutmix_mask,
    "noise": _noise_mask,
    "stackchannel": _stack0_mask,  # split on dimension 0
    "stackhorizontal": _stack_mask,  # split on dimension 1
    "stackvertical": _stack2_mask,  # split on dimension 2
    "channel": _channel_mask,
    "cow": _cow_mask,
}

LIST_METHODS_NOT_INVARIANT_CHANNELS = [
    "channel",
    "stackchannel",
]


def _n_mixup_mask(input_size, lams, config_mix):
    """
    Multivariate MixUp generalization
    lam is a tuple here (simplex) that gives the proportion between n inputs
    """
    masks = []
    for lam in lams:
        masks.append(lam * torch.ones(input_size))

    return masks


def _n_cutinmix_mask(input_size, lams, config_mix):
    """
    Multivariate CutMix generalization (see paper)
    lam is a tuple here (simplex) that gives the proportion between n inputs
    CutMix(A, MixUp(B,C,...))
    """
    # Compute the base masks
    mixup_lams = [lam / sum(lams[1:]) for lam in lams[1:]]
    mixup_masks = _n_mixup_mask(input_size, mixup_lams, config_mix)
    cutmix_mask = _cutmix_mask(input_size, lams[0], config_mix)

    # Combine the mixup and cutmix masks
    masks = [cutmix_mask]
    masks += [(1 - cutmix_mask) * mask for mask in mixup_masks]

    return masks


MASK_N_MIX_DICT = {
    "mixup": _n_mixup_mask,
    "cutmix": _n_cutinmix_mask,
}



######################################### General utility functions for masking #########################################


def _rand_bbox_of_area_lam(size, lam, seed=None):
    """
    Compute the corner coordinates of a random rectangular box such that
    area_box/area_image=lam
    """
    # Retrieving H and W depending on image format
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception

    # Compute box width and height
    cut_rat = np.sqrt(lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # Draw coordinates of the center of the box
    mynprandom = misc.get_nprandom(seed=seed)
    try:
        rg_integers = mynprandom.randint
    except AttributeError:
        rg_integers = mynprandom.integers
    cx = rg_integers(W)
    cy = rg_integers(H)

    # Box corners naturally follow
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def _gaussian_blur_kernel(sigma, sigma_max, sym=True):
    """
    Compute Gaussian kernel, as per the scipy.signal implementation
    """
    size = math.ceil(sigma_max * 3) * 2 + 1  # Keep up to 99.7 of the Gaussian for the kernel

    n = torch.arange(0, size).float() - (size - 1.0) / 2.0

    sig2 = 2 * sigma * sigma

    w = np.exp(-n**2 / sig2)

    return w / math.sqrt(math.pi * sig2)
