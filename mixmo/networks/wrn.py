"""
WideResNet network definition
ported from https://github.com/meliketoy/wide-resnet.pytorch/blob/master/networks/wide_resnet.py
"""

import torch.nn as nn
import torch.nn.functional as F

from mixmo.utils.logger import get_logger
from mixmo.networks.resnet import (PreActResNet, PreActResNetMixMo)

LOGGER = get_logger(__name__, level="DEBUG")

BATCHNORM_MOMENTUM = 0.1
# Tensorflow setup from edward2
BATCHNORM_MOMENTUM_END = 0.1
USE_BIAS = False


class WideBasic(nn.Module):
    """
    Basic wide residual block module
    """

    expansion = 1
    def __init__(self, inplanes, planes, stride=1):
        super(WideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, momentum=BATCHNORM_MOMENTUM)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, bias=USE_BIAS)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BATCHNORM_MOMENTUM)
        self.stride = stride
        padding = 1 if stride == 1 else 0
        self.pad1 = nn.ZeroPad2d((0, 1, 0, 1))

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=padding, bias=USE_BIAS)

        self.equalInOut = (inplanes == planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or not self.equalInOut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=USE_BIAS),
            )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.relu(self.bn2(out))
        if self.stride != 1 and not self.equalInOut:
            out = self.pad1(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        return out


class WideResNet(PreActResNet):
    """
    Standard WideResNet network
    Mostly re-using PreActResNet API and defining WideResNet specific parameters/architecture
    """

    def _init_block(self, widen_factor):
        # Standard wide resnet depth check
        assert ((self.depth - 4) % 6 == 0), 'Wide-resnet self.depth should be 6n+4'
        n = (self.depth - 4) / 6
        self._block = WideBasic
        self._layers = [n for _ in range(3)]
        self._nChannels = [
            16,
            16 * widen_factor, 32 * widen_factor, 64 * widen_factor
        ]
    def _make_conv1(self, nb_input_channel):
        return nn.Conv2d(
            nb_input_channel, self._nChannels[0], kernel_size=3, stride=1, padding=1, bias=USE_BIAS)

    def _make_layer(self, block, planes, blocks, stride=1):
        strides = [stride] + [1] * int(blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.inplanes, planes, stride))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def _init_core_network(self):
        PreActResNet._init_core_network(self, max_layer=3)
        self.bn1 = nn.BatchNorm2d(self._nChannels[3], momentum=BATCHNORM_MOMENTUM_END)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    dense_gaussian = False

    def _forward_core_network(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.relu(self.bn1(x))
        x_avg = self.avgpool(x)
        return x_avg.view(x_avg.size(0), -1)


class WideResNetMixMo(WideResNet):
    """
    Multi-Input Multi-Output WideResNet network
    Mostly re-using ResNetMixMo API and WideResNet architecture/parameters
    """

    def _init_first_layer(self):
        PreActResNetMixMo._init_first_layer(self)

    def _init_final_classifier(self):
        PreActResNetMixMo._init_final_classifier(self)

    def _forward_first_layer(self, *args, **kwargs):
        return PreActResNetMixMo._forward_first_layer(self, *args, **kwargs)

    def _forward_final_classifier(self, *args, **kwargs):
        return PreActResNetMixMo._forward_final_classifier(self, *args, **kwargs)


wrn_network_factory = {
    # For CIFAR
    "wideresnet": WideResNet,
    "wideresnetmixmo": WideResNetMixMo,
}
