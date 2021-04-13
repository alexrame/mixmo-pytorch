"""
Implementation following setups from PuzzleMix authors
According to the seminal code: https://github.com/google-research/augmix/blob/master/cifar.py
This code structure is borrowed from:
https://github.com/ildoonet/pytorch-randaugment/blob/616ef12a5176169b4e1e645728f3dedd1a5a148e/RandAugment/augmentations.py
"""

import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch

from mixmo.utils import misc


def ShearX(img, v, myrandom=None):  # [-0.3, 0.3]
    if myrandom is None:
        myrandom = random
    assert -0.3 <= v <= 0.3
    if myrandom.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v, myrandom=None):  # [-0.3, 0.3]
    if myrandom is None:
        myrandom = random
    assert -0.3 <= v <= 0.3
    if myrandom.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v, myrandom=None):  # [-150, 150] => percentage: [-0.45, 0.45]
    if myrandom is None:
        myrandom = random
    assert -0.45 <= v <= 0.45
    if myrandom.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v, myrandom=None):  # [-150, 150] => percentage: [-0.45, 0.45]
    if myrandom is None:
        myrandom = random
    assert -0.45 <= v <= 0.45
    if myrandom.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v, myrandom=None):  # [-30, 30]
    if myrandom is None:
        myrandom = random
    assert -30 <= v <= 30
    if myrandom.random() > 0.5:
        v = -v
    return img.rotate(v)


def AutoContrast(img, *args, **kwargs):
    return PIL.ImageOps.autocontrast(img)


def Invert(img, *args, **kwargs):
    return PIL.ImageOps.invert(img)


def Equalize(img, *args, **kwargs):
    return PIL.ImageOps.equalize(img)


def Solarize(img, v, **kwargs):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def Posterize(img, v, **kwargs):  # [4, 8]
    assert 4 <= v <= 8
    v = int(v)
    return PIL.ImageOps.posterize(img, v)


def Contrast(img, v, **kwargs):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v, **kwargs):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v, **kwargs):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v, **kwargs):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Identity(img, v, **kwargs):
    return img



def augment_list(include_auto_contrast=False):
    # Accepted: equalize, posterize, rotate, solarize, shear_x, shear_y, translate_x, translate_y
    # Rejected: autocontrast?, brightness, contrast, color, sharpness
    l = [
        (Identity, 0., 1.0),
        (Equalize, 0, 1),  # 7
        (Posterize, 4, 8),  # 9
        (Rotate, 0, 30),  # 4
        (Solarize, 0, 256),  # 8
        (ShearX, 0., 0.3),  # 0
        (ShearY, 0., 0.3),  # 1
        (TranslateX, 0., 0.45),  # 2
        (TranslateY, 0., 0.45),  # 3
        (Invert, 0, 1),  # 6,
    ]
    if include_auto_contrast:
        l.append((AutoContrast, 0, 1))  # 5
    return l



def get_value_when_none(value, default_value):
    if value is None:
        return default_value
    return value



class AugMix:
    _default_mixture_depth = -1
    _default_mixture_width = 3
    _default_severity = 3  # [0, 30]

    def __init__(self, seed=None, mixture_depth=None, mixture_width=None, aug_severity=None, include_auto_contrast=False):
        self.mixture_depth = get_value_when_none(
            mixture_depth, self._default_mixture_depth)
        self.mixture_width = get_value_when_none(mixture_width, self._default_mixture_width)
        self.aug_severity = get_value_when_none(aug_severity, self._default_severity)
        self.augment_list = augment_list(include_auto_contrast=include_auto_contrast)
        self.seed = seed

    def __call__(self, image, preprocess, ):
        myrandom = misc.get_random(self.seed)
        mynprandom = misc.get_nprandom(self.seed)

        ws = np.float32(mynprandom.dirichlet([1] * self.mixture_width))
        m = np.float32(mynprandom.beta(1, 1))

        mix = torch.zeros_like(preprocess(image))
        for i in range(self.mixture_width):
            image_aug = image.copy()
            depth = self.mixture_depth if self.mixture_depth > 0 else myrandom.randint(1, 4)
            for _ in range(depth):
                x = mynprandom.choice(range(0, len(self.augment_list)))
                op, minval, maxval = self.augment_list[x]
                val = (float(self.aug_severity) / 30) * float(maxval - minval) + minval
                image_aug = op(image_aug, val, myrandom=myrandom)

            # Preprocessing commutes since all coefficients are convex
            mix += ws[i] * preprocess(image_aug)

        mixed = (1 - m) * preprocess(image) + m * mix
        return mixed
