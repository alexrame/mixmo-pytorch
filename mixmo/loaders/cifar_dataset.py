"""
CIFAR 10 and 100 dataset wrappers
"""

import os
from PIL import Image
import numpy as np
from torchvision import datasets

from mixmo.utils.logger import get_logger

LOGGER = get_logger(__name__, level="DEBUG")


class CustomCIFAR10(datasets.CIFAR10):
    """
    Torchvision's CIFAR10 dataset class augmented with a custom __getitem__ method
    """
    def __getitem__(self, index, apply_postprocessing=True):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img, apply_postprocessing=apply_postprocessing)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target



class CustomCIFAR100(datasets.CIFAR100):
    """
    Torchvision's CIFAR100 dataset class augmented with a custom __getitem__ method
    """
    def __getitem__(self, *args, **kwargs):
        return CustomCIFAR10.__getitem__(self, *args, **kwargs)



CIFAR_ROBUSTNESS_FILENAMES = [
    "speckle_noise.npy", "shot_noise.npy", "impulse_noise.npy", "defocus_blur.npy",
    "gaussian_blur.npy", "glass_blur.npy", "zoom_blur.npy", "fog.npy", "brightness.npy",
    "contrast.npy", "elastic_transform.npy", "pixelate.npy", "jpeg_compression.npy",
    "motion_blur.npy", "snow.npy", 'frost.npy', 'gaussian_noise.npy', 'saturate.npy', 'spatter.npy'
]


class CIFARCorruptions(CustomCIFAR10):

    def __init__(self, root, transform=None, train=False):

        super(datasets.CIFAR10, self).__init__(root)
        self.transform = transform
        self.train = train  # training set or test set
        self.data = []
        self.targets = []

        labels_path = os.path.join(root, "labels.npy")
        list_labels = np.load(labels_path).tolist()

        ## iterate over CIFAR_ROBUSTNESS_FILENAMES
        for i, filename in enumerate(CIFAR_ROBUSTNESS_FILENAMES):
            filepath = os.path.join(root, filename)
            data = np.load(filepath)

            if i == 0:
                ## initialize with first filename
                self.data = data
                self.targets = list_labels[:]
            else:
                self.data = np.concatenate(
                    (self.data, data),
                    axis=0)
                self.targets.extend(list_labels)

        nb_files = len(CIFAR_ROBUSTNESS_FILENAMES)
        LOGGER.warning(f"Robustness corruptions of len: {nb_files}")
        LOGGER.warning(f"Pixels of shape: {self.data.shape}")
        LOGGER.warning(f"Targets of len: {len(self.targets)}")
