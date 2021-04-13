"""
Custom Dataloaders for each of the considered datasets
"""

import os

from torchvision import datasets

from mixmo.augmentations.standard_augmentations import get_default_composed_augmentations
from mixmo.loaders import cifar_dataset, abstract_loader
from mixmo.utils.logger import get_logger

LOGGER = get_logger(__name__, level="DEBUG")


class CIFAR10Loader(abstract_loader.AbstractDataLoader):
    """
    Loader for the CIFAR10 dataset that inherits the abstract_loader.AbstractDataLoader dataloading API
    and defines the proper augmentations and datasets
    """

    def _init_dataaugmentations(self):
        (self.augmentations_train, self.augmentations_test) = get_default_composed_augmentations(
            dataset_name="cifar",
        )

    def _init_dataset(self, corruptions=False):
        self.train_dataset = cifar_dataset.CustomCIFAR10(
            root=self.data_dir, train=True, download=True, transform=self.augmentations_train
        )
        if not corruptions:
            self.test_dataset = cifar_dataset.CustomCIFAR10(
                root=self.data_dir, train=False, download=True, transform=self.augmentations_test
            )
        else:
            self.test_dataset = cifar_dataset.CIFARCorruptions(
                root=self.corruptions_data_dir, train=False, transform=self.augmentations_test
            )

    @property
    def data_dir(self):
        return os.path.join(self.dataplace, "cifar10-data")

    @property
    def corruptions_data_dir(self):
        return os.path.join(self.dataplace, "CIFAR-10-C")


    @staticmethod
    def properties(key):
        dict_key_to_values = {
            "conv1_input_size": (16, 32, 32),
            "conv1_is_half_size": False,
            "pixels_size": 32,
        }
        return dict_key_to_values[key]


class CIFAR100Loader(CIFAR10Loader):
    """
    Loader for the CIFAR100 dataset that inherits the abstract_loader.AbstractDataLoader dataloading API
    and defines the proper augmentations and datasets
    """

    def _init_dataset(self, corruptions=False):
        self.train_dataset = cifar_dataset.CustomCIFAR100(
            root=self.data_dir, train=True, download=True, transform=self.augmentations_train
        )
        if not corruptions:
            self.test_dataset = cifar_dataset.CustomCIFAR100(
                root=self.data_dir, train=False, download=True, transform=self.augmentations_test
            )
        else:
            self.test_dataset = cifar_dataset.CIFARCorruptions(
                root=self.corruptions_data_dir, train=False, transform=self.augmentations_test
            )

    @property
    def data_dir(self):
        return os.path.join(self.dataplace, "cifar100-data")

    @property
    def corruptions_data_dir(self):
        return os.path.join(self.dataplace, "CIFAR-100-C")


class TinyImagenet200Loader(abstract_loader.AbstractDataLoader):
    """
    Loader for the TinyImageNet dataset that inherits the abstract_loader.AbstractDataLoader dataloading API
    and defines the proper augmentations and datasets
    """

    def _init_dataaugmentations(self):
        (self.augmentations_train, self.augmentations_test) = get_default_composed_augmentations(
            dataset_name="tinyimagenet",
        )

    @property
    def data_dir(self):
        return os.path.join(self.dataplace, "tinyimagenet200-data")

    def _init_dataset(self, corruptions=False):
        traindir = os.path.join(self.data_dir, 'train')
        valdir = os.path.join(self.data_dir, 'val/images')
        self.train_dataset = datasets.ImageFolder(traindir, self.augmentations_train)
        self.test_dataset = datasets.ImageFolder(valdir, self.augmentations_test)

    @staticmethod
    def properties(key):
        dict_key_to_values = {
            "conv1_input_size": (64, 32, 32),
            "conv1_is_half_size": True,
            "pixels_size": 64,
        }
        return dict_key_to_values[key]
