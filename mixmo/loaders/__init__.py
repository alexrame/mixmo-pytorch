"""
Dataset wrappers and processing
"""

from mixmo.loaders import loader


def get_loader(config_args, **kwargs):
    """
    Return a new instance of dataset loader
    """
    # Available datasets
    data_loader_factory = {
        "cifar10": loader.CIFAR10Loader,
        "cifar100": loader.CIFAR100Loader,
        "tinyimagenet200": loader.TinyImagenet200Loader,
    }

    return data_loader_factory[config_args['data']['dataset']](config_args=config_args, **kwargs)
