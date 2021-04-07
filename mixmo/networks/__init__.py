from mixmo.utils.logger import get_logger

from mixmo.networks import resnet, wrn


LOGGER = get_logger(__name__, level="DEBUG")


def get_network(config_network, config_args):
    """
        Return a new instance of network
    """
    # Available networks for tiny
    if config_args["data"]["dataset"].startswith('tinyimagenet'):
        network_factory = resnet.resnet_network_factory
    elif config_args["data"]["dataset"].startswith('cifar'):
        network_factory = wrn.wrn_network_factory
    else:
        raise NotImplementedError

    LOGGER.warning(f"Loading network: {config_network['name']}")
    return network_factory[config_network["name"]](
        config_network=config_network,
        config_args=config_args)
