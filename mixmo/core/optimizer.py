"""
Optimizer factory
"""

import torch.optim as optim
from mixmo.utils.logger import get_logger


LOGGER = get_logger(__name__, level="DEBUG")


def get_optimizer(optimizer, list_param_groups):
    """
    Builds optimizer objects from config
    """
    optimizer_name = optimizer["name"]
    optimizer_params = optimizer["params"]

    LOGGER.info(f"Using optimizer {optimizer_name} with params {optimizer_params}")
    if optimizer_name == "sgd":
        optimizer = optim.SGD(list_param_groups, **optimizer_params)
    elif optimizer_name == "adam":
        optimizer = optim.Adam(list_param_groups, **optimizer_params)
    elif optimizer_name == "adadelta":
        optimizer = optim.Adadelta(list_param_groups, **optimizer_params)
    elif optimizer_name == "rmsprop":
        optimizer = optim.RMSprop(list_param_groups, **optimizer_params)
    else:
        raise KeyError("Bad optimizer name or not implemented (sgd, adam, adadelta).")

    return optimizer
