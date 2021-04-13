"""
Base loss definitions
"""
from collections import OrderedDict
import copy
import torch
import torch.nn as nn

from mixmo.utils import misc, logger

LOGGER = logger.get_logger(__name__, level="DEBUG")


class AbstractLoss(nn.modules.loss._Loss):
    """
    Base loss class defining printing and logging utilies
    """
    def __init__(self, config_args, device, config_loss=None):
        self.device = device
        self.config_args = config_args or {}
        self.config_loss = config_loss or {}
        self.name = self.config_loss["display_name"]
        nn.modules.loss._Loss.__init__(self)

    def print_details(self):
        LOGGER.info(f"Using loss: {self.config_loss} with name: {self.name}")

    def start_accumulator(self):
        self._accumulator_loss = 0
        self._accumulator_len = 0

    def get_accumulator_stats(self, format="short", split=None):
        """
        Gather tracked stats into a dictionary as formatted strings
        """
        if not self._accumulator_len:
            return {}

        stats = OrderedDict({})
        loss_value = self._accumulator_loss / self._accumulator_len

        if format == "long":
            assert split is not None
            key = split + "/" + self.name
            stats[key] = {
                "value": loss_value,
                "string": f"{loss_value:.5}",
            }
        else:
            # make it as short as possibe to fit on one line of tqdm postfix
            loss_string = f"{loss_value:.3}".replace("e-0", "-").replace("e-", "-")
            stats[self.name] = loss_string

        return stats

    def forward(self, input, target):
        current_loss = self._forward(input, target)
        self._accumulator_loss += current_loss.detach().to("cpu").numpy()
        self._accumulator_len += 1
        return current_loss

    def _forward(self, input, target):
        raise NotImplementedError


class SoftCrossEntropyLoss(AbstractLoss):
    """
    Soft CrossEntropy loss that specifies the proper forward function for AbstractLoss
    """
    def _forward(self, input, target):
        """
        Cross entropy that accepts soft targets
        Args:
            pred: predictions for neural network
            targets: targets, can be soft
            size_average: if false, sum is returned instead of mean

        Examples::

            input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
            input = torch.autograd.Variable(out, requires_grad=True)

            target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
            target = torch.autograd.Variable(y1)
            loss = cross_entropy(input, target)
            loss.backward()
        """
        if len(target.size()) == 1:
            target = torch.nn.functional.one_hot(target, num_classes=input.size(-1))
            target = target.to(torch.float).to(self.device)
        logsoftmax = torch.nn.LogSoftmax(dim=1)

        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))



DICT_LOSS_STANDARD = {
    "soft_cross_entropy": SoftCrossEntropyLoss,
}


class WrapperLoss(AbstractLoss):
    """
    Wrapper around the multiple losses. Initialized from listloss.
    """
    def __init__(self, config_loss, config_args, device):
        AbstractLoss.__init__(
            self,
            config_args=config_args,
            config_loss=config_loss,
            device=device,
        )
        self.losses = self._init_get_losses()
        self.regularized_network = None

    def _init_get_losses(self):
        """
        Initialize and gather losses from listloss
        """
        losses = []
        for ic, config_loss in enumerate(self.config_loss["listloss"]):
            if config_loss["coeff"] == "<num_members":
                config_loss["coeff"] = (1. if ic < self.config_args["num_members"] else 0)
            if config_loss["coeff"] == 0:
                LOGGER.debug(f"Skip loss: {config_loss}")
                continue

            loss_callable = get_loss(config_loss, device=self.device, config_args=self.config_args)
            loss = copy.deepcopy(config_loss)
            loss["callable"] = loss_callable
            losses.append(loss)
        return losses

    def print_details(self):
        return

    def start_accumulator(self):
        AbstractLoss.start_accumulator(self)
        for loss in self.losses:
            loss["callable"].start_accumulator()

    def get_accumulator_stats(self, format="short", split=None):
        """
        Gather tracked stats into a dictionary as formatted strings
        """
        if not self._accumulator_len:
            return {}

        stats = AbstractLoss.get_accumulator_stats(self, format=format, split=split)

        if format == "long":
            # tensorboard logs
            if self.config_loss.get("l2_reg"):
                l2_reg = self.l2_reg().detach().to("cpu").numpy()
                stats["general/l2_reg"] = {
                    "value": l2_reg,
                    "string": f"{l2_reg:.4}",
                }
            for loss in self.losses:
                substats = loss["callable"].get_accumulator_stats(
                    format=format,
                    split=split,
                )
                misc.clean_update(stats, substats)

        return stats

    def _forward(self, input, target):
        """
        Perform loss forwards for each sublosses and l2 reg
        """
        computed_losses = [self._forward_subloss(loss, input, target) for loss in self.losses]
        stacked_computed_losses = torch.stack(computed_losses)
        final_loss = stacked_computed_losses.sum()

        if self.config_loss.get("l2_reg"):
            final_loss = final_loss + self.l2_reg() * float(self.config_loss.get("l2_reg"))
        return final_loss

    def _forward_subloss(self, loss, input, target):
        """
        Standard loss forward for one of the sublosses
        """
        coeff = float(loss["coeff"])
        subloss_input = self._match_item(loss["input"], dict_tensors=input)
        subloss_target = self._match_item(loss["target"], dict_tensors=target)
        loss = loss["callable"](input=subloss_input, target=subloss_target)
        return loss * coeff

    @staticmethod
    def _match_item(name, dict_tensors):
        if misc.is_none(name):
            return None
        if name in dict_tensors:
            return dict_tensors[str(name)]
        raise ValueError(name)

    def set_regularized_network(self, network):
        if self.config_loss.get("l2_reg"):
            self.regularized_network = network
            LOGGER.warning(f"Set l2 regularization on {network.__class__.__name__}")

    def l2_reg(self,):
        """
        Compute l2 regularization/weight decay over the non-excluded parameters
        """
        assert self.regularized_network is not None

        # Retrieve non excluded parameters
        params = list(self.regularized_network.parameters())

        # Iterate over all parameters to decay
        l2_reg = None
        for W in params:
            if l2_reg is None:
                l2_reg = torch.sum(torch.pow(W, 2))
            else:
                l2_reg = l2_reg + torch.sum(torch.pow(W, 2))
        assert l2_reg is not None

        return l2_reg


def get_loss(config_loss, device=None, config_args=None):
    """
    Construct loss object, wrapped if there are multiple losses
    """
    loss_name = config_loss["name"]
    if loss_name == "multitask":
        loss = WrapperLoss(config_args=config_args, device=device, config_loss=config_loss)
    elif loss_name in DICT_LOSS_STANDARD:
        loss = DICT_LOSS_STANDARD[loss_name](
            config_loss=config_loss, config_args=config_args, device=device
        )
    else:
        raise Exception(f"Loss {loss_name} not implemented")
    loss.print_details()
    return loss
