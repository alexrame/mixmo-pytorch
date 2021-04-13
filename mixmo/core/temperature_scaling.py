"""
Temperature scaling functions and networks modules
Taken from https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
"""

import torch
from torch import nn, optim

from mixmo.utils.logger import get_logger

LOGGER = get_logger(__name__, level="INFO")



def apply_temperature_on_logits(logits, temperature):
    """
    Apply temperature relaxation on logits
    """
    reshaped_temperature = temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
    return logits / reshaped_temperature


class NetworkWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a network with temperature scaling
    network (nn.Module):
    """
    default_temperature = 1.0
    def __init__(self, network, temperature=None, device=None):
        nn.Module.__init__(self)

        self.network = network
        self._init_temperature = temperature or self.default_temperature
        self.device = device
        self.set_temperature(self._init_temperature)

    def set_temperature(self, temperature):
        self.temperature = nn.Parameter(
            torch.ones(1) * temperature,
            requires_grad=True
        )
        self.to(self.device)

    def forward(self, input):
        out = self.network(input)
        out = self.apply_temperature(out, self.temperature)
        return out

    @staticmethod
    def apply_temperature(output, temperature):
        """
        Apply temperature scaling to outputs
        """
        output["logits_prescaled"] = output["logits"]
        output["logits"] = apply_temperature_on_logits(
            logits=output["logits_prescaled"], temperature=temperature
        )
        return output

    def learn_temperature_gridsearch(self, valid_loader, lrs, max_iters):
        best_temperature = self.default_temperature
        valid_loader_processed = self._prepare_training(valid_loader)
        best_nll = valid_loader_processed[2]

        for lr in lrs:
            for max_iter in max_iters:
                temperature, nll = self._learn_temperature(
                    valid_loader=None,
                    valid_loader_processed=valid_loader_processed,
                    lr=lr,
                    max_iter=max_iter
                )
                if best_nll > nll:
                    best_nll = nll
                    best_temperature = temperature
                self.set_temperature(self._init_temperature)

        assert best_nll <= valid_loader_processed[2], "temperature scaling failed because nll increased"
        LOGGER.warning(f"Selecting temperature: {best_temperature:.5f} - nll : {best_nll:.5f}")
        self.set_temperature(temperature=best_temperature)

    def _prepare_training(self, valid_loader):
        self.nll_criterion = nn.CrossEntropyLoss().to(self.device)

        # First: collect all the logits and targets for the validation set
        logits_list = []
        targets_list = []
        with torch.no_grad():
            for data in valid_loader:
                (input, target) = data
                input = input.to(self.device)
                logits = self.network(input)["logits"]

                logits_list.append(logits)
                targets_list.append(target)
            logits = torch.cat(logits_list).to(self.device)
            targets = torch.cat(targets_list).to(self.device)

        # Calculate NLL before temperature scaling
        before_temperature_nll = self.nll_criterion(logits, targets).item()
        LOGGER.debug(f'Before temperature - nll: {before_temperature_nll:.5f}')

        return logits, targets, before_temperature_nll

    # This function probably should live outside of this class, but whatever
    def _learn_temperature(self, valid_loader, lr, max_iter, valid_loader_processed=None):
        """
        Tune the temperature of the network (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        if valid_loader_processed is not None:
            (logits, targets, before_temperature_nll) = valid_loader_processed
        else:
            (logits, targets, before_temperature_nll) = self._prepare_training(valid_loader)

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval():
            loss = self.nll_criterion(
                apply_temperature_on_logits(logits, self.temperature),
                targets)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL after temperature scaling
        after_temperature_nll = self.nll_criterion(
            apply_temperature_on_logits(logits, self.temperature),
            targets).item()
        LOGGER.debug(
            f'With lr: {lr:.6f}, temperature {self.temperature.item():.5f} - nll: {after_temperature_nll:.5f}'
        )

        if after_temperature_nll > before_temperature_nll:
            LOGGER.error(r"Temperature scaling failed for lr: {lr:.6f}")
            return self.default_temperature, before_temperature_nll

        temperature = self.temperature.detach().to("cpu").numpy()[0]
        return temperature, after_temperature_nll
