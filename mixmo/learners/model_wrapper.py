import copy
import torch.nn.functional as F
from collections import OrderedDict

from mixmo.networks import get_network
from mixmo.core import (
    loss, optimizer, temperature_scaling, scheduler,
    metrics_wrapper)
from mixmo.utils import logger, misc, torchsummary
from mixmo.utils.config import cfg

LOGGER = logger.get_logger(__name__, level="DEBUG")


def get_predictions(logits):
    """
    Convert logits into softmax predictions
    """
    probs = F.softmax(logits, dim=1)
    confidence, pred = probs.max(dim=1, keepdim=True)
    return confidence, pred, probs


class ModelWrapper:
    """
    Augment a model with losses, metrics, internal logs and other things
    """

    def __init__(self, config, config_args, device):
        self.config = config
        self.name = config["name"]
        self.config_args = config_args
        self.device = device
        self.mode = "notinit"

        self._init_main()

    def _init_main(self):
        self.network = get_network(
            config_network=self.config["network"], config_args=self.config_args
        ).to(self.device)
        self._scaled_network = None
        self.scheduler = None
        self._scheduler_initialized = False

        self.loss = loss.get_loss(
            config_loss=self.config.get("loss"),
            config_args=self.config_args, device=self.device
        )
        if hasattr(self.loss, "set_regularized_network"):
            self.loss.set_regularized_network(self.network)

        self.optimizer = optimizer.get_optimizer(
            optimizer=self.config["optimizer"],
            list_param_groups=[{"params": list(self.network.parameters())}]
        )

    def to_eval_mode(self):
        """
        Switch model to eval mode
        """
        self.mode = "eval"
        self.network.eval()
        self.loss.start_accumulator()
        self._init_metrics()

    def to_train_mode(self, epoch):
        """
        Switch model to train mode
        """
        self.mode = "train"
        if not self._scheduler_initialized:
            self._init_scheduler(epoch)
        self.network.train()
        self.loss.start_accumulator()
        self._init_metrics()

    def _init_scheduler(self, epoch):

        self.scheduler = scheduler.get_scheduler(
            lr_schedule=self.config["lr_schedule"],
            optimizer=self.optimizer,
            start_epoch=epoch - 2,
        )
        self.scheduler.step()
        if epoch == 1 and self.config.get("warmup_period", 0) > 0:
            LOGGER.warning("Warmup period")
            self.warmup_scheduler = scheduler.get_warmup_scheduler(
                optimizer=self.optimizer,
                warmup_period=self.config.get("warmup_period"))
        else:
            self.warmup_scheduler = None
        self._scheduler_initialized = True

    def _init_metrics(self):
        if self.mode == "eval":
            metrics = [*self.config["metrics"]] + self.config.get("metrics_only_test", [])
        else:
            metrics = self.config["metrics"]
        self._metrics = metrics_wrapper.MetricsWrapper(metrics=metrics)

    def print_summary(self, pixels_size=32):
        summary_input = (3 * self.config_args["num_members"], pixels_size, pixels_size)
        try:
            torchsummary.summary(self.network, summary_input, list_dtype=None)
        except:
            LOGGER.warning("Torch summary failed", exc_info=True)

    def step(self, output, target, backprop=False):
        """
        Compute loss, backward step and metrics if required by config
        Update internal records
        """
        current_loss = self.loss(output, target)
        if backprop:
            current_loss.backward(retain_graph=False)

        logits = output["logits" if self.mode != "train" else "logits_0"]
        confidence, pred, probs = get_predictions(logits)

        target = target["target_0"]
        if len(target.size()) == 2:
            target = target.argmax(axis=1)

        self._metrics.update(pred, target, confidence, probs)
        if self.mode != "train":
            self._compute_diversity(output, target)

    def _compute_diversity(self, output, target):
        """
        Compute diversity and update internal records
        """
        if self.config_args["num_members"] > 1:
            predictions = [
                output["logits_" + str(head)].max(dim=1, keepdim=False)[1].detach().to("cpu").numpy()
                for head in range(
                    0, self.config_args["num_members"])
            ]
            if self.config_args["num_members"] != 1:
                self._metrics.update_diversity(
                    target=[int(t) for t in target.detach().to("cpu").numpy()],
                    predictions=predictions,
                )

    def get_short_logs(self):
        """
        Return summary of internal records
        """
        return self.loss.get_accumulator_stats(format="short", split=None)

    def get_dict_to_scores(self, split,):
        """
        Format logs into a dictionary
        """
        logs_dict = OrderedDict({})
        if split == "train":
            lr_value = self.optimizer.param_groups[0]["lr"]
            logs_dict[f"general/{self.name}_lr"] = {
                "value": lr_value,
                "string": f"{lr_value:05.5}",
            }
        misc.clean_update(logs_dict, self.loss.get_accumulator_stats(format="long", split=split))

        if self.mode == "eval":
            LOGGER.info(f"Compute metrics for {self.name} at split: {split}")
            scores = self._metrics.get_scores(split=split)
            for s in scores:
                logs_dict[s] = scores[s]
        return logs_dict

    def predict(self, data):
        """
        Perform a forward pass through the model and return the output
        """
        return self.scaled_network(data)

    @property
    def scaled_network(self):
        """
        Returns scaled_model if necessary for amp
        """
        if self._scaled_network is None:
            return self.network
        else:
            return self._scaled_network

    def calibrate_via_tempscale(self, tempscale_loader):
        """
        Returns calibrated temperature on val/test set
        """
        self.to_eval_mode()
        self._scaled_network = temperature_scaling.NetworkWithTemperature(
            network=self.network, device=self.device
        )
        self._scaled_network.learn_temperature_gridsearch(
            valid_loader=tempscale_loader,
            lrs=cfg.CALIBRATION.LRS,
            max_iters=cfg.CALIBRATION.MAX_ITERS
        )
        return self._scaled_network.temperature.cpu().detach().numpy()[0]
