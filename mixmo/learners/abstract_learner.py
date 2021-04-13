"""
Base Learner wrapper definitions for logging, training and evaluating models
"""

import torch
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter

from mixmo.utils import misc, logger, config
from mixmo.learners import model_wrapper


LOGGER = logger.get_logger(__name__, level="INFO")


class AbstractLearner:
    """
    Base learner class that groups models, optimizers and loggers
    Performs the entire model building, training and evaluating process
    """
    def __init__(self, config_args, dloader, device):
        self.config_args = config_args
        self.device = device
        self.dloader = dloader
        self._tb_logger = None

        self._create_model_wrapper()

        self._best_acc = 0
        self._best_epoch = 0

    def _create_model_wrapper(self):
        """
        Initialize the model along with other elements through a ModelWrapper
        """
        self.model_wrapper = model_wrapper.ModelWrapper(
            config=self.config_args["model_wrapper"],
            config_args=self.config_args,
            device=self.device
        )
        self.model_wrapper.to_eval_mode()
        self.model_wrapper.print_summary(
            pixels_size=self.dloader.properties("pixels_size")
            )

    @property
    def tb_logger(self):
        """
        Get (or initialize) the Tensorboard SummaryWriter
        """
        if self._tb_logger is None:
            self._tb_logger = SummaryWriter(log_dir=self.config_args["training"]["output_folder"])
        return self._tb_logger

    def save_tb(self, logs_dict, epoch):
        """
        Write stats from logs_dict at epoch to the Tensoboard summary writer
        """
        for tag in logs_dict:
            self.tb_logger.add_scalar(tag, logs_dict[tag]["value"], epoch)
        if "test/diversity_accuracy_mean" not in logs_dict:
            self.tb_logger.add_scalar(
                "test/diversity_accuracy_mean",
                logs_dict["test/accuracy"]["value"], epoch
            )

    def load_checkpoint(self, checkpoint, include_optimizer=True, return_epoch=False):
        """
        Load checkpoint (and optimizer if included) to the wrapped model
        """

        checkpoint = torch.load(checkpoint, map_location=self.device)
        self.model_wrapper.network.load_state_dict(checkpoint[self.model_wrapper.name + "_state_dict"], strict=True)
        if include_optimizer:
            if self.model_wrapper.optimizer is not None:
                self.model_wrapper.optimizer.load_state_dict(
                    checkpoint[self.model_wrapper.name + "_optimizer_state_dict"])
            else:
                assert self.model_wrapper.name + "_optimizer_state_dict" not in checkpoint
        if return_epoch:
            return checkpoint["epoch"]

    def save_checkpoint(self, epoch, save_path=None):
        """
        Save model (and optimizer) state dict
        """
        # get save_path
        if epoch is not None:
            dict_to_save = {"epoch": epoch}
            if save_path is None:
                save_path = misc.get_model_path(
                    self.config_args["training"]["output_folder"], epoch=epoch
                )
        else:
            assert save_path is not None

        # update dict to save
        dict_to_save[self.model_wrapper.name + "_state_dict"] = (
            self.model_wrapper.network.state_dict()
            if isinstance(self.model_wrapper.network, torch.nn.DataParallel)
            else self.model_wrapper.network.state_dict())
        if self.model_wrapper.optimizer is not None:
            dict_to_save[self.model_wrapper.name + "_optimizer_state_dict"] = self.model_wrapper.optimizer.state_dict()

        # final save
        torch.save(dict_to_save, save_path)

    def train_loop(self, epoch):
        raise NotImplementedError

    def train(self, epoch):
        """
        Train for one epoch
        """
        self.model_wrapper.to_train_mode(epoch=epoch)

        # Train over the entire epoch
        self.train_loop(epoch)

        # Eval on epoch end
        logs_dict = OrderedDict(
            {
                "epoch": {"value": epoch, "string": f"{epoch}"},
            }
        )
        scores = self.model_wrapper.get_dict_to_scores(split="train")
        for s in scores:
            logs_dict[s] = scores[s]

        ## Val scores
        if self.dloader.val_loader is not None:
            val_scores = self.evaluate(
                inference_loader=self.dloader.val_loader,
                split="val")
            for val_score in val_scores:
                logs_dict[val_score] = val_scores[val_score]

        ## Test scores
        test_scores = self.evaluate(
            inference_loader=self.dloader.test_loader,
            split="test")
        for test_score in test_scores:
            logs_dict[test_score] = test_scores[test_score]

        ## Print metrics
        misc.print_dict(logs_dict)

        ## Check if best epoch
        is_best_epoch = False
        ens_acc = float(logs_dict["test/accuracy"]["value"])
        if ens_acc >= self._best_acc:
            self._best_acc = ens_acc
            self._best_epoch = epoch
            is_best_epoch = True

        ## Save the model checkpoint
        ## and not config.cfg.DEBUG
        if is_best_epoch:
            logs_dict["general/checkpoint_saved"] = {"value": 1.0, "string": "1.0"}
            save_epoch = True
        else:
            logs_dict["general/checkpoint_saved"] = {"value": 0.0, "string": "0.0"}
            save_epoch = (epoch % config.cfg.SAVE_EVERY_X_EPOCH == 0)

        if save_epoch:
            self.save_checkpoint(epoch)
            LOGGER.warning(f"Epoch: {epoch} was saved")

        ## CSV logging
        short_logs_dict = OrderedDict(
            {k: v for k, v in logs_dict.items()
             if any([regex in k for regex in [
                 "test/accuracy",
                 "train/accuracy",
                 "epoch",
                 "checkpoint_saved"
                 ]])
            })
        misc.csv_writter(
            path=misc.get_logs_path(self.config_args["training"]["output_folder"]),
            dic=short_logs_dict
        )
        # Tensorboard logging
        if not config.cfg.DEBUG:
            self.save_tb(logs_dict, epoch=epoch)

        # Perform end of step procedure like scheduler update
        self.model_wrapper.scheduler.step()

    def evaluate_loop(self, dloader, verbose, **kwargs):
        raise NotImplementedError

    def evaluate(self, inference_loader, split="test"):
        """
        Perform an evaluation of the model
        """
        # Restart stats
        self.model_wrapper.to_eval_mode()

        # Evaluation over the dataset properly speaking
        self.evaluate_loop(inference_loader)

        # Gather scores
        scores = self.model_wrapper.get_dict_to_scores(split=split)

        return scores
