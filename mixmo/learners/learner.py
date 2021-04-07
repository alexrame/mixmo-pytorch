from tqdm import tqdm

import torch

from mixmo.utils import logger, config
from mixmo.learners import abstract_learner

LOGGER = logger.get_logger(__name__, level="DEBUG")

class Learner(abstract_learner.AbstractLearner):
    """
    Learner object that defines the specific train and test loops for the model
    """
    def _subloop(self, dict_tensors, backprop):
        """
        Basic subloop for a step/batch (without optimization)
        """

        # Format input
        input_model = {"pixels": dict_tensors["pixels"]}
        if "metadata" in dict_tensors:
            input_model["metadata"] = dict_tensors["metadata"]

        # Forward pass
        output_network = self.model_wrapper.predict(
            input_model)

        # Compute loss, backward and metrics
        self.model_wrapper.step(
            output=output_network,
            target=dict_tensors["target"],
            backprop=backprop,
        )

        return self.model_wrapper.get_short_logs()

    def _train_subloop(self, dict_tensors,):
        """
        Complete training step for a batch, return summary logs
        """
        # Reset optimizers
        self.model_wrapper.optimizer.zero_grad()
        # Backprop
        dict_to_log = self._subloop(dict_tensors, backprop=True)
        # Optimizer step
        self.model_wrapper.optimizer.step()

        return dict_to_log

    def train_loop(self, epoch):
        """
        Training loop for one epoch
        """
        # Set loop counter for the epoch
        loop = tqdm(self.dloader.train_loader, dynamic_ncols=True)

        # Loop over all samples in the train set
        for batch_id, data in enumerate(loop):
            loop.set_description(f"Epoch {epoch}")

            # Prepare the batch
            dict_tensors = self._prepare_batch_train(data)

            # Perform the training step for the batch
            dict_to_log = self._train_subloop(dict_tensors=dict_tensors)
            del dict_tensors

            # Tie up end of step details
            loop.set_postfix(dict_to_log)
            loop.update()
            if config.cfg.DEBUG >= 2 and batch_id >= 10:
                break
            if self.model_wrapper.warmup_scheduler is not None:
                self.model_wrapper.warmup_scheduler.step()

    def evaluate_loop(self, inference_loader):
        """
        Evaluation loop over the dataset specified by the loader
        """
        # Set loop counter for the loader/dataset
        loop = tqdm(inference_loader, disable=False, dynamic_ncols=True)

        # Loop over all samples in the evaluated dataset
        for batch_id, data in enumerate(loop):
            loop.set_description(f"Evaluation")

            # Prepare the batch
            dict_tensors = self._prepare_batch_test(data)

            # Forward over the batch, stats are logged internally
            with torch.no_grad():
                _ = self._subloop(dict_tensors, backprop=False)

            if config.cfg.DEBUG >= 2 and batch_id >= 10:
                break

    def _prepare_batch_train(self, data):
        """
        Prepares the train batch by setting up the input dictionary and putting tensors on devices
        """
        dict_tensors = {"pixels": [], "target": {}}

        # Concatenate inputs along channel dimension and collect targets
        for num_member in range(self.config_args["num_members"]):
            dict_tensors["pixels"].append(data["pixels_" + str(num_member)])
            dict_tensors["target"]["target_" + str(num_member)] = data[
                "target_" + str(num_member)].to(self.device)
        dict_tensors["pixels"] = torch.cat(dict_tensors["pixels"], dim=1).to(self.device)

        # Pass along batch metadata
        dict_tensors["metadata"] = data.get("metadata", {})
        dict_tensors["metadata"]["mode"] = "train"

        return dict_tensors

    def _prepare_batch_test(self, data):
        """
        Prepares the test batch by setting up the input dictionary and putting tensors on devices
        """
        (pixels, target) = data
        dict_tensors = {
            "pixels": pixels.to(self.device),
            "target": {
                "target_" + str(num_member): target.to(self.device)
                for num_member in range(self.config_args["num_members"])
            },
            "metadata": {
                "mode": "inference"
            }
        }

        return dict_tensors
