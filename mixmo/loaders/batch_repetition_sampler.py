"""
Sampler definition for multi-input models
"""
import torch
import random

from mixmo.utils import (
    config, torchutils, logger, misc)

LOGGER = logger.get_logger(__name__, level="DEBUG")


class BatchRepetitionSampler(torch.utils.data.sampler.BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of repeated indices.
    """
    def __init__(
        self,
        sampler,
        batch_size,
        num_members,
        config_batch_sampler,
        drop_last=False,
    ):
        torch.utils.data.sampler.BatchSampler.__init__(self, sampler, batch_size, drop_last)

        self.num_members = num_members
        self._batch_repetitions = config_batch_sampler["batch_repetitions"]
        self._proba_input_repetition = config_batch_sampler["proba_input_repetition"]

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            for _ in range(self._batch_repetitions):
                batch.append(idx)
            if len(batch) >= self.batch_size:
                yield self.output_format(batch)
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield self.output_format(batch)

    def output_format(self, std_batch):
        """
        Transforms standards batches into batches of sample summaries
        """
        # Create M shuffled batches, one for each input
        batch_size = len(std_batch)
        list_shuffled_index = [
            torchutils.randperm_static(batch_size, proba_static=self._proba_input_repetition)
            for _ in range(self.num_members)
        ]

        shuffled_batch = [
            std_batch[list_shuffled_index[0][count]]
            for count in range(batch_size)]

        # sample batch seed, shared among samples from the given batch
        batch_seed = random.randint(0, config.cfg.RANDOM.MAX_RANDOM)
        list_index = [
            misc.clean_update(
                {
                    "batch_seed": batch_seed,
                    "index_" + str(0): shuffled_batch[count]
                }, {
                    "index_" + str(num_member):
                    shuffled_batch[list_shuffled_index[num_member][count]]
                    for num_member in range(1, self.num_members)
                }
            )
            for count in range(batch_size)
        ]

        return list_index

    def __len__(self):
        len_sampler = len(self.sampler) * self._batch_repetitions
        if self.drop_last:
            return len_sampler // self.batch_size
        else:
            return (len_sampler + self.batch_size - 1) // self.batch_size
