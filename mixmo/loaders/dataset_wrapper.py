"""
Dataset wrappers for multi-input multi-output models with data augmentation
"""

import math
import torch
from torch.utils.data.dataset import Dataset

from mixmo.utils import misc, config, torchutils
from mixmo.augmentations import augmix, mixing_blocks


class DADataset(Dataset):
    """
    Dataset wrapper with with outputs formatted as dictionaries and AugMix augmentation
    """

    def __init__(self, dataset, num_classes, num_members, dict_config, properties):
        self.dataset = dataset
        self.num_classes = num_classes
        self.num_members = num_members
        self.dict_config = dict_config
        self.properties = properties
        self._custom_init()
        self.set_ratio_epoch(0)

    def _custom_init(self):
        pass

    def set_ratio_epoch(self, ratioepoch):
        self.ratio_epoch_current = ratioepoch

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Retrieve target and image, return a dictionary with the two
        """
        pixels_0, target_0 = self.call_dataset(index["index_0"])
        dict_output = {"pixels_0": pixels_0, "target_0": target_0}
        return dict_output

    def call_dataset(self, index, seed=None):
        """
        Get target and image, apply AugMix if necessary and return dictionary
        """
        if misc.is_none(self.dict_config["da_method"]):
            pixels, target = self.dataset[index]
        else:
            dict_pixels_postprocessing, target = self.dataset.__getitem__(
                index, apply_postprocessing=False
            )
            if self.dict_config["da_method"] == "augmix":
                pixels = augmix.AugMix(seed=seed)(
                    image=dict_pixels_postprocessing["pixels"],
                    preprocess=dict_pixels_postprocessing["postprocessing"]
                )
            else:
                raise ValueError(self.dict_config["da_method"])

        return pixels, torchutils.onehot(self.num_classes, target)


class MSDADataset(DADataset):
    """
    Dataset wrapper that returns dictionaries and applies MSDA augmentations
    """

    reverse_if_first_minor = False

    def _custom_init(self):
        self._custom_init_msda()

    def _custom_init_msda(self):
        self.msda_mix_method = self.dict_config["msda"]["mix_method"]
        self.msda_beta = self.dict_config["msda"]["beta"]
        self.msda_prob = self.dict_config["msda"]["prob"]

    def call_msda(self, index_0, mixmo_mask=None, seed_da=None):
        """
        Get two samples and mix them. Return a dictionary of sample and label
        """
        # Gather the two image/label pairs used by the augmentation
        pixels_0, target_0 = self.call_dataset(index_0, seed=seed_da)
        skip_msda = (
            self.msda_mix_method is None or not misc.random_lower_than(self.msda_prob)
        )
        if skip_msda:           # Early exit if we are not mixing
            return pixels_0, target_0

        index_1 = misc.get_random(seed=None).choice(range(len(self)))
        pixels_1, target_1 = self.call_dataset(index_1, seed=seed_da)

        targets = [target_0, target_1]

        # Get mixing masks
        msda_lams = misc.sample_lams(self.msda_beta, n=2)
        msda_masks, msda_lams = mixing_blocks.mix(
            method=self.msda_mix_method,
            lams=msda_lams,
            input_size=pixels_0.size(),
        )

        # Adjust the lams to account for later mixmo mixing that might alter masks
        if mixmo_mask is not None:
            ## approx for computational issues: mask should be symmetrical in channels
            mixmo_mask_0 = mixmo_mask[0, :, :]

            if self.properties("conv1_is_half_size"):
                _msda_mask_0 = torch.nn.AvgPool2d(kernel_size=(2, 2))(msda_masks[0][:1, :, :])
                msda_masks_for_lam = [_msda_mask_0.to(torch.float16)]
            else:
                mixmo_mask_0 = mixmo_mask_0.to(torch.float32)
                msda_masks_for_lam = msda_masks

            ## Compute the adjusted ratios after mixmo mixing
            mean_mixmo_mask_0 = mixmo_mask_0.mean()
            msda_lams = [
                (msda_mask[0, :, :] * mixmo_mask_0).mean() / (mean_mixmo_mask_0 + 1e-8)
                for msda_mask in msda_masks_for_lam
            ]

            if self.properties("conv1_is_half_size"):
                lam = msda_lams[0].to(torch.float32)
                msda_lams = [lam, 1-lam]

        # Randomly reverse the roles of mixed samples (important to symmetrize CutMix, Patch-Up, ...)
        if self.reverse_if_first_minor and msda_lams[0] < 0.5:
            msda_pixels = msda_masks[1] * pixels_0 + msda_masks[0] * pixels_1
            msda_lams = [msda_lams[1], msda_lams[0]]
        else:
            msda_pixels = msda_masks[0] * pixels_0 + msda_masks[1] * pixels_1

        # Standard MSDA label interpolation
        msda_targets = sum([
                lam * target for lam, target
                in zip(msda_lams, targets)])

        return msda_pixels, msda_targets

    def __getitem__(self, index):
        """
        Return a dictionary with the relevant sample and target, possibly mixed with another
        """
        if self.msda_mix_method is None:
            return DADataset.__getitem__(self, index)

        pixels_0, target_0 = self.call_msda(index_0 = index["index_0"])
        dict_output = {"pixels_0": pixels_0, "target_0": target_0}
        return dict_output


class MixMoDataset(MSDADataset):
    """
    Dataset wrapper that returns dictionaries of multiple samples, and applies MSDA augmentations
    """

    reverse_if_first_minor = True

    def _custom_init(self):
        self._custom_init_msda()
        self._custom_init_mixmo()

    def _custom_init_mixmo(self):
        self.dict_mixmo_mix_method = self.dict_config["mixmo"]["mix_method"]
        # dict with key 'method_name', 'prob' and 'replacement_method_name'
        self.mixmo_alpha = float(self.dict_config["mixmo"]["alpha"])
        self.mixmo_weight_root = self.dict_config["mixmo"]["weight_root"]

    def get_mixmo_mix_method_at_ratio_epoch(self, batch_seed=None):
        """
        Select which mixing method should be used according to training scheduling.

        Procedure:
        Select self.dict_mixmo_mix_method["method_name"] with proba self.dict_mixmo_mix_method["prob"] that is linearly decreased towards 0 after 11/12 of training process.
        Otherwise, use self.dict_mixmo_mix_method["replacement_method_name"] (in general mixup)
        """

        method = self.dict_mixmo_mix_method["method_name"]
        replacement_method = self.dict_mixmo_mix_method["replacement_method_name"]
        if method == replacement_method:
            return method

        # Check the actual switch probability according to scheduler and current epoch
        default_prob = self.dict_mixmo_mix_method["prob"]
        if self.ratio_epoch_current < config.cfg.RATIO_EPOCH_DECREASE:
            prob = default_prob
        else:
            eta = max(0, (1 - self.ratio_epoch_current) / (1 - config.cfg.RATIO_EPOCH_DECREASE))
            prob = default_prob * eta

        # Choose the method depending on draw result
        if misc.random_lower_than(prob, seed=batch_seed):
            return method
        return replacement_method

    def _init_dict_output_mixmo(self, batch_seed):
        """
        Compute MixMo block variables (masks, lams) and prepare it as a dictionary output
        """
        # Get MixMo mixing method and the corresponding masks/lams
        mixmo_mix_method = self.get_mixmo_mix_method_at_ratio_epoch(
            batch_seed=batch_seed
        )
        mixmo_lams = misc.sample_lams(self.mixmo_alpha, n=self.num_members)
        mixmo_masks, mixmo_lams = mixing_blocks.mix(
            method=mixmo_mix_method,
            lams=mixmo_lams,
            input_size=self.properties("conv1_input_size"),
        )

        # Shuffle the roles of the inputs (same for every sample in the batch)
        # Mostly useful for asymmetrical mixing (CutMix, ...)
        assert batch_seed is not None
        myrandom = misc.get_random(seed=batch_seed+config.cfg.RANDOM.SEED_OFFSET_MIXMO)
        zipped_masking = list(zip(mixmo_lams, mixmo_masks))
        myrandom.shuffle(zipped_masking)

        # Format everything nicely in dictionaries
        dict_output = {"metadata": {"mixmo_lams": [el[0] for el in zipped_masking], "mixmo_masks": [el[1] for el in zipped_masking]}}
        if mixmo_mix_method not in mixing_blocks.LIST_METHODS_NOT_INVARIANT_CHANNELS:
            dict_output["metadata"]["mixmo_masks"] = [
                mimo_mix_mask[:1, :, :].to(torch.float16)
                for mimo_mix_mask in dict_output["metadata"]["mixmo_masks"]]

        return dict_output

    def __getitem__(self, index):
        """
        Get a (mixed) sample/label pair for each head and output it in a dictionary
        """

        # Initialize output with mixing block descriptors
        dict_output = self._init_dict_output_mixmo(
            batch_seed=index["batch_seed"])

        # Compute sample/label pairs for each head
        for num_member in range(0, self.num_members):
            member_index = index["index_" + str(num_member)]

            # useful only for augmix: force same transfos on batch duplicated samples
            seed_da = index["batch_seed"] + config.cfg.RANDOM.SEED_DA * member_index + num_member

            ## Retrieve and compute mixed sample/label, accounting for mixmo mixing
            seed_da = index["batch_seed"] + config.cfg.RANDOM.SEED_DA * member_index + num_member

            pixels_member, target_member = self.call_msda(
                index_0=member_index,
                mixmo_mask=dict_output["metadata"]["mixmo_masks"][num_member],
                seed_da=seed_da
            )

            ## Format output
            dict_output.update({
                "pixels_" + str(num_member): pixels_member,
                "target_" + str(num_member): target_member
            })

        dict_output = self._target_balancing(dict_output)

        if self.num_members == 2:
            # only keep first to reduce memory footprint
            # as the second can be obtained by 1 - mask
            dict_output["metadata"]["mixmo_masks"] = dict_output["metadata"]["mixmo_masks"][:1]

        return dict_output

    def _target_balancing(self, dict_output):
        """
        Final formatting of outputs with mixmo balancing
        """
        def apply_root(a):
            return math.pow(a, (1 / self.mixmo_weight_root))

        # Get balancing weights
        _list_weights_not_normalized = [apply_root(lam) for lam in dict_output["metadata"]["mixmo_lams"]]
        norm = sum(_list_weights_not_normalized)
        list_weights = [self.num_members * weight / norm for weight in _list_weights_not_normalized]

        # Apply weights: as we use categorical cross entropy, we can simply multiply the target
        for i in range(self.num_members):
            dict_output["target_{}".format(i)] *= list_weights[i]

        return dict_output
