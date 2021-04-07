import os
import copy
import argparse
import datetime
from shutil import rmtree
from mixmo.utils import (misc, logger)

LOGGER = logger.get_logger(__name__, level="DEBUG")


DICT_NETWORK = {
    # Tiny ImageNet with PreActResNet-18-w
    "res18": {
        "classifier": "resnet",
        "depth": 18,
        "widen_factor": 1,
        "num_members": 1,
    },
    "res18-2": {
        "classifier": "resnetmixmo",
        "depth": 18,
        "widen_factor": 1,
        "num_members": 2,
    },
    "res182": {
        "classifier": "resnet",
        "depth": 18,
        "widen_factor": 2,
        "num_members": 1,
    },
    "res182-2": {
        "classifier": "resnetmixmo",
        "depth": 18,
        "widen_factor": 2,
        "num_members": 2,
    },
    "res183": {
        "classifier": "resnet",
        "depth": 18,
        "widen_factor": 3,
        "num_members": 1,
    },
    "res183-2": {
        "classifier": "resnetmixmo",
        "depth": 18,
        "widen_factor": 3,
        "num_members": 2,
    },

    # CIFAR with WideResNet-28-10
    "wrn2810": {
        "classifier": "wideresnet",
        "depth": 28,
        "widen_factor": 10,
        "num_members": 1,
    },
    "wrn2810-2": {
        "classifier": "wideresnetmixmo",
        "depth": 28,
        "widen_factor": 10,
        "num_members": 2,
    },
    "wrn2810-3": {
        "classifier": "wideresnetmixmo",
        "depth": 28,
        "widen_factor": 10,
        "num_members": 3,
    },
}


DICT_DATASET_CONFIG = {
    "cifar10": {
        "shared_config": {
            "num_classes": 10,
            "dataset_name": "cifar",
        },
        "templates": [
            {
                "networktype":
                "wrn2810",
                "trainingfiltering": [
                    {
                        "mixmoparams": ["1net"],
                        "dataaugparams": ["standard", "msdamixup", "msdacutmix"],
                        "scheduling": ["bar1"],
                    },
                ]
            },
            {
                "networktype":
                "wrn2810-2",
                "trainingfiltering": [
                    {
                        "mixmoparams": ["mimo", "linearmixmo", "cutmixmo-p5"],
                        "dataaugparams": ["standard"],
                        "scheduling": ["bar4"],
                    },
                    {
                        "mixmoparams": ["linearmixmo", "cutmixmo-p5"],
                        "dataaugparams": ["msdacutmix"],
                        "scheduling": ["bar4"],
                    },
                ]
            },
        ],
    },
    "cifar100": {
        "shared_config": {
            "num_classes": 100,
            "dataset_name": "cifar",
        },
        "templates": [
            {
                "networktype":
                "wrn2810",
                "trainingfiltering": [
                    {
                        "mixmoparams": ["1net"],
                        "dataaugparams": ["standard", "msdamixup", "msdacutmix"],
                        "scheduling": ["bar1"],
                    },
                ]
            },
            {
                "networktype":
                "wrn2810-2",
                "trainingfiltering": [
                    {
                        "mixmoparams": ["mimo", "linearmixmo", "cutmixmo-p5"],
                        "dataaugparams": ["standard"],
                        "scheduling": ["bar4"],
                    },
                    {
                        "mixmoparams": ["linearmixmo", "cutmixmo-p5"],
                        "dataaugparams": ["msdacutmix"],
                        "scheduling": ["bar4"],
                    },
                ]
            },
        ],
    },
    "tinyimagenet": {
        "shared_config": {
            "num_classes": 200,
            "dataset_name": "tinyimagenet",
        },
        "templates": [
            {
                "networktype":
                "res18",
                "trainingfiltering": [
                    {
                        "mixmoparams": ["1net"],
                        "dataaugparams": ["standard", "msdamixup", "msdacutmix"],
                        "scheduling": ["bar1"],
                    },
                ]
            },
            {
                "networktype":
                "res182",
                "trainingfiltering": [
                    {
                        "mixmoparams": ["1net"],
                        "dataaugparams": ["standard"],
                        "scheduling": ["bar1"],
                    },
                ]
            },
            {
                "networktype":
                "res183",
                "trainingfiltering": [
                    {
                        "mixmoparams": ["1net"],
                        "dataaugparams": ["standard"],
                        "scheduling": ["bar1"],
                    },
                ]
            },
            {
                "networktype":
                "res182-2",
                "trainingfiltering": [
                    {
                        "mixmoparams": ["linearmixmo", "cutmixmo-p5"],
                        "dataaugparams": ["standard"],
                        "scheduling": ["bar4"],
                    },
                ]
            },
            {
                "networktype":
                "res183-2",
                "trainingfiltering": [
                    {
                        "mixmoparams": ["linearmixmo", "cutmixmo-p5"],
                        "dataaugparams": ["standard"],
                        "scheduling": ["bar4"],
                    },
                ]
            }
        ],
    },
}

DICT_CONFIG = {
    "scheduling": {
        "tinyimagenet": {
            "_default": {
                "nb_epochs": 1200,
                "batch_size": 100,

                # regularization
                "weight_decay_sgd": 1e-4,
                "l2_reg": 0,

                # lr
                "milestone1": 600,
                "milestone2": 900,
                "milestone3": -1,
            },
            # tinymagenet
            "bar1": {
                "batch_repetitions": 1,
                "warmup_period": 1 * 1000,
                "lrinit": 0.2 / 1,
            },
            "bar2": {
                "batch_repetitions": 2,
                "warmup_period": 2 * 1000,
                "lrinit": 0.2 / 2,
            },
            "bar4": {
                "batch_repetitions": 4,
                "warmup_period": 4 * 1000,
                "lrinit": 0.2 / 4,
            },
        },
        "cifar": {
            "_default": {
                "nb_epochs": 300,
                "batch_size": 64,

                # lr
                "weight_decay_sgd": 0,
                "l2_reg": 0.0003,

                # lrsche
                "milestone1": 101,
                "milestone2": 201,
                "milestone3": 226,
            },
            "bar1": {
                "warmup_period": 782 * 1,
                "batch_repetitions": 1,
                "lrinit": 0.1 / (2 * 1),
            },
            "bar2": {
                "warmup_period": 782 * 2,
                "batch_repetitions": 2,
                "lrinit": 0.1 / (2 * 2),
            },
            "bar4": {
                "warmup_period": 782 * 4,
                "batch_repetitions": 4,
                "lrinit": 0.1 / (2 * 4),
            },
        }
    },
    "mixmoparams": {
        "_default": {
            "mixmo_mix_method_name": "null",
            "mixmo_mix_prob": 1,
            "mixmo_alpha": 2,
            "mixmo_weight_root": 3
        },
        "1net": {},
        "mimo": {
            "mixmo_mix_method_name": "mixup",
            "mixmo_alpha": 0,
            "mixmo_weight_root": 1,
        },
        "linearmixmo": {
            "mixmo_mix_method_name": "mixup",
        },
        "cutmixmo-p5": {
            "mixmo_mix_method_name": "cutmix",
            "mixmo_mix_prob": 0.5
        },
        "cutmixmo-p5-a4": {
            "mixmo_mix_method_name": "cutmix",
            "mixmo_mix_prob": 0.5,
            "mixmo_alpha": 4,
        },
        "cutmixmo-p5-r1": {
            "mixmo_mix_method_name": "cutmix",
            "mixmo_mix_prob": 0.5,
            "mixmo_weight_root": 1,
        },
        "cutmixmo-p2": {
            "mixmo_mix_method_name": "cutmix",
            "mixmo_mix_prob": 0.2
        },
        "cowmixmo-p5": {
            "mixmo_mix_method_name": "cow",
            "mixmo_mix_prob": 0.5
        },
    },
    "dataaugparams": {
        "_default": {
            "msda_mix_method": "null",
            "da_method": "null",
        },
        "standard": {},
        "daaugmix": {
            "da_method": "augmix"
        },
        "msdamixup": {
            "msda_mix_method": "mixup",
        },
        "msdacutmix": {
            "msda_mix_method": "cutmix",
        },
    }
}


def use_template(
    template_path,
    output_path,
    params,
):
    """
    Open a template file and fill it with the vars in params
    to write the result in output_path
    """
    with open(template_path, 'r') as f_template:
        template = f_template.read()

    content = template % params

    with open(output_path, 'w') as f_out:
        f_out.write(content)



def create_templates(template_path, config_dir, dataset):
    if os.path.exists(config_dir):
        LOGGER.debug("Folder templates already exists")
        rmtree(config_dir)
    os.mkdir(config_dir)

    template_output_path = os.path.join(
        config_dir,
        "exp_{dataset}_{networktype}_{mixmoparams}_{dataaugparams}_{scheduling}.yaml"
    )

    for dict_template in DICT_DATASET_CONFIG[dataset]["templates"]:
        params = copy.deepcopy(DICT_DATASET_CONFIG[dataset]["shared_config"])
        params.update(DICT_NETWORK[dict_template["networktype"]])
        save_params = copy.deepcopy(params)

        for trainingfiltering in dict_template["trainingfiltering"]:
            for imixmo in trainingfiltering["mixmoparams"]:
                for idataaug in trainingfiltering["dataaugparams"]:
                    for ische in trainingfiltering["scheduling"]:
                        misc.clean_update(
                            params, DICT_CONFIG["mixmoparams"]["_default"]
                        )
                        misc.update(params, DICT_CONFIG["mixmoparams"][imixmo], method="dirty")
                        misc.clean_update(
                            params, DICT_CONFIG["dataaugparams"]["_default"]
                        )
                        misc.update(params, DICT_CONFIG["dataaugparams"][idataaug], method="dirty")

                        misc.clean_update(
                            params, DICT_CONFIG["scheduling"][params['dataset_name']]["_default"]
                        )
                        misc.clean_update(
                            params, DICT_CONFIG["scheduling"][params['dataset_name']][ische]
                        )
                        # templating
                        output_path = template_output_path.format(**{
                            "dataset": dataset,
                            "networktype": dict_template["networktype"],
                            "scheduling": ische,
                            "mixmoparams": imixmo,
                            "dataaugparams": idataaug
                        })
                        if os.path.exists(output_path):
                            raise ValueError(output_path)

                        output_path = use_template(
                            template_path=template_path,
                            output_path=output_path,
                            params=params,
                        )
                        params = copy.deepcopy(save_params)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template_path",
        "-t",
        type=str,
        default="scripts/exp_mixmo_template.yaml",
        help="Path to config template"
    )
    parser.add_argument(
        "--config_dir", "-c", type=str, default="config/", help="Folder to save these new configs"
    )
    parser.add_argument(
        "--dataset",
        default="cifar100",
        help="dataset name",
    )
    args = parser.parse_args()
    misc.print_args(args)
    return args



if __name__ == "__main__":
    args = parse_args()
    create_templates(
        template_path=args.template_path,
        config_dir=args.config_dir,
        dataset=args.dataset,
    )
