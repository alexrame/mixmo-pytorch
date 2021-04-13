"""
Utility functions for random sampling, type checking, dictionary handling,
checkpoint parsing and configuration
"""

import csv
from csv import DictWriter
import os
import random
import numpy as np
import yaml
import torch


from mixmo.utils.logger import get_logger


LOGGER = get_logger(__name__, level="DEBUG")


def sample_lams(beta, n=2, seed=None):
    """
    Sampling lam ratios for mixing from a symmetric Dirichlet distribution
    """
    if beta <= 0:
        return [1/n for _ in range(n)]
    return get_nprandom(seed).dirichlet((float(beta), ) * n)


def random_lower_than(prob, seed=None, r=None):
    """
    Uniform probability check
    """
    if prob <= 0:
        return False
    if r is None:
        r = get_random(seed).uniform(0, 1)
    return r < prob

def get_random(seed):
    """
    (Possibly seeded) random.Random instantiation
    """
    if seed is not None:
        myrandom = random.Random(seed)
    else:
        myrandom = random
    return myrandom


def get_nprandom(seed):
    """
    (Possibly seeded) np.random instantiation
    """
    if seed is not None:
        mynprandom = np.random.default_rng(seed=seed)
    else:
        mynprandom = np.random
    return mynprandom



def is_nan(num):
    return num != num


def is_none(num):
    return num in ["None", "none", "null", None]


def is_zero(num):
    return num  in ["None", "none", "null", None, 0, 0.0, False, "0"]


def is_float(num):
    try:
        float(num)
        return True
    except:
        return False


def is_int(num):
    try:
        output = is_float(num) and (int(float(num)) == float(num))
        return output
    except:
        return False


def set_determ(seed):
    """
    Seeding function for reproducibility
    """
    LOGGER.warning(f"Set seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True


def clean_startswith(string, regex):
    if not isinstance(string, str):
        return False
    return string.startswith(regex)





def ifnotfound_update(old_dict, new_dict):
    update(old_dict, new_dict, method="ifnotfound")

def clean_update(old_dict, new_dict):
    update(old_dict, new_dict, method="clean")
    return old_dict


def dirty_update(old_dict, new_dict):
    update(old_dict, new_dict, method="dirty")

def update(old_dict, new_dict, method=None):
    """
    Update a dictionary in three possible ways depending on situation
    """
    for key, value in new_dict.items():
        if method == "clean":
            assert key not in old_dict, key
        elif method == "dirty":
            assert key in old_dict, key
        elif method == "ifnotfound":
            if key in old_dict:
                continue
        old_dict[key] = value


def csv_writter(path, dic):
    """
    Utility function to write dictionaries to a csv file
    """
    # Check if the file already exists
    if os.path.exists(path):
        append_mode = True
        rw_mode = "a"
    else:
        append_mode = False
        rw_mode = "w"

    # Write dic
    if append_mode is False:
        field_names = dic.keys()
    else:
        with open(path, "r") as f:
            reader = csv.reader(f)
            for field_names in reader:
                break

    dict_of_elem = {k: v["string"] for k, v in dic.items() if k in field_names}

    with open(path, rw_mode) as csvfile:
        dict_writer = DictWriter(csvfile, fieldnames=field_names, delimiter=",")
        if not append_mode:
            dict_writer.writeheader()
        # Do not write header in append mode
        dict_writer.writerow(dict_of_elem)


def _print_dict(_dict):
    """
    function that print args dictionaries in a beautiful way
    """
    print("\n" + "#" * 40)
    col_width = max(len(str(word)) for word in _dict) + 2
    for arg in sorted(list(_dict.keys())):
        str_print = str(_dict[arg])
        _str = "".join([str(arg).ljust(col_width), str_print])
        print(_str)
    print("#" * 40 + "\n")


def print_dict(logs_dict):
    _dict = {arg: logs_dict[arg]["string"] for arg in logs_dict}
    _print_dict(_dict)

def print_args(args):
    _dict = args if isinstance(args, dict) else args.__dict__
    _print_dict(_dict)


def load_yaml(path):
    with open(path, "r") as f:
        config_args = yaml.load(f, Loader=yaml.SafeLoader)
    return config_args


def load_config_yaml(path):
    config_args = load_yaml(path)
    config_args["training"]["config_path"] = path
    return config_args


def find_best_epoch(criteria, logs_file, minepoch=0):
    """
    Check the log files for the best epoch according to a specific criterion
    """
    if criteria == "last":
        key = "epoch"
        sorting = "max"
    else:
        key, sorting = criteria.split(":")
    if sorting not in ["max", "min"]:
        raise ValueError(sorting)

    best = 0
    final_epoch = 0
    best_epoch = None
    LOGGER.info("logs_file: {logs_file}".format(logs_file=logs_file))
    with open(logs_file, "r") as f_stats:
        reader = csv.DictReader(f_stats)
        for irow, row in enumerate(reader):
            try:
                new_logs = {k: str(v) for k, v in row.items()}
                assert key in new_logs
                value = new_logs[key]

                if "epoch" in new_logs:
                    epoch = new_logs['epoch']
                else:
                    epoch = irow + 1

                if is_int(epoch):
                    final_epoch = max(int(epoch), final_epoch)
                if not int(float(new_logs.get("general/checkpoint_saved", 0))):
                    continue
                if int(epoch) < int(minepoch):
                    continue
                if value.endswith("%"):
                    value = value[:-1]
                value = eval(value)
                if best_epoch is None:
                    best_epoch = epoch
                    best = value
                elif sorting == "max":
                    if best <= value:
                        best_epoch = epoch
                        best = value
                elif sorting == "min":
                    if best >= value:
                        best_epoch = epoch
                        best = value
            except KeyboardInterrupt:
                raise
            except Exception:
                LOGGER.info(new_logs, exc_info=True)
    LOGGER.warning(f"Best epoch: {best_epoch} until: {final_epoch}")
    return int(best_epoch)



def get_output_folder_from_config(saveplace, config_path):
    config_name = os.path.split(os.path.splitext(config_path)[0])[-1]
    return os.path.join(saveplace, config_name)


PREFIX_CHECKPOINTS = "checkpoint_epoch"

def get_previous_ckpt(output_folder):
    list_previous_epochs = sorted(
        [
            int(f.split(".")[0].split("_")[-1])
            for f in os.listdir(output_folder)
            if PREFIX_CHECKPOINTS in f
        ]
    )

    if not list_previous_epochs:
        return None
    last_ckpt = get_model_path(output_folder, epoch=list_previous_epochs[-1])
    return last_ckpt


def get_model_path(output_folder, epoch):
    ckpt_path = os.path.join(output_folder, PREFIX_CHECKPOINTS + f"_{epoch:03d}.ckpt")
    return ckpt_path


def get_checkpoint(output_folder, epoch):
    if epoch == "best":
        logs_file = get_logs_path(output_folder)
        epoch = find_best_epoch(criteria="test/accuracy:max", logs_file=logs_file)
    else:
        assert is_int(epoch)
        epoch = int(epoch)

    checkpoint = get_model_path(output_folder, epoch=epoch)
    LOGGER.warning(f"Best checkpoint: {checkpoint}")
    return checkpoint


def get_logs_path(output_folder):
    logs_path = os.path.join(output_folder,  "logs.csv")
    return logs_path
