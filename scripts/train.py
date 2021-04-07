import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import argparse
import os
from shutil import copyfile, rmtree
import click
import torch

from mixmo.loaders import get_loader
from mixmo.learners.learner import Learner
from mixmo.utils import (misc, config, logger)
from scripts.evaluate import evaluate


LOGGER = logger.get_logger(__name__, level="DEBUG")


def parse_args():
    parser = argparse.ArgumentParser()

    # shared params
    parser.add_argument("--config_path", "-c", type=str, default=None, help="Path for config yaml", required=True)
    parser.add_argument("--dataplace", "-dp", type=str, default=None, help="Parent folder to data", required=True)
    parser.add_argument("--saveplace", "-sp", type=str, default=None, help="Parent folder to save", required=True)
    parser.add_argument("--gpu", "-g", default="0", type=str, help="Selecting gpu. If not exists, then cpu")
    parser.add_argument("--debug", type=int, default=0, help="Debug mode: 0, 1 or 2. The more the more debug.")

    # specific params
    parser.add_argument(
        "--from_scratch",
        "-f",
        action="store_true",
        default=False,
        help="Force training from scratch",
    )
    parser.add_argument(
        "--seed",
        default=config.cfg.RANDOM.SEED,
        type=int,
        help="Random seed",
    )

    # parse
    args = parser.parse_args()
    misc.print_args(args)
    return args

def transform_args(args):
    # shared transform
    config_args = misc.load_config_yaml(args.config_path)
    config_args["training"]["output_folder"] = misc.get_output_folder_from_config(
        saveplace=args.saveplace, config_path=args.config_path
    )
    config.cfg.DEBUG = args.debug
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # recover epoch and checkpoints
    ckpt_path = None
    if os.path.exists(config_args["training"]["output_folder"]):
        if not args.from_scratch:
            ckpt_path = misc.get_previous_ckpt(config_args["training"]["output_folder"])
        else:
            rmtree(config_args["training"]["output_folder"])
    if ckpt_path is not None:
        LOGGER.warning(f"Last checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        start_epoch = checkpoint["epoch"] + 1
    else:
        os.mkdir(config_args["training"]["output_folder"])
        LOGGER.info("Starting from scratch")
        checkpoint = None
        start_epoch = 1
    copyfile(
        args.config_path,
        os.path.join(config_args["training"]["output_folder"], f"config_{start_epoch}.yaml")
    )

    # seed
    config.cfg.RANDOM.SEED = args.seed - 1 + start_epoch
    misc.set_determ(config.cfg.RANDOM.SEED)
    return config_args, start_epoch, device, checkpoint


def train(config_args, start_epoch, device, checkpoint, dataplace):

    # Load dataset
    LOGGER.info(f"Loading dataset {config_args['data']['dataset']}")
    dloader = get_loader(config_args, dataplace=dataplace)

    # Set learner
    learner = Learner(
        config_args=config_args,
        dloader=dloader,
        device=device,
    )

    # Resume existing model or from pretrained one
    if start_epoch > 1:
        assert checkpoint is not None
        learner.load_checkpoint(checkpoint, include_optimizer=True)

    LOGGER.info(f"Saving logs in: {config_args['training']['output_folder']}")

        # Start training
    _config_name = os.path.split(
        os.path.splitext(config_args['training']['config_path'])[0])[-1]

    try:
        for epoch in range(start_epoch, config_args["training"]["nb_epochs"] + 1):
            LOGGER.debug(f"Epoch: {epoch} for: {_config_name}")
            learner.dloader.traindatasetwrapper.set_ratio_epoch(
                ratioepoch=epoch / config_args["training"]["nb_epochs"]
            )
            learner.train(epoch)

    except KeyboardInterrupt:
        LOGGER.warning(f"KeyboardInterrupt for: {_config_name}")
        if not click.confirm("continue ?", abort=False):
            raise KeyboardInterrupt

    except Exception as exc:
        LOGGER.error(f"Exception for: {_config_name}")
        raise exc

    return epoch


def main_train():
    # train
    args = parse_args()
    config_args, start_epoch, device, checkpoint = transform_args(args)
    if start_epoch < int(config_args["training"]["nb_epochs"]) + 1:
        train(config_args, start_epoch, device, checkpoint, dataplace=args.dataplace)

    # test at best epoch
    best_checkpoint = misc.get_checkpoint(
        output_folder=config_args['training']['output_folder'],
        epoch="best",
    )
    evaluate(
        config_args=config_args,
        device=device,
        checkpoint=best_checkpoint,
        tempscale=False,
        corruptions=False,
        dataplace=args.dataplace
    )
    LOGGER.error(f"Finish: {config_args['training']['config_path']}")


if __name__ == "__main__":
    main_train()
