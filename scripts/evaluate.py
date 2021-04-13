import matplotlib
matplotlib.use('Agg')

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import os
import argparse
import torch

from mixmo.loaders import get_loader
from mixmo.learners.learner import Learner
from mixmo.core import metrics_wrapper
from mixmo.utils import misc, config, logger

LOGGER = logger.get_logger(__name__, level="DEBUG")


def parse_args():
    parser = argparse.ArgumentParser()

    # shared params
    parser.add_argument("--config_path", "-c", type=str, default=None, help="Path for config yaml", required=True)
    parser.add_argument("--dataplace", "-dp", type=str, default=None, help="Parent folder to data", required=True)
    parser.add_argument(
        "--saveplace", "-sp", type=str, default=None,
        help="Parent folder to save. Not required when checkpoint is directly provided.", required=False)
    parser.add_argument(
        "--gpu", "-g", default="0", type=str, help="Selecting gpu. If not exists, then cpu"
    )
    parser.add_argument(
        "--debug", type=int, default=0, help="Debug mode: 0, 1 or 2. The more the more debug."
    )

    # specific params
    parser.add_argument(
        "--checkpoint", type=str, default="best",
        help="Can be either: 1. 'best' and we will select the best epoch 2. an int that indicates the epoch 3. A precise path to checkpoint.")

    parser.add_argument(
        "--corruptions",
        default=False,
        action='store_true',
        help="Apply corruptions from Benchmarking Neural Network corruptions to Common Corruptions and Perturbations",
        # https://github.com/hendrycks/corruptions
    )
    parser.add_argument(
        "--tempscale",
        default=False,
        action='store_true',
        help="Apply tempscale"
    )

    # parse
    args = parser.parse_args()
    misc.print_args(args)
    return args



def transform_args(args):
    # shared transform
    config_args = misc.load_config_yaml(args.config_path)
    config.cfg.DEBUG = args.debug
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # recover epoch and checkpoint
    if args.checkpoint == "best" or misc.is_int(args.checkpoint):
        assert args.saveplace is not None
        checkpoint = misc.get_checkpoint(
            epoch=args.checkpoint,
            output_folder=misc.get_output_folder_from_config(
                saveplace=args.saveplace, config_path=args.config_path
            )
        )
    elif os.path.exists(args.checkpoint):
        checkpoint = args.checkpoint
    else:
        raise ValueError(
            f"args.checkpoint: {args.checkpoint} should be either 'best', an int or a path to a previously trained checkpoint"
        )

    # seed
    misc.set_determ(config.cfg.RANDOM.SEED)
    return config_args, device, checkpoint


def evaluate(config_args, device, checkpoint, tempscale, dataplace, corruptions=False):
    assert not (tempscale and corruptions)

    # Load dataset
    dloader = get_loader(
        config_args, split_test_val=tempscale, dataplace=dataplace, corruptions=corruptions
    )

    # Load learner
    learner = Learner(
        config_args=config_args,
        dloader=dloader,
        device=device,
    )

    # Load trained model
    assert checkpoint is not None
    learner.load_checkpoint(checkpoint, include_optimizer=False)

    # calibration ?
    if not tempscale:
        LOGGER.debug("No calibration via temperature scaling")
        temp = 1.
        scores = learner.evaluate(
            inference_loader=dloader.test_loader,
            split="final",
        )
    else:
        # temperature scaling
        LOGGER.warning("First, temperature scaling on val and evaluation on test")
        tempval = learner.model_wrapper.calibrate_via_tempscale(dloader.val_loader)
        scores_test = learner.evaluate(
            inference_loader=dloader.test_loader,
            split="test",
        )
        misc.print_dict(scores_test)
        LOGGER.warning("Second, temperature scaling on test and evaluation on val")
        temptest = learner.model_wrapper.calibrate_via_tempscale(dloader.test_loader)
        scores_val = learner.evaluate(
            inference_loader=dloader.val_loader,
            split="val",
        )
        misc.print_dict(scores_val)

        scores = metrics_wrapper.merge_scores(scores_test, scores_val)
        temp = 0.5 * (tempval + temptest)

    # Get scores

    scores["temperature"] = {"value": temp, "string": f"{temp:.6}"}

    LOGGER.info("Final results")
    misc.print_dict(scores)
    metrics_wrapper.show_metrics(scores)
    return scores


def main_evaluate():
    args = parse_args()
    config_args, device, best_checkpoint = transform_args(args)
    evaluate(
        config_args,
        device,
        checkpoint=best_checkpoint,
        tempscale=args.tempscale,
        corruptions=args.corruptions,
        dataplace=args.dataplace
    )


if __name__ == "__main__":
    main_evaluate()
