"""
Calibration and image printing utility functions
"""

import matplotlib.pyplot as plt
import torchvision
import numpy as np

__all__ = ['make_image', 'show_batch', 'write_calibration']


def write_calibration(
    avg_confs_in_bins,
    acc_in_bin_list,
    prop_bin,
    min_bin,
    max_bin,
    min_pred=0,
    write_file=None,
    suffix=""
):
    """
    Utility function to show calibration through a histogram of classifier confidences
    """
    fig, ax1 = plt.subplots()

    ax1.plot([min_pred, 1], [min_pred, 1], "k:", label="Perfectly calibrated")

    ax1.plot(avg_confs_in_bins,
             acc_in_bin_list,
             "s-",
             label="%s" % ("Discriminator Calibration"))

    if write_file:
        suffix = write_file.split("/")[-1].split(".")[0].split("_")[0] + "_" + suffix

    ax1.set_xlabel(f"Mean predicted value {suffix}")
    ax1.set_ylabel("Accuracy")
    ymin = min(acc_in_bin_list + [min_pred])
    ax1.set_ylim([ymin, 1.0])
    ax1.legend(loc="lower right")

    ax2 = ax1.twinx()
    ax2.hlines(prop_bin, min_bin, max_bin, label="%s" % ("Proportion in each bin"), color="r")
    ax2.set_ylabel("Proportion")
    ax2.legend(loc="upper center")

    if not write_file:
        plt.tight_layout()
        plt.show()
    else:
        fig.savefig(
            write_file
        )


def make_image(img, mean=None, std=None, normalize=True):
    """
    Transform a CIFAR numpy image into a pytorch image (need to swap dimensions)
    """
    if mean is None and std is None:
        from mixmo.augmentations.standard_augmentations import cifar_mean, cifar_std
        mean = cifar_mean
        std = cifar_std
    npimg = img.numpy().copy()
    if normalize:
        for i in range(0, 3):
            npimg[i] = npimg[i] * std[i] + mean[i]    # unnormalize
    return np.transpose(npimg, (1, 2, 0))


def show_batch(images, normalize=True):
    """
    Plot images in a batch of images
    """
    images = make_image(torchvision.utils.make_grid(images), normalize=normalize)
    plt.imshow(images)
    plt.show()
    return images
