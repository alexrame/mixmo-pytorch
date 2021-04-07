"""
Mostly taken from https://github.com/bayesgroup/pytorch-ensembles/blob/master/metrics.py
"""

import numpy as np
from sklearn.metrics import roc_auc_score

from mixmo.utils import visualize
from mixmo.utils.logger import get_logger
from mixmo.core import metrics_ensemble

LOGGER = get_logger(__name__, level="INFO")



def merge_scores(scores_test, scores_val):
    """
    Aggregate scores
    """
    scores_valtest = {}
    for key in scores_test:
        key_valtest = "final/" + key.split("/")[1]
        if key.startswith("test/"):
            keyval = "val/" + key.split("/")[1]
            value = 0.5 * (scores_test[key]["value"] + scores_val[keyval]["value"])
            if scores_test[key]["string"].endswith("%"):
                value_str = f"{value:05.2%}"
            else:
                value_str = f"{value:.6}"
            stats = {"value": value, "string": value_str}
            scores_valtest[key_valtest] = stats
        else:
            scores_valtest[key_valtest] = scores_test[key]
    return scores_valtest


def _clean_metrics(metrics, output_format="float"):
    """
    Reformat metrics dictionary
    """
    new_dict = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            v = v["string"]
        if isinstance(v, str):
            if v.endswith("%"):
                v = v[:-1]
        if output_format == "float":
            v = float(v)
        new_dict[k] = v
    return new_dict

def show_metrics(scores_test):
    """
    Results printer
    """
    keys = [
        "final/accuracy",
        "final/accuracytop5",
        "final/nll",
        "final/ece",
    ]

    clean_scores_test = _clean_metrics(scores_test, output_format="str")
    our_results = [clean_scores_test[key] for key in keys]

    print(" & ".join(keys))
    print(" & ".join(our_results))


def get_ece(proba_pred, accurate, n_bins=15, min_pred=0, write_file=None, **args):
    """
    Compute ECE and write to file
    """
    if min_pred == "minpred":
        min_pred = min(proba_pred)
    else:
        assert min_pred >= 0
    bin_boundaries = np.linspace(min_pred, 1., n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    acc_in_bin_list = []
    avg_confs_in_bins = []
    list_prop_bin = []
    ece = 0.0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(proba_pred > bin_lower, proba_pred <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        list_prop_bin.append(prop_in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accurate[in_bin])
            avg_confidence_in_bin = np.mean(proba_pred[in_bin])
            delta = avg_confidence_in_bin - accuracy_in_bin
            acc_in_bin_list.append(accuracy_in_bin)
            avg_confs_in_bins.append(avg_confidence_in_bin)
            ece += np.abs(delta) * prop_in_bin
            LOGGER.debug(
                f"From {bin_lower:4.5} to {bin_upper:4.5} and mean {avg_confidence_in_bin:3.5}, {(prop_in_bin * 100):4.5} % samples with accuracy {accuracy_in_bin:4.5}"
                )
        else:
            avg_confs_in_bins.append(None)
            acc_in_bin_list.append(None)

    if write_file is not None:
        visualize.write_calibration(
            avg_confs_in_bins=avg_confs_in_bins,
            acc_in_bin_list=acc_in_bin_list,
            prop_bin=list_prop_bin,
            min_bin=bin_lowers,
            max_bin=bin_uppers,
            min_pred=min_pred,
            suffix=f"{ece:.4}",
            write_file=write_file)

    # For reliability diagrams, also need to return these:
    # return ece, bin_lowers, avg_confs_in_bins
    return ece, avg_confs_in_bins, acc_in_bin_list


def get_tace_bayesgroup(preds, targets, n_bins=15, threshold=1e-3, write_file=None, **args):
    """
    Compute TACE and write to file
    """
    n_objects, n_classes = preds.shape

    res = 0.0
    for cur_class in range(n_classes):
        cur_class_conf = preds[:, cur_class]

        targets_sorted = targets[cur_class_conf.argsort()]
        cur_class_conf_sorted = np.sort(cur_class_conf)

        targets_sorted = targets_sorted[cur_class_conf_sorted > threshold]
        cur_class_conf_sorted = cur_class_conf_sorted[cur_class_conf_sorted > threshold]

        bin_size = len(cur_class_conf_sorted) // n_bins

        for bin_i in range(n_bins):
            bin_start_ind = bin_i * bin_size
            if bin_i < n_bins-1:
                bin_end_ind = bin_start_ind + bin_size
            else:
                bin_end_ind = len(targets_sorted)
                bin_size = bin_end_ind - bin_start_ind  # extend last bin until the end of prediction array
            bin_acc = (targets_sorted[bin_start_ind : bin_end_ind] == cur_class)
            bin_conf = cur_class_conf_sorted[bin_start_ind : bin_end_ind]
            avg_confidence_in_bin = np.mean(bin_conf)
            avg_accuracy_in_bin = np.mean(bin_acc)
            delta = np.abs(avg_confidence_in_bin - avg_accuracy_in_bin)
            res += delta * bin_size / (n_objects * n_classes)

    return res


def get_ll(preds, targets, **args):
    """
    Compute log likelihood
    """
    preds_target = preds[np.arange(len(targets)), targets]
    return np.log(1e-12 + preds_target).sum()


def get_brier(preds, targets, **args):
    """
    Compute brier score
    """
    one_hot_targets = np.zeros(preds.shape)
    one_hot_targets[np.arange(len(targets)), targets] = 1.0
    return np.mean((preds - one_hot_targets) ** 2) * len(targets)


class MetricsWrapper:
    """
    Metric storing object
    """

    def __init__(self, metrics):
        self.metrics = metrics
        self.accurate_or_wrong, self.list_confidences = [], []

        # metrics
        if "nll" in self.metrics:
            self.nll = 0
        if "brier" in self.metrics:
            self.brier = 0
        if "accuracytop5" in self.metrics:
            self.num_accurate_top5 = 0
        if "tace" in self.metrics:
            LOGGER.debug("Keep all predictions to compute TACE. Can be heavy, do not use in training")
            self._list_np_probs = []
            self._list_np_targets = []
        if "diversity" in self.metrics:
            self._list_target_diversity = []
            self._list_matrix_predictions_diversity = []

    def update(self, pred, target, confidence, probs):
        """
        Compute tracked metrics and update records
        """
        self.accurate_or_wrong.extend(pred.eq(target.view_as(pred)).detach().to("cpu").numpy())
        self.list_confidences.extend(confidence.detach().to("cpu").numpy())

        np_probs = probs.detach().to("cpu").numpy()
        np_targets = target.detach().to("cpu").numpy()
        if "nll" in self.metrics:
            self.nll -= get_ll(np_probs, np_targets)
        if "brier" in self.metrics:
            brier = get_brier(np_probs, np_targets)
            self.brier += brier
        if "accuracytop5" in self.metrics:
            _, pred5 = probs.topk(5, 1, True, True)
            pred5 = pred5.t()
            correct5 = pred5.eq(target.view(1, -1).expand_as(pred5))
            correct5 = correct5[:5].view(-1).float().sum(0, keepdim=True).detach().to("cpu").numpy()[0]
            self.num_accurate_top5 += correct5
        if "tace" in self.metrics:
            self._list_np_probs.append(np_probs)
            self._list_np_targets.append(np_targets)

    def update_diversity(self, target, predictions):
        """
        Compute and update records of diversity metrics
        """
        self._list_target_diversity.extend(target)
        self._list_matrix_predictions_diversity.append(predictions)

    def get_scores(self, split="train"):
        """
        Print stored results
        """
        if not len(self.list_confidences):
            LOGGER.warning("No predictions so far")
            return {}

        accurate_or_wrong = np.reshape(self.accurate_or_wrong, newshape=(len(self.accurate_or_wrong), -1)).flatten()
        list_confidences = np.reshape(self.list_confidences, newshape=(len(self.list_confidences), -1)).flatten()

        len_dataset = len(list_confidences)
        scores = {}

        if "diversity" in self.metrics:
            diversity_stats = self._get_diversity_stats()
            for diversity_key, diversity_stat in diversity_stats.items():
                scores[f"{split}/" + diversity_key] = diversity_stat

        if "accuracy" in self.metrics:
            accuracy = np.mean(accurate_or_wrong)
            scores[f"{split}/accuracy"] = {"value": accuracy, "string": f"{accuracy:05.2%}"}

        if "nll" in self.metrics:
            nll = self.nll / len_dataset
            scores[f"{split}/nll"] = {"value": nll, "string": f"{nll:.6}"}

        if "accuracytop5" in self.metrics:
            accuracytop5 = self.num_accurate_top5 / len_dataset
            scores[f"{split}/accuracytop5"] = {"value": accuracytop5, "string": f"{accuracytop5:05.2%}"}

        if "auc" in self.metrics:
            auc = roc_auc_score(accurate_or_wrong, list_confidences)
            scores[f"{split}/auc"] = {"value": auc, "string": f"{auc:.6}"}

        if "brier" in self.metrics:
            brier = self.brier / len_dataset
            scores[f"{split}/brier"] = {"value": brier, "string": f"{brier:.6}"}

        if "ece" in self.metrics:
            ece, _, _ = get_ece(
                list_confidences, accurate_or_wrong, min_pred=0, write_file=None)
            scores[f"{split}/ece"] = {"value": ece, "string": f"{ece:.6}"}

        if "tace" in self.metrics:
            preds = np.concatenate(self._list_np_probs, axis=0)
            targets = np.concatenate(self._list_np_targets)
            tace = get_tace_bayesgroup(preds=preds, targets=targets)
            scores[f"{split}/tace"] = {"value": tace, "string": f"{tace:.6}"}

        return scores

    def _get_diversity_stats(self,):
        """
        Retrieve stored diversity stats
        """
        if not len(self._list_matrix_predictions_diversity):
            return {}

        num_members = len(self._list_matrix_predictions_diversity[0])
        list_predictions_diversity = [[] for _ in range(num_members)]

        for matrix_predictions_diversity in self._list_matrix_predictions_diversity:
            for num_member in range(num_members):
                list_predictions_diversity[num_member].extend(
                    matrix_predictions_diversity[num_member]
                )

        stats = metrics_ensemble.MetricsEnsemble(
            true=self._list_target_diversity,
            predictions=list_predictions_diversity,
            names=[str(i) for i in range(num_members)]
        ).get_report()

        return stats
