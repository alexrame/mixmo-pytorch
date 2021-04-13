"""
Ensembling metrics
Inspired from
* https://github.com/kbogas/EnsembleDiversityTests/blob/master/EnsembleDiversityTests.py
* https://github.com/scikit-learn-contrib/DESlib
"""

import numpy as np
from collections import OrderedDict

from mixmo.utils import misc
from mixmo.utils.logger import get_logger


LOGGER = get_logger(__name__, level="DEBUG")


class MetricsEnsemble(object):
    """
    Class Wrapper to get Diversity Measures over collection of predictions.
    Args:
        @predictions: list of lists. Each sublist contains the predictions
                      of a classifier
        @names: list of strings. Each string is the name of the classifier.
        @true: list of labels. Each label is the truth label
    """

    def __init__(self, predictions, names, true):

        N = len(true)
        labels = set(true)
        if len(predictions) != len(names):
            raise AttributeError(
                'Number of classifiers is different than number \
                                  of names. %d != %d.'                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     % (len(predictions), len(names))
            )
        for i, predict in enumerate(predictions):
            if len(predict) != N:
                raise AttributeError(
                    'Number of predictions of classifier %s is different then the number of true labels. %d != %d'                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         % (names[i], len(predict), N)
                )
            if labels.isdisjoint(set(predict)):
                import pdb; pdb.set_trace()
                raise AttributeError(
                    'Label in predictions of %s not in truth set.'                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         % (names[i])
                )
        self.names = names
        self.true = true
        self.predictions = predictions

    def get_report(self, print_flag=False):
        stats = OrderedDict({})
        misc.clean_update(stats, self.get_diversity_ratioerrors(print_flag=print_flag))
        misc.clean_update(stats, self.get_individualaccuracies(print_flag=print_flag))
        return stats

    def get_diversity_ratioerrors(self, print_flag=True):
        prediction_matrix = np.transpose(np.array(self.predictions))
        stats = OrderedDict({})
        if print_flag:
            print('### Pairwise Diversity Metrics: ###')
        for diversity_name, diversity_func in [
                ("diversity_ratioerrors", ratio_errors),
                ]:
            diversity_matrix = compute_pairwise_diversity(
                targets=self.true,
                prediction_matrix=prediction_matrix,
                diversity_func=diversity_func)
            value = np.mean(compute_mean_without_diagonal(diversity_matrix).tolist())
            stats[diversity_name] = {
                "value": value,
                "string": f"{value:.5}"}
            if print_flag:
                print(f"Avg. {diversity_name}: {value}")
        return stats


    def get_individualaccuracies(self, print_flag=True):
        """
        """
        meanaccuracy, accuracies = get_accuracy_multi(self.predictions, self.true)

        if print_flag:
            print(f"Accuracies: {accuracies}")
            print(f"Mean accuracy: {meanaccuracy}")
        stats = {"accuracy_mean": meanaccuracy}

        for i, accuracy in enumerate(accuracies):
            stats["accuracy_" + str(i)] = accuracy

        return {
            key: {
                "value": accuracy, "string": f"{accuracy:05.2%}"}
            for key, accuracy in stats.items()
            }


    def help(self):
        """Just a helper function to print the class docstring."""
        return self.__doc__


def get_accuracy_multi(predictions, y_true):

    num_labels = len(y_true)
    accurate_predictions = [0 for _ in predictions]
    for j in range(num_labels):
        for pred_i, pred in enumerate(predictions):
            if pred[j] == y_true[j]:
                accurate_predictions[pred_i] += 1

    accuracies = [accurate/num_labels for accurate in accurate_predictions]

    return np.mean(accuracies), accuracies



def _process_predictions(y, y_pred1, y_pred2):
    """Pre-process the predictions of a pair of base classifiers for the
    computation of the diversity measures

    Parameters
    ----------
    y : array of shape = [n_samples]:
        class labels of each sample.

    y_pred1 : array of shape = [n_samples]:
              predicted class labels by the classifier 1 for each sample.

    y_pred2 : array of shape = [n_samples]:
              predicted class labels by the classifier 2 for each sample.

    Returns
    -------
    N00 : Percentage of samples that both classifiers predict the wrong label

    N10 : Percentage of samples that only classifier 2 predicts the wrong label

    N10 : Percentage of samples that only classifier 1 predicts the wrong label

    N11 : Percentage of samples that both classifiers predict the correct label
    """
    size_y = len(y)
    if size_y != len(y_pred1) or size_y != len(y_pred2):
        raise ValueError('The vector with class labels must have the same size.')

    N00, N10, N01, N11 = 0.0, 0.0, 0.0, 0.0
    for index in range(size_y):
        if y_pred1[index] == y[index] and y_pred2[index] == y[index]:
            N11 += 1.0
        elif y_pred1[index] == y[index] and y_pred2[index] != y[index]:
            N10 += 1.0
        elif y_pred1[index] != y[index] and y_pred2[index] == y[index]:
            N01 += 1.0
        else:
            N00 += 1.0

    return N00 / size_y, N10 / size_y, N01 / size_y, N11 / size_y


def ratio_errors(y, y_pred1, y_pred2):
    """Calculates Ratio of errors diversity measure between a pair of
    classifiers. A higher value means that the base classifiers are less likely
    to make the same errors. The ratio must be maximized for a higher diversity

    Parameters
    ----------
    y : array of shape = [n_samples]:
        class labels of each sample.

    y_pred1 : array of shape = [n_samples]:
              predicted class labels by the classifier 1 for each sample.

    y_pred2 : array of shape = [n_samples]:
              predicted class labels by the classifier 2 for each sample.

    Returns
    -------
    ratio : The q-statistic measure between two classifiers

    References
    ----------
    Aksela, Matti. "Comparison of classifier selection methods for improving
    committee performance."
    Multiple Classifier Systems (2003): 159-159.
    """
    N00, N10, N01, N11 = _process_predictions(y, y_pred1, y_pred2)
    if N00 == 0:
        LOGGER.warning("No shared errors !")
        ratio = 2 * (N01 + N10)
    else:
        ratio = (N01 + N10) / N00
    return ratio


def compute_pairwise_diversity(targets, prediction_matrix, diversity_func):
    """Computes the pairwise diversity matrix.

     Parameters
     ----------
     targets : array of shape = [n_samples]:
        Class labels of each sample in X.

     prediction_matrix : array of shape = [n_samples, n_classifiers]:
        Predicted class labels for each classifier in the pool

     diversity_func : Function
        Function used to estimate the pairwise diversity

     Returns
     -------
     diversity : array of shape = [n_classifiers]
        The average pairwise diversity matrix calculated for the pool of
        classifiers

     """
    n_classifiers = prediction_matrix.shape[1]
    diversity_matrix = np.zeros([n_classifiers, n_classifiers])

    for clf_index in range(n_classifiers):
        for clf_index2 in range(clf_index + 1, n_classifiers):
            this_diversity = diversity_func(
                targets, prediction_matrix[:, clf_index], prediction_matrix[:, clf_index2]
            )

            diversity_matrix[clf_index, clf_index2] = this_diversity
            diversity_matrix[clf_index2, clf_index] = this_diversity

    return diversity_matrix


def compute_mean_without_diagonal(matrix):
    return np.sum(matrix, axis=1) / (matrix.shape[0] - 1)
