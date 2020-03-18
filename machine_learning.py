#!/usr/bin/env python

import random, copy
from typing import TypeVar, List, Tuple

X = TypeVar('X')  # generic type to represent a data point
Y = TypeVar('Y')  # generic type to represent an output variable


def split_data(data: List[X], proportion_train: float) -> Tuple[List[X], List[X]]:
    """
    Given a dataset `data` and a proportion `proportion_train` that should be the training set,
    return a tuple with the dataset split into train, test appropriately. Shallow-copies dataset.
    """
    data = copy.copy(data)  # shallow. data[:] also works for reference if you see it in the wild, but too implicit imo
    random.shuffle(data)
    cut = int(len(data) * proportion_train)
    return data[0:cut], data[cut:]


def train_test_split(xs: List[X], ys: List[Y], proportion_train: float) -> Tuple[List[X], List[X], List[Y], List[Y]]:
    """
    Given a dataset `xs`  with corresponding outputs `ys` and a proportion `proportion_train` that should be 
    the training set,return a tuple with the dataset split into train, test appropriately. Shallow-copies dataset.
    """
    assert len(xs) == len(ys), 'number of inputs and outputs should be the same'
    train_indices, test_indices = split_data(range(len(xs)), proportion_train)
    xs, ys = copy.copy(xs), copy.copy(ys)
    return ([xs[i] for i in train_indices],
            [xs[i] for i in test_indices],
            [ys[i] for i in train_indices],
            [ys[i] for i in test_indices])


def accuracy(true_positives: int, false_positives: int, false_negatives: int, true_negatives: int) -> float:
    """
    return the accuracy from confusion matrix data (correct / total)
    """
    correct: int = true_positives + true_negatives
    total: int = true_positives + true_negatives + false_positives + false_negatives
    return float(correct) / total


def precision(true_positives: int, false_positives: int, false_negatives: int, true_negatives: int) -> float:
    """
    return the precision from confusion matrix data (true positives / total positive predictions). To be compatible with `accuracy`
    we'll preserve the signature despite not using the negative counts
    """
    total_positive_predictions: int = true_positives + false_positives
    return float(true_positives) / total_positive_predictions


def recall(true_positives: int, false_positives: int, false_negatives: int, true_negatives: int) -> float:
    """
    return the recall from confusion matrix data (true positives / total positives). To be compatible with `accuracy`
    we'll preserve the signature despite not using the negative counts
    """
    total_positive_outcomes: int = true_positives + false_negatives
    return float(true_positives) / total_positive_outcomes


def f_beta_score(true_positives: int, false_positives: int, false_negatives: int, true_negatives: int, 
                 beta: float = 1.) -> float:
    """
    compute the F-beta score from the confusion matrix
    """
    p = precision(true_positives, false_positives, false_negatives, true_negatives)    
    r = recall(true_positives, false_positives, false_negatives, true_negatives)
    return (1 + beta) * p * r / ((beta * p) + r)


def f1_score(true_positives: int, false_positives: int, false_negatives: int, true_negatives: int) -> float:
    """
    compute the F1 score from the confusion matrix
    """
    return f_beta_score(true_positives, false_positives, false_negatives, true_negatives, beta=1.)


if __name__ == '__main__':
    assert accuracy(70, 4930, 13930, 981070) == .98114
    assert precision(70, 4930, 13930, 981070) == .014
    assert recall(70, 4930, 13930, 981070) == .005
    # compare with expression for f1_score
    assert f1_score(70, 4930, 13930, 981070) == 2 * .005 * .014 / (.014 + .005)
    assert f1_score(70, 4930, 13930, 981070) == f_beta_score(70, 4930, 13930, 981070, beta=1.)
    assert f_beta_score(70, 4930, 13930, 981070, beta=2.) == 3 * .014 * .005 / ((2. * .014) + .005)






