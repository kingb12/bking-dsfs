import math
import random
from typing import List

import tqdm
from matplotlib import pyplot as plt

from gradient_descent import gradient_step
from linalg import Vector, dot, vector_sum


def predicted_vs_actual(predictions: List[float], ys: List[float]):
    plt.scatter(predictions, ys, marker='+')
    plt.xlabel("Predictions")
    plt.ylabel("Actual")
    plt.show()


def logistic(x: float) -> float:
    """
    the mathematical logistic function
    """
    return 1.0 / (1 + math.exp(-x))


def logistic_prime(x: float) -> float:
    """
    derivative of the logistic function w.r.t. x
    """
    y = logistic(x)
    return y * (1 - y)


def _neg_log_likelihood(x: Vector, y: float, beta: Vector) -> float:
    """
    The negative log-likelihood for logistic regression w.r.t. a single data point
    """
    if y == 1:
        return -math.log(logistic(dot(x, beta)))
    else: 
        return -math.log(1 - logistic(dot(x, beta)))


def neg_log_likelihood(xs: List[Vector], ys: List[float], beta: Vector) -> float:
    """
    The negative log-likelihood for logistic regression w.r.t. a full dataset $(xs, ys)$
    """
    return sum(_neg_log_likelihood(x, y, beta) for (x, y) in zip(xs, ys))


def _negative_log_partial_j(x: Vector, y: float, beta: Vector, j: int) -> float:
    """
    The jth partial derivative of a single datapoint produced by y = logistic(dot(x, beta))
    """
    # TODO: why this value?
    return -(y - logistic(dot(x, beta))) * x[j]


def _negative_log_gradient(x: Vector, y: float, beta: Vector) -> Vector:
    """
    The gradient w.r.t. each beta parameter for a single point
    """
    return [_negative_log_partial_j(x, y, beta, j) 
            for j in range(len(beta))]


def negative_log_gradient(xs: List[Vector], ys: List[float], beta: Vector) -> Vector:
    """
    The gradient w.r.t. each beta parameter for an entire dataset
    """
    return vector_sum([_negative_log_gradient(x, y, beta)
                       for x, y in zip(xs, ys)])


def logistic_regression_fit(xs: List[Vector], ys: List[float], lr: float = 0.01, epochs: int = 5000, 
                            random_seed: int = None) -> Vector:
    """
    return the parameters $\beta$ of a logistic regression model relating xs, ys fit with gradient descent for
    {@code epochs} epochs with a learning rate of {@code lr}
    """
    if random_seed:
        random.seed(random_seed)
    # random initial guess
    beta: Vector = [random.random() for _ in range(len(xs[0]))]
    # out of laziness (of the developer), we'll just do gradient descent on the whole dataset instead of SGD
    with tqdm.trange(epochs) as t:
        for _ in t:
            gradient: Vector = negative_log_gradient(xs, ys, beta)
            beta = gradient_step(beta, gradient, -lr)
            loss = neg_log_likelihood(xs, ys, beta)
            t.set_description(f"loss {loss:.3f} beta: {beta})")
    return beta
