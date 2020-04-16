
import sys

sys.path.insert(0, "../")
from linalg import Vector
from typing import Tuple
from stats import correlation, standard_deviation, mean


def predict(alpha: float, beta: float, x_i: float) -> float:
    """
    for some alpha, beta, x_i, return the estimate for y_i
    """
    return alpha * x_i + beta


def error(alpha: float, beta: float, x_i: float, y_i: float) -> float:
    """
    for some alpha, beta, x_i, and true value for y_i, return the difference between our estimate and the real value
    """
    return predict(alpha, beta, x_i) - y_i


def sum_of_squared_errors(alpha: float, beta: float, xs: Vector, ys: Vector) -> float:
    """
    for some alpha, beta and dataset xs, ys, return the sum of squared errors when predicting ys from xs using alpha and beta
    """
    errors = [error(alpha, beta, x_i, y_i) for x_i, y_i in zip(xs, ys)]
    return sum(e**2 for e in errors)


def least_squares_fit(xs: Vector, ys: Vector) -> Tuple[float, float]:
    """
    Given a dataset represented by xs and ys, return the alpha, beta that provide the least squared error fit for a
    function y_i = alpha * x_i + beta
    """
    alpha = correlation(xs, ys) * standard_deviation(ys) / standard_deviation(xs)
    beta = mean(ys) - alpha * mean(xs)
    return alpha, beta


def total_sum_of_squares(ys: Vector) -> float:
    """
    For a given vector (in this R-squared context our y values), return the total squared distance from the mean
    """
    mu = mean(ys)
    return sum((y - mu)**2 for y in ys)


def r_squared(alpha: float, beta: float, xs: Vector, ys: Vector) -> float:
    """
    give the R-squared value of our line of best fit relative to actual data xs, ys. This is equivalent to the 
    fraction of the variance in ys that we account for in our predictions, itself equivalent to 1.0 - the fraction we
    miss
    """
    var_missed = sum_of_squared_errors(alpha, beta, xs, ys)
    var_present = total_sum_of_squares(ys)
    return 1.0 - (var_missed / var_present)
