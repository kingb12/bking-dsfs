from linalg import Vector, dot, distance, add, scalar_multiply, add, vector_mean
from typing import Callable
import random
import numpy as np


def sum_of_squares(xs: Vector) -> float:
    """
    Return the sum of the square of each element in xs
    """
    # this is equivalent to x dot x
    return dot(xs, xs)


def difference_quotient(f: Callable[[float], float], x: float, h: float) -> float:
    return (f(x + h) - f(x)) / h


def partial_diff_quotient(f: Callable[[Vector], float], xs: Vector, i: int, h: float) -> float:
    w = [x_j + (h if i == j else 0) for j, x_j in enumerate(xs)]  # single out and add h to just the ith element of xs
    return (f(w) - f(xs)) / h  # reflects only the change we made to the ith variable


def estimate_gradient(f: Callable[[Vector], float], xs: Vector, h: float = 10**-4) -> Vector:
    """
    Estimate the gradient of f with respect to xs by computing partial diff quotients element-wise
    """
    # note this is expensive and why auto-grad libraries mathematically compute most derivatives
    return [partial_diff_quotient(f, xs, i, h) for i in range(len(xs))]


def gradient_step(xs: Vector, gradient: Vector, step_size: float) -> Vector:
    """
    Moves `step_size` along the gradient of f w.r.t. xs, returning a input
    """
    assert len(xs) == len(gradient)
    update = scalar_multiply(step_size, gradient)
    return add(xs, update)


def sum_of_squares_gradient(xs: Vector) -> Vector:
    """
    We know the partial-derivative for a sum of squares is just 2*`the_term`
    """
    return [2*x for x in xs]


# we think f(x) is a polynomial, and want to compute a gradient using coefficients ws
def linear_gradient_mse(x: float, y: float, ws: Vector) -> Vector:
    predicted = sum([w * (x**i) for i, w in enumerate(ws)])  # our weights are coefficients to the polynomial
    target = y 
    error = predicted - target  
    grad = [2 * error * (x**i) for i in range(len(ws))]
    return grad
