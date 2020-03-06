import math, random
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm, uniform
from collections import Counter


def uniform_pdf(x: float) -> float:
    """the p.d.f. for uniform distribution between [0, 1)"""
    return 1 if 0 <= x < 1 else 0


# note the statement holds: integral of f(n) = 1 over (0, x) is x.
# unlike the pdf, we need to have all x in [1, inf) have cdf of 1
def uniform_cdf(x: float) -> float:
    """the c.d.f. for uniform distribution between [0, 1)"""
    return min(x, 1) if x >= 0 else 0


def normal_pdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    """the pdf of a normal distribution with mean mu and std-dev sigma"""
    sqrt_2pi = math.sqrt(2 * math.pi)
    power = -.5 * ((x - mu) / sigma) ** 2
    coeff = 1 / (sigma * sqrt_2pi)
    return coeff * math.exp(power)


def normal_cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    """
    Cumulative distribution function for a random variable x, for a normal distribution with mean mu and std dev sigma
    """
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2


def inverse_normal_cdf(p: float, mu: float = 0, sigma: float = 1, tolerance=10 ** -7) -> float:
    """
    inverse of the normal cdf function
    """
    # if not standard, compute standard and rescale
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)
    # now were working with a standard normal cdf
    low_z, hi_z = -10.0, 10.0  # extreme bounds for a STANDARD normal cdf, though low enough tolerance breaks this
    mid_z = (hi_z - low_z) / 2.0
    mid_p = normal_cdf(mid_z)
    while hi_z - low_z > tolerance:
        mid_z = (hi_z + low_z) / 2.0
        mid_p = normal_cdf(mid_z)
        if mid_p > p:
            # our mid-point is too high
            hi_z = mid_z
        else:
            # our mid-point is too low
            low_z = mid_z
    return mid_z


def bernoulli_expmt(p: float = 0.5) -> int:
    """
    returns a sample from a bernoulli distribution with probability of 1 being p, 0 being 1 - p
    """
    return 1 if random.random() < p else 0


def binomial(n: int, p: float = 0.5) -> int:
    """
    A binomial random sample from n bernoulli trials with probability p per trial
    """
    return sum([bernoulli_expmt(p) for i in range(n)])


# plot a binomial histogram against a normal distribution that should approximate it
def binomial_histogram(p: float, n: int = 50, num_samples: int = 50):
    # use a bar char to show some actual binomial samples
    data = [binomial(n, p) for _ in range(num_samples)]
    histogram = Counter(data)
    plt.bar([x - 0.4 for x in histogram.keys()],
            [v / num_samples for v in histogram.values()],
            0.8, color='0.75')
    # now show normal approximation: expected value of a binomial is n * p
    xs = range(min(data), max(data) + 1)
    mu = n * p
    sigma = math.sqrt(n * p * (1 - p))
    ys = [normal_cdf(i + 0.5, mu, sigma) - normal_cdf(i - 0.5, mu, sigma) for i in xs]
    plt.plot(xs, ys, '-', label='mu={0},sigma={1}'.format(mu, sigma))
    return plt


if __name__ == '__main__':
    assert abs(uniform.pdf(.59) - uniform_pdf(.59)) < 10 ** -5
    assert abs(uniform.cdf(.59) - uniform_cdf(.59)) < 10 ** -5
    assert abs(norm.pdf(.59, loc=0, scale=1) - normal_pdf(.59, 0, 1)) < 10 ** -5
    assert abs(norm.cdf(.59, loc=0, scale=1) - normal_cdf(.59, 0, 1)) < 10 ** -5
    assert (norm.ppf(.4, 3., .4) - inverse_normal_cdf(.4, 3., .4) < 10 ** -5)
