import datetime
import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, NamedTuple, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tqdm

from gradient_descent import gradient_step
from linalg import Matrix, Vector, make_matrix, vector_mean, subtract, dot, magnitude, scalar_multiply
from probability import inverse_normal_cdf
from stats import correlation, standard_deviation, variance


def _bucketize(point: float, bucket_size: float) -> float:
    """
    For a given data point, floor it to the appropriate bucket
    """
    return bucket_size * math.floor(point / bucket_size)


def make_histogram(points: List[float], bucket_size: float) -> Dict[float, int]:
    return Counter([_bucketize(point, bucket_size) for point in points])


def plot_histogram(histogram_data: Dict[float, int], title: str = 'Histogram') -> None:
    xs = list(histogram_data.keys())
    xs.sort()
    ys = [histogram_data[k] for k in xs]
    width = xs[1] - xs[0] if len(xs) > 1 else 10
    plt.bar(xs, ys, width=width)
    plt.title(title)
    plt.show()


def summary_stats(points: List[float]) -> Dict[str, float]:
    stats = {}
    stats['mean'] = np.mean(points)
    stats['stddev'] = np.std(points)
    stats['min'] = min(points)
    stats['max'] = max(points)
    return stats


def random_normal(mu:float = 0, sigma: float = 1) -> float:
    """
    random sample X from a normal distribution X ~ Normal(mu, sigma)
    """
    return inverse_normal_cdf(random.random(), mu, sigma)


def correlation_matrix(data: List[Vector]) -> Matrix:
    """
    Returns the len(data) * len(data)) correlation matrix whose (i, j)-th entry is 
    the correlation between data[i] and data[j]
    
    Note: in this example, the data is shaped (features * examples) vs. (examples * features): we provide 
    correlation across axis=0
    """
    def correlation_ij(i: int, j: int) -> float:
        # inner function which will serve as our data-generator for make_matrix 
        # (note: the function defines relationship to argument data)
        return correlation(data[i], data[j])
    return make_matrix(len(data), len(data), correlation_ij)


def scatter_matrix(data: List[Vector]) -> None:
    """
    For data of shape M * N, produce an M * M set of subplot scatter-plots where  the the plot at i, j is the
    scatter plot of data[i] against data[j]
    """
    num_features = len(data)
    
    # define sub-plots
    fig, ax = plt.subplots(num_features, num_features)
    
    # add data to them
    for i in range(num_features):
        for j in range(num_features):
            if i != j:
                ax[i][j].scatter(data[j], data[i]) # why j x i vs. i x j?
            else:
                ax[i][j].annotate("series " + str(i), (0.5, 0.5), xycoords='axes fraction', ha='center', va='center')
            if j > 0: 
                ax[i][j].yaxis.set_visible(False)
            if i < num_features - 1: 
                ax[i][j].xaxis.set_visible(False)
    plt.show()


# This solves spelling and field-hints, but doesn't solve non-uniform type case. We can do one better by sub-classing
class StockPrice(NamedTuple):
    symbol: str
    date: datetime.date
    closing_price: float

    def pays_taxes(self) -> bool:
        # its a class so we can add methods too
        return False


@dataclass
class StonkPrice:
    symbol: str
    date: datetime.date
    closing_price: float

    def pays_taxes(self) -> bool:
        # its a class so we can add methods too
        return False


def _semi_random_symbol() -> str:
    base = list('ABCD')
    random.shuffle(base)
    return ''.join(base[0:4])


def _random_stock():
    return StockPrice(_semi_random_symbol(), 
                      datetime.date(random.randint(2010, 2019), random.randint(1, 12), random.randint(1, 12)), 
                      np.random.normal(350, 100))


class DailyChange(NamedTuple):
    symbol: str
    date: datetime.date
    pct_change: float


def pct_change(start: float, end: float) -> float:
    return ((end - start) / start) * 100
    

class HeightAndWeight(NamedTuple):
    id: str
    height: float
    weight: float


def parse_row(row: List[str]) -> HeightAndWeight:
    # skipping the failure cases for time
    return HeightAndWeight(row[0], float(row[1]), float(row[2])) if len(row) == 3 else None


def scale(data: List[Vector]) -> Tuple[Vector, Vector]:
    """
    Given a list of data points, return 
    1) a vector of their means across features and 
    2) a vector of their stddevs across features
    """
    assert data is not None and len(data) > 0
    num_features = len(data[0])
    means = vector_mean(data)
    # for each feature compute a standard deviation of the value at that features index for each vector
    # we could one-call this if we wrote a vector_stddev function
    stdevs = [standard_deviation([vector[i] for vector in data]) for i in range(num_features)]
    return means, stdevs


def rescale(data: List[Vector]) -> List[Vector]:
    """
    rescale the input data so that each feature v[i] has mean = 0 and stdev = 1
    """
    assert data is not None and len(data) > 0
    num_features = len(data[0])
    means, stdevs = scale(data)
    
    # copy each vector prior to rescaling
    rescaled = [v[:] for v in data]
    for v in rescaled:
        for i in range(num_features):
            mu, sigma = means[i], stdevs[i]
            if sigma > 0:
                v[i] = (v[i] - mu) / sigma
    return rescaled


def recenter(data: List[Vector]) -> List[Vector]:
    """
    Like rescale, but we only center the data around a mean of 0 across all dimensions: there is no adjustment for variance
    """
    means = vector_mean(data)
    return [subtract(vector, means) for vector in data]


# Setting up some basic pieces of PCA
def direction(v: Vector) -> Vector:
    """
    Return a vector whose magnitude is normalized to 1, indicating the primary direction of the vector
    """
    mag = magnitude(v)
    return [v_i / mag for v_i in v]


def directional_variance(data: List[Vector], w: Vector) -> float:
    """
    Given a dataset and a vector w from which to take a direction, return the variance in the data along that direction
    """
    dir_w = direction(w)
    # key insight: the dot product of two orthogonal vectors is zero. dot product against a magnitude vector is the portion 
    # of the magnitude of the query vector in THAT direction
    dot_projections: Vector = [dot(v, dir_w) for v in data]
    return variance(dot_projections)  # books code doesn't center mean but that should be ok, we already centered it


def dv_gradient(data: List[Vector], w: Vector) -> Vector:
    """
    Given a dataset and direction, compute the gradient of the directional variance relative to that direction
    """
    # if variance is sum of squares, do we just want to sum 2 * i in the dot projections?
    dir_w = direction(w)
    return [sum(2 * dot(v, dir_w) * v[i] for v in data)
            for i in range(len(w))]


def first_principal_component(data: List[Vector], epochs: int = 1000, step_size: float = .1) -> Vector:
    """
    Given a dataset, determine the first principle component using gradient descent
    epochs = # of epochs, step_size = learning rate
    """
    # start with a random guess
    dir_w = [np.random.uniform(-1, 1) for _ in data[0]]
    with tqdm.trange(epochs) as t:
        for _ in t:
            dv = directional_variance(data, dir_w)
            gradient = dv_gradient(data, dir_w)
            dir_w = gradient_step(dir_w, gradient, step_size)
            t.set_description(f"dv: {dv:.3f}")  # note the very nice formatting syntax for dv => 0.123
    return direction(dir_w)
    

def project(v: Vector, w: Vector) -> Vector:
    """
    return the projection of v onto w
    """
    projection_length = dot(v, w)
    return scalar_multiply(projection_length, w)


def remove_projection_from_vector(v: Vector, w: Vector) -> Vector:
    """
    projects v onto w and subtracts that projection from v
    """
    return subtract(v, project(v, w))


def remove_projection(data: List[Vector], w: Vector) -> List[Vector]:
    """
    remove a direction from the dataset by removing it from each vector
    """
    return [remove_projection_from_vector(v, w) for v in data]


def project2(v: Vector, w: Vector) -> Vector:
    """return the projection of v onto the direction w"""
    projection_length = dot(v, w)
    return scalar_multiply(projection_length, w)


def remove_projection_from_vector2(v: Vector, w: Vector) -> Vector:
    """projects v onto w and subtracts the result from v"""
    return subtract(v, project2(v, w))


def remove_projection2(data: List[Vector], w: Vector) -> List[Vector]:
    return [remove_projection_from_vector2(v, w) for v in data]


def pca(data: List[Vector], num_components: int) -> List[Vector]:
    """
    Find a series of num_components principal components of data
    """
    data = recenter(data)  # to copy and make sure we don't forget
    components: List[Vector] = []
    for _ in range(num_components):
        component = first_principal_component(data)
        components.append(component)
        data = remove_projection(data, component)
    return components


def transform_vector(v: Vector, span: List[Vector]) -> Vector:
    """
    transform the vector to live in the span of the components passed in
    """
    return [dot(v, w) for w in span]


def transform(data: List[Vector], span: List[Vector]) -> List[Vector]:
    return [transform_vector(v, span) for v in data]
