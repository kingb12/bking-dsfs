from collections import Counter
from typing import List
import matplotlib.pyplot as plt
import random, math
from linalg import Vector, sum_of_squares, subtract, dot
import numpy as np


def sample_friend_count() -> int:
    sample = int(math.ceil(random.gauss(50, 15)))
    return max(1, min(100, sample))


# now lets add synthetic co-variant data for daily minutes on site. We'll say # minutes is correlated by num friends
def sample_daily_minutes(friend_count: int) -> int:
    noise = int(math.ceil(random.gauss(30, 12)))
    noise_corrected = max(1, min(120, noise))
    # cap at 140 for friend count
    return int((140 * (friend_count / 100.)) + noise)


def mean(xs: Vector) -> float:
    "returns mean value of vector"
    return sum(xs) / len(xs)


def median(xs: Vector) -> float:
    "returns the median value of the vector"
    # edit: x // y provides simpler floor-division than math.floor(x / y)
    assert xs and len(xs) > 0, 'cannot get median of a null or empty vector'
    xs_sorted = sorted(xs)
    mid_index = len(xs_sorted) // 2
    # if len(xs_sorted) is odd: mid index contains true median (e.g. 5 element list has median at index 2)
    # if len(xs_sorted) is even: mid index is the right-half of the median values to average
    # (e.g. a 6 element list has mid index 3, but median is defined by indices 2 & 3)
    if len(xs_sorted) % 2 == 1:
        return xs_sorted[mid_index]
    else:
        return (xs_sorted[mid_index] + xs_sorted[mid_index - 1]) / 2


# quantile generalization of median
def quantile(xs: Vector, p: float) -> float:
    """Returns the value for the pth-percentile of the data (e.g. median for p=.5)"""
    assert xs and len(xs) > 0, 'cannot get median of a null or empty vector'
    xs_sorted = sorted(xs)
    p_index = int(len(xs_sorted) * p)
    return xs_sorted[p_index]


def mode(xs: Vector) -> List[float]:
    """finds the mode of the vector, returning a list in case there is more than one mode"""
    counter = Counter(xs)
    max_count = max(counter.values())
    return [val for val, count in counter.items() if count == max_count]


def variance(xs: Vector) -> float:
    """returns the variance of a vector"""
    assert xs and len(xs) >= 2, 'variance requires at least two elements'
    sample_mean = mean(xs)
    v_diff = [x - sample_mean for x in xs]
    return sum_of_squares(v_diff) / (len(v_diff) - 1)


def standard_deviation(xs: Vector) -> float:
    """returns the standard deviation of a vector, which is sqrt variance

    Units: if xs is in meters, returns std dev in meters
    """
    assert xs and len(xs) >= 2, 'std dev requires at least two elements'
    return math.sqrt(variance(xs))


def data_range(xs: Vector) -> float:
    """returns the range of a vector"""
    assert xs, 'need a vector!'
    return max(xs) - min(xs)


def interquantile_range(xs: Vector) -> float:
    """returns the difference between the 75th and 25th percent quantile"""
    return quantile(xs, .75) - quantile(xs, .25)


def covariance(xs: Vector, ys: Vector) -> float:
    """return the covariance of two vectors"""
    assert xs and ys and len(xs) == len(ys), 'vectors must exist and have equal length'
    x_mean, y_mean = mean(xs), mean(ys)
    x_bar, y_bar = [x - x_mean for x in xs], [y - y_mean for y in ys]
    # now we have two vectors of the form x_i - x_mean, the total covariance
    # is a dot product sum((x_i - x_mean)*(y_i - y_mean)) / length (length alone shouldn't dictate covariance)
    return dot(x_bar, y_bar) / (len(xs) - 1) # Note the Bessel correction


def correlation(xs: Vector, ys: Vector) -> float:
    """
    return the Pearson correlation-coefficient of two vectors (a value in [-1, 1] that indicates correlation
    strength. This can be thought of as covariance normalized by the standard deviations of each vector
    """
    cov = covariance(xs, ys)
    std_x = standard_deviation(xs)
    std_y = standard_deviation(ys)
    # if there is no variance in either xs or ys, there is no correlation
    return cov / (std_x * std_y) if std_x > 0 and std_y > 0 else 0


if __name__ == '__main__':
    # seed random
    random.seed(42)
    num_friends = [sample_friend_count() for i in range(1000)]
    daily_minutes = [sample_daily_minutes(num_friends[i]) for i in num_friends]
    friend_counts = Counter(num_friends)
    xs = range(101)
    ys = [friend_counts[i] for i in xs]
    plt.bar(xs, ys)
    plt.axis([0, 101, 0, 50])
    plt.title('Histogram of Friend Counts')
    plt.xlabel('# of friends')
    plt.ylabel('# of people')
    plt.show()
    # report max, min, count of values
    print('Max:', max(num_friends), 'Min:', min(num_friends), 'Count:', len(num_friends))
    # this is just a special case of wanting to know sorted positions, which opens up median, quartiles, etc.
    sorted_values = sorted(num_friends)
    print('Max:', sorted_values[-1], 'Min:', sorted_values[0], 'Count:', len(sorted_values),
          'Median:', sorted_values[len(sorted_values) // 2])
    data_xs = list(np.random.rand(100))
    data_ys = list(np.random.rand(100))

    # np.cov returns full covariance matrix (0, 0) is variance in xs, (1, 1) is variance in ys,
    # (0, 1) and (1, 0) are covariance xs, ys
    assert abs(np.cov(data_xs, y=data_ys, ddof=1)[0][1] - covariance(data_xs, data_ys)) < 10 ** -9


