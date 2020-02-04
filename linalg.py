from typing import List, Tuple, Callable
import math
"""
A class of basic linear algebra operations for use in later chapters
"""

# first, lets add a type hint for a vector, which will be a list of floats
Vector = List[float]


def add(v: Vector, w: Vector) -> Vector:
    """component-wise addition of two vectors"""
    assert len(v) == len(w), 'cannot add vectors of inequal length'
    return [x + y for (x, y) in zip(v, w)]


def subtract(v: Vector, w: Vector) -> Vector:
    """component-wise subtraction of two vectors"""
    assert len(v) == len(w), 'cannot subtract vectors of inequal length'
    return [x - y for (x, y) in zip(v, w)]


def vector_sum(vs: List[Vector]) -> Vector:
    """component-wise addition of N vectors"""
    assert vs, "provide vectors"
    # assert the length of each vector in vs is the same
    assert all(len(v) == len(vs[0]) for v in vs),'cannot add vectors of inequal length'
    # use of * below unpacks vs into the arguments, equivalent to zip(v_1, v_2, ... v_n)
    return [sum(t) for t in zip(*vs)]


def scalar_multiply(scalar: float, v: Vector) -> Vector:
    """scalar multiplication of scalar by each vector"""
    assert v and scalar, "provide scalar & vector"
    return [scalar * i for i in v]


def vector_mean(vs: List[Vector]) -> Vector:
    """element-wise average of a list of vectors"""
    # add all the vectors and multiply by 1/n
    return scalar_multiply(1/len(vs), vector_sum(vs))


def dot(v: Vector, w: Vector) -> float:
    """returns the dot product of two vectors"""
    assert v and w, 'need vectors'
    assert len(v) == len(w), 'cannot dot vectors of inequal length'
    return sum(x * y for (x, y) in zip(v, w))


def sum_of_squares(v: Vector) -> float:
    """returns the sum of squares of a vector (e.g v_0**2 + v_1**2 ... v_n**2)"""
    # this is the same as dot(v, v)
    return dot(v, v)


def magnitude(v: Vector) -> float:
    """magnitude of a vector == sqrt of summed squares"""
    assert v, 'need a vector!'
    return math.sqrt(sum_of_squares(v))


def squared_distance(v: Vector, w: Vector) -> float:
    """the distance between two vectors, squared: (v_0 - w_0)**2 + ... + (v_n - w_n)**2"""
    assert v and w, 'need vectors'
    assert len(v) == len(w), 'vectors must have equal length'
    return sum_of_squares(subtract(v, w))


def distance(v: Vector, w: Vector) -> float:
    # return math.sqrt(squared_distance(v, w))
    return magnitude(subtract(v, w))


# On to matrices!
Matrix = List[List[float]]


def valid_matrix(m: Matrix) -> bool:
    """checks our representation of a matrix"""
    assert m, 'empty matrix!'
    return all(len(v) == len(m[0]) for v in m)


def shape(m: Matrix) -> Tuple[int, int]:
    """returns the shape of a matrix"""
    assert valid_matrix(m), 'invalid matrix supplied'
    return len(m), len(m[0]) # n by k


def get_column(m: Matrix, j: int) -> Vector:
    """returns the jth column of an n * k matrix as a vector of length n"""
    assert m and valid_matrix(m), 'invalid matrix!'
    return [v[j] for v in m]


def make_matrix(n: int, k: int, entry_fn: Callable[[int, int], float]) -> Matrix:
    """constructs a n * k matrix A by creating each element with entry_fn(i, j) -> A(i, j)"""
    return [[entry_fn(i, j) for j in range(k)] for i in range(n)]


def identity_matrix(n: int) -> Matrix:
    """returns an identity matrix of size n * n"""
    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)


if __name__ == "__main__":
    a = [1., 2., 3.]
    b = [2., 3., 4.]
    c = [0., 2., 3.]
    d = [2., 0., 4.]
    e = [1., 2., 0.]
    vs = [a, b, c, d, e]
    assert add(a, b) == [3, 5, 7]
    assert subtract(a, b) == [-1, -1, -1]
    assert vector_sum(vs) == [6, 9, 14]
    assert scalar_multiply(3, a) == [3, 6, 9]
    # fancy assertion to deal with rounding issues
    for (x, y) in zip(vector_mean(vs), [1.2, 1.8, 2.8]):
        assert abs(x - y) < 10 ** (-8)
    assert dot(b, c) == 18
    assert sum_of_squares(b) == 29
    assert valid_matrix([[1, 2], [3, 4]])
    assert not valid_matrix([[1, 2], [3, 4, 5]])
    assert shape([[1, 2], [3, 4]]) == (2, 2)
    assert get_column([[1, 2], [3, 4]], 1) == [2, 4]
    assert make_matrix(2, 2, lambda i, j: i + j + 1) == [[1, 2], [2, 3]]
    assert identity_matrix(3) == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert distance([1., 2., 3.], [2., 3., 4]) == math.sqrt(3)
    assert distance([1., 2., 3.], [3., 4., 5]) == math.sqrt(12)
    assert squared_distance([1., 2., 3.], [2., 3., 4]) == 3
    assert magnitude([3, 4]) == 5
    print('all assertions pass!')
