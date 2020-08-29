import tqdm, random
from linalg import dot, Vector, vector_mean, vector_sum, scalar_multiply, add
from probability import normal_cdf
from simple_linear_regression import total_sum_of_squares
from gradient_descent import gradient_step
from typing import List, TypeVar, Callable, Tuple

X = TypeVar('X')
Stat = TypeVar('Stat')


def predict(x: Vector, beta: Vector) -> float:
    """
    Given values for x and parameters beta, give a prediction for y. This assumes any bias transform 
    [1, x_0 ... x_n] has already been performed on x, beta
    """
    return dot(x, beta)


def error(x: Vector, y: float, beta: Vector) -> float:
    """
    with a prediction defined by x * beta, return the error w.r.t the true value y
    """
    return predict(x, beta) - y


def squared_error(x: Vector, y: float, beta: Vector) -> float:
    """
    with a prediction defined by x * beta, return the squared error w.r.t the true value y
    """
    return error(x, y, beta) ** 2


def sq_error_gradient(x: Vector, y: float, beta: Vector) -> Vector:
    """
    For each input in x, return the gradient with respect to that input for a squared-error loss
    """
    err: float = error(x, y, beta)
    return [2 * err * x_i for x_i in x]


def _grad_step(beta: Vector, grad: Vector, lr: float = 10**-3) -> Vector:
    # repeating this to check my knowledge of the gradient update
    update: Vector = scalar_multiply(-1 * lr, grad)
    return vector_sum([beta, update])


def least_squares_fit(xs: List[Vector], ys: List[float], lr: float = 10**-3, num_steps: int = 1000, batch_size: int = 1) -> Vector:
    """
    For the given inputs (`xs`) and outputs (`ys`), find the parameters (`beta`) that provide the best fit 
    via multiple regression. Performs `num_steps` gradient descent steps with batch sizes of `batch_size` 
    and a learning rate of `lr`.
    """
    # paired = list(zip(xs, ys))
    # random.shuffle(paired)  doing this seems correct and exerimentally valid, but gets slightly worse results than the
    # book. Maybe the book has well-selected hyper parameters?
    # xs, ys = zip(*paired)
    beta = [random.random() for r in range(0, len(xs[0]))]
    for _ in tqdm.trange(num_steps, desc="least squares fit"):
        for start in range(0, len(xs), batch_size):
            # prepare a batch
            batch_xs = xs[start:start + batch_size]
            batch_ys = ys[start:start + batch_size]
            # compute an average gradient across the batch
            grads = [sq_error_gradient(x, y, beta) for (x, y) in zip(batch_xs, batch_ys)]
            avg_grad: Vector = vector_mean(grads)
            # update beta according to the gradient
            beta = gradient_step(beta, avg_grad, lr * -1)  # by negative one b/c we are minimizing
    return beta


def multiple_r_squared(xs: List[Vector], ys: Vector, beta: Vector) -> float:
    """
    Given the dataset and a vector beta of parameters, return the R-squared value for how well the `beta` 
    """
    explained_y_variance = total_sum_of_squares(ys)
    predicted_y_variance = total_sum_of_squares([predict(x, beta) for x in xs])
    return predicted_y_variance / explained_y_variance


def bootstrap_sample(xs: List[X], n: int = 0) -> List[X]:
    """
    Sample a dataset with replacement to get a sub-sample
    """
    return [random.choice(xs) for _ in (range(n) if n > 0 else xs)]


def bootstrap_statistic(xs: List[X], stats_fn: Callable[[List[X]], Stat], num_samples: int) -> List[Stat]:
    """
    evaluates stats_fn on num_samples bootstrap samples of xs
    """
    return [stats_fn(bootstrap_sample(xs)) for _ in range(num_samples)]


def estimate_sample_beta(pairs: List[Tuple[Vector, float]]) -> Vector:
    """
    estimate beta for a given sample of the dataset
    """
    # xs, ys = zip(*pairs)  # split sample into xs, ys
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    beta = least_squares_fit(xs, ys, 10**-3, 5000, 25)
    print('Beta sample', beta)
    return beta


def p_value(beta_hat_i: float, sigma_hat_i: float) -> float:
    """
    return the probabity estimate that we would observe this parameter beta given the 
    standard error if the true value of beta is zero
    """
    if beta_hat_i > 0: 
        positive_tail = 1 - normal_cdf(beta_hat_i / sigma_hat_i)  # probability that X ~ Normal(0, 1) > beta_hat_i
        return 2 * positive_tail  # we're asking about absolute deviation
    else:
        negative_tail = normal_cdf(beta_hat_i / sigma_hat_i)  # probability that X ~ Normal(0, 1) > beta_hat_i
        return 2 * negative_tail  # we're asking about absolute deviation    


def ridge_penalty(beta: Vector, alpha: float) -> float:
    """
    Return the error for x, y, beta w.r.t a ridge regression with alpha=alpha
    """
    # don't penalize the bias/constant term
    return alpha * sum(dot(beta[1:], beta[1:]))


def ridge_penalty_gradient(beta: Vector, alpha: float) -> Vector:
    """
    Return the gradient for x, y, beta w.r.t just the regularization term in a ridge regression with alpha=alpha
    """
    # don't penalize the bias/constant term in the gradient, either
    return [0] + [2 * alpha * b_i for b_i in beta[1:]]
    

def ridge_error_gradient(x: Vector, y: float, beta: Vector, alpha: float) -> Vector:
    """
    Return the gradient for x, y, beta w.r.t a ridge regression with alpha=alpha (just add both gradients)
    """
    return add(sq_error_gradient(x, y, beta), ridge_penalty_gradient(beta, alpha))
    

def ridge_regression_fit(xs: List[Vector], ys: List[float], lr: float = 10**-3, num_steps: int = 1000,
                         batch_size: int = 1, alpha: float = 1) -> Vector:
    """
    For the given inputs (`xs`) and outputs (`ys`), find the parameters (`beta`) that provide the best fit 
    via ridge regression. Performs `num_steps` gradient descent steps with batch sizes of `batch_size` 
    and a learning rate of `lr`.
    """
    # paired = list(zip(xs, ys))
    # random.shuffle(paired)  doing this seems correct and exerimentally valid, but gets slightly worse results than the
    # book. Maybe the book has well-selected hyper parameters?
    # xs, ys = zip(*paired)
    beta = [random.random() for r in range(0, len(xs[0]))]
    for _ in tqdm.trange(num_steps, desc="ridge regression squares fit"):
        for start in range(0, len(xs), batch_size):
            # prepare a batch
            batch_xs = xs[start:start + batch_size]
            batch_ys = ys[start:start + batch_size]
            # compute an average gradient across the batch
            grads = [ridge_error_gradient(x, y, beta, alpha) for (x, y) in zip(batch_xs, batch_ys)]
            avg_grad: Vector = vector_mean(grads)
            # update beta according to the gradient
            beta = gradient_step(beta, avg_grad, lr * -1)  # by negative one b/c we are minimizing
    return beta
