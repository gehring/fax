from typing import Callable

import jax.experimental.optimizers
from jax import np, tree_util

from fax.jax_utils import add, division, mul, division_constant, square, make_exp_smoothing


def adam_descentascent_optimizer(step_size_x, step_size_y, b1=0.3, b2=0.2, eps=1e-8) -> (Callable, Callable, Callable):
    """Construct optimizer triple for Adam.

    Args:
      step_size_x: positive scalar, or a callable representing a step size schedule
        that maps the iteration index to positive scalar for the first player.
      step_size_y: positive scalar, or a callable representing a step size schedule
        that maps the iteration index to positive scalar for the second player.
      b1: optional, a positive scalar value for beta_1, the exponential decay rate
        for the first moment estimates (default 0.3).
      b2: optional, a positive scalar value for beta_2, the exponential decay rate
        for the second moment estimates (default 0.2).
      eps: optional, a positive scalar value for epsilon, a small constant for
        numerical stability (default 1e-8).

    Returns:
      An (init_fun, update_fun, get_params) triple.
    """
    step_size_x = jax.experimental.optimizers.make_schedule(step_size_x)
    step_size_y = jax.experimental.optimizers.make_schedule(step_size_y)

    def init(initial_values):
        mean_avg = tree_util.tree_map(lambda x: np.zeros(x.shape, x.dtype), initial_values)
        var_avg = tree_util.tree_map(lambda x: np.zeros(x.shape, x.dtype), initial_values)
        return initial_values, (mean_avg, var_avg)

    def update(step, grad_fns, state):
        x0, optimizer_state = state
        step_sizes = - step_size_x(step), step_size_y(step)  # negate the step size so that we do gradient ascent-descent

        grads = grad_fns(*x0)
        deltas, optimizer_state = adam_step(b1, b2, eps, step_sizes, grads, optimizer_state, step)

        x1 = add(x0, deltas)

        return x1, optimizer_state

    def get_params(state):
        x, _optimizer_state = state
        return x

    return init, update, get_params


def adam_step(beta1, beta2, eps, step_sizes, grads, optimizer_state, step):
    exp_avg, exp_avg_sq = optimizer_state

    bias_correction1 = 1 - beta1 ** (step + 1)
    bias_correction2 = 1 - beta2 ** (step + 1)

    exp_avg = tree_util.tree_multimap(make_exp_smoothing(beta1), exp_avg, grads)
    exp_avg_sq = tree_util.tree_multimap(make_exp_smoothing(beta2), exp_avg_sq, square(grads))

    corrected_moment = division_constant(bias_correction1)(exp_avg)
    corrected_second_moment = division_constant(bias_correction2)(exp_avg_sq)

    denom = tree_util.tree_multimap(lambda _var: np.sqrt(_var) + eps, corrected_second_moment)
    step_improvement = division(corrected_moment, denom)
    delta = mul(step_sizes, step_improvement)

    optimizer_state = exp_avg, exp_avg_sq
    return delta, optimizer_state
