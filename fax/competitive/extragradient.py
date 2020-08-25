from typing import Callable

import jax.experimental.optimizers
from jax import numpy as np, tree_util

import fax.competitive.sgd
from fax.jax_utils import add


def adam_extragradient_optimizer(step_size_x, step_size_y, b1=0.3, b2=0.2, eps=1e-8) -> (Callable, Callable, Callable):
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
        deltas, optimizer_state = fax.competitive.sgd.adam_step(b1, b2, eps, step_sizes, grads, optimizer_state, step)

        x_bar = add(x0, deltas)

        grads = grad_fns(*x_bar)  # the gradient is evaluated at x_bar
        deltas, optimizer_state = fax.competitive.sgd.adam_step(b1, b2, eps, step_sizes, grads, optimizer_state, step)
        x1 = add(x0, deltas)  # but applied at x_0

        return x1, optimizer_state

    def get_params(state):
        x, _optimizer_state = state
        return x

    return init, update, get_params
