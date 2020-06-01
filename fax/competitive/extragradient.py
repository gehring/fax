from typing import Callable

import jax.experimental.optimizers
from jax import np


@jax.experimental.optimizers.optimizer
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
        mean_avg = np.zeros_like(initial_values)
        var_avg = np.zeros_like(initial_values)
        return initial_values, mean_avg, var_avg

    def update(step, grad_fns, state):
        (x0, y0), grad_state = state
        step_sizes = step_size_x(step), step_size_y(step)

        delta_x, delta_y, grad_state = adam_step(b1, b2, eps, step_sizes, grad_fns, grad_state, x0, y0, step)
        x_bar = x0 - delta_x
        y_var = y0 + delta_y

        delta_x, delta_y, grad_state = adam_step(b1, b2, eps, step_sizes, grad_fns, grad_state, x_bar, y_var, step)
        x1 = x0 - delta_x
        y1 = y0 + delta_y

        return (x1, y1), grad_state

    def get_params(state):
        x, _mean_avg, _var_avg = state
        return x

    return init, update, get_params


def adam_step(beta1, beta2, eps, step_sizes, grads_fn, grad_state, x, y, step):
    exp_avg, exp_avg_sq = grad_state
    step_size_x, step_size_y = step_sizes
    grad_x0, grad_y0 = grads_fn(x, y)
    grads = np.concatenate((grad_x0, grad_y0))

    bias_correction1 = 1 - beta1 ** (step + 1)
    bias_correction2 = 1 - beta2 ** (step + 1)

    exp_avg = exp_avg * beta1 + (1 - beta1) * grads
    exp_avg_sq = (beta2 * exp_avg_sq) + (1 - beta2) * np.square(grads)

    corrected_moment = exp_avg / bias_correction1
    corrected_second_moment = exp_avg_sq / bias_correction2

    denom = np.sqrt(corrected_second_moment) + eps
    step_improvement = corrected_moment / denom

    delta_x = step_size_x * step_improvement[:grad_x0.shape[0]]
    delta_y = step_size_y * step_improvement[grad_x0.shape[0]:]

    grad_state = exp_avg, exp_avg_sq
    return delta_x, delta_y, grad_state

