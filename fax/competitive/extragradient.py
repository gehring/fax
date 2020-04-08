from typing import Callable

import jax.experimental.optimizers
from jax import np


def extragradient_optimizer(*args, **kwargs) -> (Callable, Callable, Callable):
    return rprop_extragradient_optimizer(*args, **kwargs, use_rprop=False)


def rprop_extragradient_optimizer(step_size_x, step_size_y, proj_x=lambda x: x, proj_y=lambda y: y, use_rprop=True) -> (Callable, Callable, Callable):
    """Provides an optimizer interface to the extra-gradient method

    We are trying to find a pair (x*, y*) such that:

    f(x*, y) ≤ f(x*, y*) ≤ f(x, y*), ∀ x ∈ X, y ∈ Y

    where X and Y are closed convex sets.

    Args:
        init_values:
        step_size_x: TODO
        step_size_y: TODO
        f: Saddle-point function
        convergence_test:  TODO
        max_iter:  TODO
        batched_iter_size:  TODO
        unroll:  TODO
        proj_x: Projection on the convex set X
        proj_y: Projection on the convex set Y
        eps: rms prop eps
        gamma: rms prop gamma

    """
    step_size_x = jax.experimental.optimizers.make_schedule(step_size_x)
    step_size_y = jax.experimental.optimizers.make_schedule(step_size_y)

    def init(init_values):
        x0, y0 = init_values
        assert len(x0.shape) == len(y0.shape) == 1
        return (x0, y0), np.ones(x0.shape[0] + y0.shape[0])

    def update(i, grads, state):
        (x0, y0), grad_state = state
        step_sizes = (jax.experimental.optimizers.make_schedule(step_size_x), jax.experimental.optimizers.make_schedule(step_size_y))

        delta_x, delta_y, _ = sign_adaptive_step(step_sizes, grads, grad_state, x0, y0, i, use_rprop=use_rprop)

        xbar = proj_x(x0 - delta_x)
        ybar = proj_y(y0 + delta_y)

        delta_x, delta_y, _ = sign_adaptive_step(step_sizes, grads, grad_state, xbar, ybar, i, use_rprop=use_rprop)
        x1 = proj_x(x0 - delta_x)
        y1 = proj_y(y0 + delta_y)

        return (x1, y1), grad_state

    def get_params(state):
        x, _ = state
        return x

    return init, update, get_params


def sign_adaptive_step(step_size, grads, grad_state, x, y, i, use_rprop=True):
    grad_x, grad_y = grads
    step_size_x, step_size_y = step_size

    grad_x0 = grad_x(x, y)
    grad_y0 = grad_y(x, y)
    # the next part is to avoid ifs
    #  d |  d + 1 |  d - 1
    #  1 |    2   |    0
    # -1 |    0   |   -2
    if use_rprop:
        eta_plus = 1.2
        eta_minus = 0.5
        direction = np.sign(grad_state * np.concatenate((grad_x0, grad_y0)))
        step_improvement_rate = (direction + 1) * eta_plus / 2. + (1 - direction) * eta_minus / 2
        eta_x = step_size_x(i) * step_improvement_rate[:grad_x0.shape[0]]
        eta_y = step_size_y(i) * step_improvement_rate[grad_x0.shape[0]:]
        grad_state = np.concatenate((grad_x0, grad_y0))
    else:
        grad_state = None
        eta_x = step_size_x(i)
        eta_y = step_size_y(i)

    delta_x = eta_x * grad_x0
    delta_y = eta_y * grad_y0
    return delta_x, delta_y, grad_state


def rms_prop_step():
    # grad_state = grad_state * gamma + grad_x0 ** 2 * (1. - gamma)
    # delta_x = eta_x * grad_x0 / np.sqrt(grad_state + eps)
    # avg_sq_grad_y = avg_sq_grad_y * gamma + grad_y0 ** 2 * (1. - gamma)
    # delta_y = eta_y * grad_y0 / np.sqrt(avg_sq_grad_y + eps)
    raise NotImplementedError
