from jax import grad
from jax.experimental import optimizers

from fax import loop


def extra_gradient_iteration(
        init_values, step_size_x, step_size_y, f, convergence_test, max_iter, batched_iter_size=1,
        unroll=False, proj_x=lambda x: x, proj_y=lambda y: y, ):
    """Provides an optimizer interface to the extra-gradient methodm

    TODO: CGA uses f, g functions, while this interface uses only f, is that ok?

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

    """
    step_size_x = optimizers.make_schedule(step_size_x)
    step_size_y = optimizers.make_schedule(step_size_y)

    grad_x = grad(f, 0)
    grad_y = grad(f, 1)

    def step(i, inputs):
        x, y = inputs
        eta_x = step_size_x(i)
        eta_y = step_size_y(i)
        xbar = proj_x(x - eta_x * grad_x(x, y))
        ybar = proj_y(y + eta_y * grad_y(x, y))
        x = proj_x(x - eta_x * grad_x(xbar, ybar))
        y = proj_y(y + eta_y * grad_y(xbar, ybar))
        return x, y

    solution = loop.fixed_point_iteration(
        init_x=init_values,
        func=step,
        convergence_test=convergence_test,
        max_iter=max_iter,
        batched_iter_size=batched_iter_size,
        unroll=unroll,
        f=f,
    )

    return solution
