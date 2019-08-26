import collections

import jax
import jax.numpy as np
from jax.experimental import optimizers

from fax import converge
from fax import loop

CGAState = collections.namedtuple("CGAState", "x y delta_x delta_y")


def make_mixed_jvp(f, x, y, reversed=False):
    """Make a mixed jacobian-vector product function
    Args:
        f (callable): Binary callable with signature f(x,y)
        x (numpy.ndarray): First argument to f
        y (numpy.ndarray): Second argument to f
        reversed (bool, optional): Take Dyx if False, Dxy if True. Defaults to False.
    Returns:
        callable: Unary callable 'jvp(v)' taking a numpy.ndarray as input.
    """
    if reversed is not True:
        given = y
        gradfun = jax.grad(f, 0)

        def frozen_grad(y):
            return gradfun(x, y)
    else:
        given = x
        gradfun = jax.grad(f, 1)

        def frozen_grad(x):
            return gradfun(x, y)

    def _jvp(v):
        return jax.jvp(frozen_grad, (given,), (v,))[1]

    return _jvp


def make_mixed_hessian(d, first_args, second_args):
    gradfun = jax.grad(d, argnums=first_args)
    return jax.jacfwd(gradfun, argnums=second_args)


def full_solve_cga(step_size_f, step_size_g, f, g):
    """CGA using a naive implementation which build the full hessians."""
    step_size_f = optimizers.make_schedule(step_size_f)
    step_size_g = optimizers.make_schedule(step_size_g)

    def init(inputs):
        return CGAState(
            x=inputs[0],
            y=inputs[1],
            delta_x=np.zeros_like(inputs[0]),
            delta_y=np.zeros_like(inputs[1]),
        )

    def update(i, grads, inputs):
        if len(inputs) < 4:
            x, y = inputs
            delta_x = None
            delta_y = None
        else:
            x, y, delta_x, delta_y = inputs

        grad_xf, grad_yg = grads
        eta_f = step_size_f(i)
        eta_g = step_size_g(i)

        Dxyf = make_mixed_hessian(f, 0, 1)(x, y)
        Dyxg = make_mixed_hessian(g, 1, 0)(x, y)

        bx = grad_xf + eta_f * np.dot(Dxyf, grad_yg)
        delta_x = np.linalg.solve(
            np.eye(x.shape[0]) - eta_f**2 * np.dot(Dxyf, Dyxg),
            bx,
        )

        by = grad_yg + eta_g * np.dot(Dyxg, grad_xf)
        delta_y = np.linalg.solve(
            np.eye(y.shape[0]) - eta_g**2 * np.dot(Dyxg, Dxyf),
            by,
        )

        x = x + eta_f * delta_x
        y = y + eta_g * delta_y
        return CGAState(x, y, delta_x, delta_y)

    def get_params(state):
        return state[:2]

    return init, update, get_params


def cga(step_size_f, step_size_g, f, g, linear_op_solver=None,
        default_max_iter=1000):

    if linear_op_solver is None:
        def default_convergence_test(x_new, x_old):
            rtol, atol = converge.adjust_tol_for_dtype(1e-10, 1e-10,
                                                       x_new.dtype)
            return converge.max_diff_test(x_new, x_old, rtol, atol)

        def default_solver(linear_op, bvec, init_x=None):
            if init_x is None:
                init_x = bvec
            return loop.fixed_point_iteration(
                init_x=init_x,
                func=lambda i, x: linear_op(x) + bvec,
                convergence_test=default_convergence_test,
                max_iter=default_max_iter,
            )
        linear_op_solver = default_solver

    step_size_f = optimizers.make_schedule(step_size_f)
    step_size_g = optimizers.make_schedule(step_size_g)

    def init(inputs):
        return CGAState(
            x=inputs[0],
            y=inputs[1],
            delta_x=np.zeros_like(inputs[0]),
            delta_y=np.zeros_like(inputs[1]),
        )

    def update(i, grads, inputs):
        if len(inputs) < 4:
            x, y = inputs
            delta_x = None
            delta_y = None
        else:
            x, y, delta_x, delta_y = inputs

        grad_xf, grad_yg = grads
        eta_f = step_size_f(i)
        eta_g = step_size_g(i)

        jvp_xyf = make_mixed_jvp(f, x, y)
        jvp_yxg = make_mixed_jvp(g, x, y, reversed=True)

        bx = grad_xf + eta_g * jvp_xyf(grad_yg)
        delta_x = linear_op_solver(
            linear_op=lambda x: (eta_g * eta_f) * jvp_xyf(jvp_yxg(x)),
            bvec=bx,
            init_x=delta_x).value

        by = grad_yg + eta_f * jvp_yxg(grad_xf)
        delta_y = linear_op_solver(
            linear_op=lambda x: (eta_g * eta_f) * jvp_yxg(jvp_xyf(x)),
            bvec=by,
            init_x=delta_y).value

        x = x + eta_f * delta_x
        y = y + eta_g * delta_y
        return CGAState(x, y, delta_x, delta_y)

    def get_params(state):
        return state[:2]

    return init, update, get_params


def cga_iteration(init_values, f, g, convergence_test, max_iter, step_size_f,
                  step_size_g=None, linear_op_solver=None, batched_iter_size=1,
                  unroll=False, use_full_matrix=False):
    """Run competitive gradient ascent until convergence or some max iteration.

    Use this function to find a fixed point of the competitive gradient
    ascent (CGA) update by repeatedly applying CGA to a candidate solution.
    This is done until the solution converges or until the maximum number of
    iterations, `max_iter` is reached.

    NOTE: if the maximum number of iterations is reached, the convergence
    will not be checked on the final application of `func` and the solution
    will always be marked as not converged when `unroll` is `False`.

    Args:
        init_values: a tuple of type `(a, b)` corresponding to the types
            accepted by `f` and `g`.
        f (callable): The function we which to maximize with type
            `a, b -> float`.
        g (callable): The "opposing" function which is also maximized with type
            `a, b -> float`.
        convergence_test (callable): A two argument function of type
            `(a,b), (a, b) -> bool` that takes in the newest solution and the
            previous solution and returns `True` if they have converged. The
            optimization will stop and return when `True` is returned.
        max_iter (int or None): The maximum number of iterations.
        step_size_f: The step size used by CGA for `f`. This can be a scalar or
            a callable taking in the current iteration and returning a scalar.
            If no step size is given for `g`, then `step_size_f` is also used
            for `g`.
        step_size_g (optional): The step size used by CGA for `g`. Like
            `step_size_f`, this can be a scalar or a callable. If no step size
            is given for `g`, then `step_size_f` is used.
        linear_op_solver (callable, optional): This is a function which outputs
            the solution to `x = Ax + b` when given a callable linear operator
            representing the matrix-vector product `Ax` and an array `b`. If
            `None` is given, then a simple fixed point iteration solver is used.
        batched_iter_size (int, optional): The number of iterations to be
            unrolled and executed per iterations of `while_loop` op. Convergence
            is only tested at the beginning of each batch. Set this to a number
            larger than 1 to reduce the number of times convergence is checked
            and to potentially allow for the graph of the unrolled batch to be
            more aggressively optimized.
        unroll (bool, optional): If True, use a normal python while loop, i.e.,
            unrolled ops. This enables back-propagating through the iterations.

            NOTE: due to current limitations in `JAX`, when `unroll` is `True`,
            convergence is ignored and the loop always runs for the maximum
            number of iterations. Additionally, compilation times can be long
            when running for a large number of iterations as a result.
        use_full_matrix (bool, optional): Use a CGA implementation which uses
            full hessians instead of potentially more efficient jacobian-vector
            products. This is useful for debugging and might provide a small
            performance boost when the dimensions are small. If set to True,
            then, if provided, the `linear_op_solver` is ignored.

    Returns:
        FixedPointSolution: A named tuple containing the results of the
            optimization. The tuple contains the attributes `value`
            (the final solution tuple), `converged` (a bool indicating whether
            convergence was achieved), `iterations` (the number of iterations
            used), and `previous_value` (the value of the solution on the
            previous iteration). The previous value satisfies
            `sol.value=step_cga(sol.previous_value)` and allows us to log the
            size of the last step if desired.
    """

    if use_full_matrix:
        cga_init, cga_update, get_params = full_solve_cga(
            step_size_f=step_size_f,
            step_size_g=step_size_g or step_size_f,
            f=f,
            g=g,
        )
    else:
        cga_init, cga_update, get_params = cga(
            step_size_f=step_size_f,
            step_size_g=step_size_g or step_size_f,
            f=f,
            g=g,
            linear_op_solver=linear_op_solver,
        )

    grad_yg = jax.grad(g, 1)
    grad_xf = jax.grad(f, 0)

    def step(i, inputs):
        x, y = inputs[:2]
        grads = (grad_xf(x, y), grad_yg(x, y))
        return cga_update(i, grads, inputs)

    def cga_convergence_test(x_new, x_old):
        return convergence_test(x_new[:2], x_old[:2])

    solution = loop.fixed_point_iteration(
        init_x=cga_init(init_values),
        func=step,
        convergence_test=cga_convergence_test,
        max_iter=max_iter,
        batched_iter_size=batched_iter_size,
        unroll=unroll,
    )
    return solution._replace(
        value=get_params(solution.value),
        previous_value=get_params(solution.previous_value),
    )
