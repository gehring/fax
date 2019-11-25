from functools import partial
import logging

import jax

from fax import converge
from fax import loop

logger = logging.getLogger(__name__)


def make_forward_fixed_point_iteration(
        param_func, default_rtol=1e-4, default_atol=1e-4, default_max_iter=5000,
        default_batched_iter_size=1):
    def _default_solver(init_x, params):
        rtol, atol = converge.adjust_tol_for_dtype(default_rtol,
                                                   default_atol,
                                                   init_x.dtype)

        def convergence_test(x_new, x_old):
            return converge.max_diff_test(x_new, x_old, rtol, atol)

        func = param_func(params)
        sol = loop.fixed_point_iteration(
            init_x=init_x,
            func=func,
            convergence_test=convergence_test,
            max_iter=default_max_iter,
            batched_iter_size=default_batched_iter_size,
        )

        return sol
    return _default_solver


def make_adjoint_fixed_point_iteration(
        param_func, default_rtol=1e-4, default_atol=1e-4, default_max_iter=5000,
        default_batched_iter_size=1):

    # define a flat function to fit the vjp API
    def flat_func(i, x, params):
        return param_func(params)(i, x)

    def _adjoint_iteration_vjp(g, ans, init_xs, params):
        dvalue = g
        del init_xs
        init_dxs = dvalue

        fp_vjp_fn = jax.vjp(partial(flat_func, ans.iterations),
                            ans.value, params)[1]

        def dfp_fn(i, dout):
            del i
            dout = fp_vjp_fn(dout)[0] + dvalue
            return dout

        rtol, atol = converge.adjust_tol_for_dtype(
            default_rtol, default_atol, init_dxs.dtype)

        def convergence_test(x_new, x_old):
            return converge.max_diff_test(x_new, x_old, rtol, atol)

        dsol = loop.fixed_point_iteration(
            init_x=init_dxs,
            func=dfp_fn,
            convergence_test=convergence_test,
            max_iter=default_max_iter,
            batched_iter_size=default_batched_iter_size,
        )

        return fp_vjp_fn(dsol.value)[1], dsol

    return _adjoint_iteration_vjp


def two_phase_solver(param_func, forward_solver=None, default_rtol=1e-4,
                     default_atol=1e-4, default_max_iter=5000,
                     default_batched_iter_size=1):
    """ Create an implicit function of the parameters and define its VJP rule.

    Args:
        param_func: A "parametric" operator (i.e., callable) taking in some
            parameters and returning a function for which we seek a fixed point.
        forward_solver:
        default_rtol (float, optional): The relative tolerance (as used by `np.isclose`).
            Defaults to 1e-4.
        default_atol (float, optional): The absolute tolerance (as used by `np.isclose`).
            Defaults to 1e-4.
        default_max_iter (int, optional): The maximum number of iterations. Defaults to 5000.
        default_batched_iter_size (int, optional):  The number of iterations to be
            unrolled and executed per iterations of `while_loop` op. Defaults to 1.

    Returns:
        callable: Binary callable with signature ``f(x_0, params)`` returning a solution ``x``
            such that ``param_func(params)(x) = x``. The returned callabled is registered as a
            ``jax.custom_transform`` with its associated VJP rule so that it can be composed with
            other functions in and end-to-end fashion.
    """
    # if no solver is specified, create a default solver
    if forward_solver is None:
        forward_solver = make_forward_fixed_point_iteration(
            param_func, default_rtol=default_rtol, default_atol=default_atol,
            default_max_iter=default_max_iter, default_batched_iter_size=default_batched_iter_size)

    adjoint_iteration_vjp = make_adjoint_fixed_point_iteration(
        param_func, default_rtol=default_rtol, default_atol=default_atol,
        default_max_iter=default_max_iter, default_batched_iter_size=default_batched_iter_size)

    @jax.custom_transforms
    def two_phase_op(init_xs, params):
        return forward_solver(init_xs, params)

    def two_phase_vjp(g, ans, init_xs, params):
        dvalue, dconverged, diter, dprev_value = g
        # these tensors are returned only for monitoring and have no defined gradient
        del dconverged, diter, dprev_value
        return adjoint_iteration_vjp(dvalue, ans, init_xs, params)[0]

    jax.defvjp(two_phase_op, None, two_phase_vjp)

    return two_phase_op
