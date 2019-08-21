import logging

import jax

from fax import converge
from fax import loop

logger = logging.getLogger(__name__)


def two_phase_solver(param_func, forward_solver=None, default_rtol=1e-4,
                     default_atol=1e-4, default_max_iter=5000,
                     default_batched_iter_size=1):
    """

    Args:
        param_func:
        forward_solver:
        default_rtol:
        default_atol:
        default_max_iter:
        default_batched_iter_size:

    Returns:

    """
    # if no solver is specified, create a default solver
    if forward_solver is None:
        def default_solver(init_x, params):
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

        forward_solver = default_solver

    # define a flat function to fit the vjp API
    def flat_func(x, params):
        return param_func(params)(x)

    @jax.custom_transforms
    def two_phase_op(init_xs, params):
        return forward_solver(init_xs, params)

    def two_phase_vjp(g, ans, init_xs, params):
        dvalue, dconverged, diter, dprev_value = g
        # these tensors are returned only for monitoring and have no
        # defined gradient
        del dconverged, diter, dprev_value
        init_dxs = dvalue

        fp_vjp_fn = jax.vjp(flat_func, ans.value, params)[1]

        def dfp_fn(dout):
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

        return fp_vjp_fn(dsol.value)[1]

    jax.defvjp(two_phase_op, None, two_phase_vjp)

    return two_phase_op
