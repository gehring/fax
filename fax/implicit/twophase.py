import functools
import logging

import jax
import jax.numpy as np

from fax import converge
from fax import loop

logger = logging.getLogger(__name__)


def default_solver(param_func, default_rtol=1e-10, default_atol=1e-10,
                   default_max_iter=5000, default_batched_iter_size=1):
    def _default_solve(init_x, params):
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

        return sol.value
    return _default_solve


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 3))
def two_phase_solve(param_func, init_xs, params, solvers=()):
    if solvers:
        fwd_solver = solvers[0]
    else:
        fwd_solver = default_solver(param_func)

    return fwd_solver(init_xs, params)


def two_phase_fwd(param_func, init_xs, params, solvers):
    sol = two_phase_solve(param_func, init_xs, params, solvers)
    return sol, (sol, params)


def two_phase_rev(param_func, solvers, res, sol_bar):

    def param_dfp_fn(packed_params):
        v, p, dvalue = packed_params
        _, fp_vjp_fn = jax.vjp(lambda x: param_func(p)(x), v)

        def dfp_fn(dout):
            dout = fp_vjp_fn(dout)[0] + dvalue
            return dout

        return dfp_fn

    sol, params = res
    dsol = two_phase_solve(param_dfp_fn,
                           sol_bar,
                           (sol, params, sol_bar),
                           solvers[1:])
    _, dparam_vjp = jax.vjp(lambda p: param_func(p)(sol), params)
    return jax.tree_map(np.zeros_like, sol), dparam_vjp(dsol)[0]


two_phase_solve.defvjp(two_phase_fwd, two_phase_rev)
