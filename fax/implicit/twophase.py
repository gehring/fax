import functools
import logging

import jax
import jax.numpy as np

from fax import converge
from fax import loop

logger = logging.getLogger(__name__)


def default_solver(rtol=1e-10, atol=1e-10, max_iter=5000, batched_iter_size=1):
    """ Create a simple fixed-point iteration solver.

    Args:
        rtol (float, optional): The relative tolerance for convergence.
        atol (float, optional): The absolute tolerance for convergence.
        max_iter (int or None): The maximum number of iterations.
        batched_iter_size (int, optional): The number of iterations to be
            unrolled and executed per iterations of `while_loop` op. See
            `fax.loop.fixed_point_iteration` for details.

    Returns:
        A callable which repeatedly applies `param_func` until converges and
        returns the found fixed-point or the last intermediate solution if the
        maximum number of iterations is reached before convergence.

    """

    def _default_solve(param_func, init_x, params):
        dtype = converge.tree_smallest_float_dtype(init_x)
        adjusted_tol = converge.adjust_tol_for_dtype(rtol, atol, dtype)

        def convergence_test(x_new, x_old):
            return converge.max_diff_test(x_new, x_old, *adjusted_tol)

        func = param_func(params)
        sol = loop.fixed_point_iteration(
            init_x=init_x,
            func=func,
            convergence_test=convergence_test,
            max_iter=max_iter,
            batched_iter_size=batched_iter_size,
        )

        return sol.value
    return _default_solve


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 3))
def two_phase_solve(param_func, init_xs, params, solvers=()):
    """ Create an implicit function of the parameters and define its VJP rule.

    Args:
        param_func: A "parametric" operator (i.e., callable) taking in some
            parameters and returning a function for which we seek a
            fixed-point.
        init_xs: The initial "guess" for the fixed-point of
            `param_func(params)(x) = x`.
        params: The parameters to use when evaluating `param_func`.
        solvers (optional): A sequence of solvers to be used for solving for
            the different fixed-point required for solving the given parametric
            fixed-point and its derivatives. Specifying solvers is optional. If
            a `None` value is encountered in the sequence, the default solver
            will be used.

            The first solver in the sequence is used to solve for the
            parametric fixed point and every subsequent solver is used to
            compute the derivatives of increasing order. For example,
            `solvers[1]` is the solver used when computing the first order
            while `solvers[2]` would be used to solve for the second order
            derivatives.

            Each solver should be a callable taking in a parametric function
            (i.e., a callable which returns a callable) for which we seek a
            fixed-point, the initial guess, and the parameters to use. Except
            for the first solvers, the function given as first argument will
            not be `param_func` but a VJP function derived from `param_func`.

            Formally, a solver is expected to have the following signature:

            ```
            ((b -> (a -> a)) -> a -> b) -> a
            ```

            For `solver[0]`, a is simply the type of `init_xs`, b is the type
            of `param`.

    Returns:
        The solution to the parametric fixed-point with reverse differentiation
        rules defined using the implicit function theorem.
    """

    # If no solver is given or if None is found in its place, use the default
    # fixed-point iteration solver.
    if solvers or solvers[0] is None:
        fwd_solver = solvers[0]
    else:
        fwd_solver = default_solver()

    return fwd_solver(param_func, init_xs, params)


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
