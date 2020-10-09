import functools
import operator

import jax
import jax.numpy as np

from fax import linalg


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 3, 4))
def root_solve(func, init_xs, params, solver, rev_solver=None):
    """ Create a differentiable implicit function for the root `func`.

    Args:
        func Callable[[PyTree, PyTree], PyTree]: A "parametric" function taking
            in a pytree `x` and a pytree `params` and outputs an arbitrary
            pytree `y`.
        init_xs PyTree: The initial "guess" for the root of
            `func(x, params) = 0`.
        params: The parameters to use when evaluating `func`.
        solver: A solver which takes in some initial guess and some parameters,
            and returns a pytree `x` such that `func(x, params) = 0`.
        rev_solver (optional): A matrix free linear solver which outputs a tuple
            containing the found solution and a dict with any additional
            information about the status of the solution.

    Returns:
        The solution to the parametric root with reverse differentiation rules
        defined using the implicit function theorem.
    """
    del func, rev_solver
    init_xs = jax.tree_map(jax.lax.stop_gradient, init_xs)
    params = jax.tree_map(jax.lax.stop_gradient, params)
    return solver(init_xs, params)


def root_solve_fwd(func, init_xs, params, solver, rev_solver):
    sol = root_solve(func, init_xs, params, solver, rev_solver)
    return sol, (sol, params)


def root_solve_rev(func, solver, rev_solver, res, sol_bar):
    del solver
    sol, params = res
    if rev_solver is None:
        rev_solver = linalg.gmres

    # Define a function which efficiently computes the vector-matrix product of
    # a vector and the jacobian with respect to the parameters.
    _, dsol_vjp = jax.vjp(lambda x: func(x, params), sol)

    # Solve for the `sol_bar^T J_sol` where `J_sol` is the jacobian of
    # 1param_func1 with respect to `sol`. Note, this step only needs the
    # Jacobian's vector-matrix product and should avoid building the full
    # Jacobian matrix.
    dsol, _ = rev_solver(jax.jit(lambda v: dsol_vjp(v)[0]), sol_bar)

    _, dparam_vjp = jax.vjp(lambda p: func(sol, p), params)
    dparam = jax.tree_map(operator.neg, dparam_vjp(dsol)[0])
    return jax.tree_map(np.zeros_like, sol), dparam


root_solve.defvjp(root_solve_fwd, root_solve_rev)
