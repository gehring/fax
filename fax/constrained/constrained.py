""" Optimization methods for parametric nonlinear equality constrained problems.
"""
import collections

from scipy.optimize import minimize

import jax
from jax import lax
from jax import jit
from jax import grad
from jax import jacrev
import jax.numpy as np
from jax import tree_util
from jax.experimental import optimizers
from jax.flatten_util import ravel_pytree

from fax import math
from fax import converge
from fax.competitive import cg
from fax.competitive import cga
from fax.loop import fixed_point_iteration
from fax.implicit.twophase import make_adjoint_fixed_point_iteration
from fax.implicit.twophase import make_forward_fixed_point_iteration


ConstrainedSolution = collections.namedtuple(
    "ConstrainedSolution",
    "value converged iterations"
)


def default_convergence_test(x_new, x_old):
    min_type = converge.tree_smallest_float_dtype(x_new)
    rtol, atol = converge.adjust_tol_for_dtype(1e-10, 1e-10, min_type)
    return converge.max_diff_test(x_new, x_old, rtol, atol)


def implicit_ecp(
        objective, equality_constraints, initial_values, lr_func, max_iter=500,
        convergence_test=default_convergence_test, batched_iter_size=1, optimizer=optimizers.sgd,
        tol=1e-6):
    """Use implicit differentiation to solve a nonlinear equality-constrained program of the form:

    max f(x, θ) subject to h(x, θ) = 0 .

    We perform a change of variable via the implicit function theorem and obtain the unconstrained
    program:

    max f(φ(θ), θ) ,

    where φ is an implicit function of the parameters θ such that h(φ(θ), θ) = 0.

    Args:
        objective (callable): Binary callable with signature `f(x, θ)`
        equality_constraints (callble): Binary callable with signature `h(x, θ)`
        initial_values (tuple): Tuple of initial values `(x_0, θ_0)`
        lr_func (scalar or callable): The step size used by the unconstrained optimizer. This can
            be a scalar ora callable taking in the current iteration and returning a scalar.
        max_iter (int, optional): Maximum number of outer iterations. Defaults to 500.
        convergence_test (callable): Binary callable with signature `callback(new_state, old_state)`
            where `new_state` and `old_state` are tuples of the form `(x_k^*, θ_k)` such that
            `h(x_k^*, θ_k) = 0` (and with `k-1` for `old_state`). The default convergence test
            returns `true` if both elements of the tuple have not changed within some tolerance.
        batched_iter_size (int, optional):  The number of iterations to be
            unrolled and executed per iterations of the `while_loop` op for the forward iteration
            and the fixed-point adjoint iteration. Defaults to 1.
        optimizer (callable, optional): Unary callable waking a `lr_func` as a argument and
            returning an unconstrained optimizer. Defaults to `jax.experimental.optimizers.sgd`.
        tol (float, optional): Tolerance for the forward and backward iterations. Defaults to 1e-6.

    Returns:
        fax.loop.FixedPointSolution: A named tuple containing the solution `(x, θ)` as as the
            `value` attribute, `converged` (a bool indicating whether convergence was achieved),
            `iterations` (the number of iterations used), and `previous_value`
            (the value of the solution on the previous iteration). The previous value satisfies
            `sol.value=func(sol.previous_value)` and allows us to log the size
            of the last step if desired.
    """
    def _objective(*args):
        return -objective(*args)

    def make_fp_operator(params):
        def _fp_operator(i, x):
            del i
            return x + equality_constraints(x, params)
        return _fp_operator

    constraints_solver = make_forward_fixed_point_iteration(
        make_fp_operator, default_max_iter=max_iter, default_batched_iter_size=batched_iter_size,
        default_atol=tol, default_rtol=tol)

    adjoint_iteration_vjp = make_adjoint_fixed_point_iteration(
        make_fp_operator, default_max_iter=max_iter, default_batched_iter_size=batched_iter_size,
        default_atol=tol, default_rtol=tol)

    opt_init, opt_update, get_params = optimizer(step_size=lr_func)

    grad_objective = grad(_objective, (0, 1))

    def update(i, values):
        old_xstar, opt_state = values
        old_params = get_params(opt_state)

        forward_solution = constraints_solver(old_xstar, old_params)

        grads_x, grads_params = grad_objective(forward_solution.value, get_params(opt_state))

        ybar, _ = adjoint_iteration_vjp(
            grads_x, forward_solution, old_xstar, old_params)

        implicit_grads = tree_util.tree_multimap(
            lax.add, grads_params, ybar)

        opt_state = opt_update(i, implicit_grads, opt_state)

        return forward_solution.value, opt_state

    def _convergence_test(new_state, old_state):
        x_new, params_new = new_state[0], get_params(new_state[1])
        x_old, params_old = old_state[0], get_params(old_state[1])
        return convergence_test((x_new, params_new), (x_old, params_old))

    x0, init_params = initial_values
    opt_state = opt_init(init_params)

    solution = fixed_point_iteration(init_x=(x0, opt_state),
                                     func=update,
                                     convergence_test=jit(_convergence_test),
                                     max_iter=max_iter,
                                     batched_iter_size=batched_iter_size,
                                     unroll=False)
    return solution._replace(
        value=(solution.value[0], get_params(solution.value[1])),
        previous_value=(solution.previous_value[0], get_params(solution.previous_value[1])),
    )


def make_lagrangian(func, equality_constraints):
    """Make a Lagrangian function from an objective function `func` and `equality_constraints`

    Args:
        func (callable): Unary callable with signature `f(x, *args, **kwargs)`
        equality_constraints (callable): Unary callable with signature `h(x, *args, **kwargs)`

    Returns:
        tuple: Triple of callables (init_multipliers, lagrangian, get_params)
    """
    def init_multipliers(params, *args, **kwargs):
        h = jax.eval_shape(equality_constraints, params, *args, **kwargs)
        multipliers = tree_util.tree_map(lambda x: np.zeros(x.shape, x.dtype), h)
        return params, multipliers

    def lagrangian(params, multipliers, *args, **kwargs):
        h = equality_constraints(params, *args, **kwargs)
        return -func(params, *args, **kwargs) + math.pytree_dot(multipliers, h)

    def get_params(opt_state):
        return opt_state[0]

    return init_multipliers, lagrangian, get_params


def cga_lagrange_min(lagrangian, lr_func, lr_multipliers=None,
                     linear_op_solver=None, solve_order='alternating'):
    """Use competitive gradient ascent to solve a nonlinear equality-constrained program:

    max f(x) subject to h(x) = 0,

    by forming the lagrangian L(x, λ) = f(x) - λ^⊤ h(x) and finding a saddle-point solution to:

    max_x min_λ L(x, λ)

    Args:
        lagrangian (callable): Binary callable with signature `L(x, λ, *args, **kwargs)`.
        lr_func (scalar or callable): The step size used by CGA for `f`. This can be a scalar or
            a callable taking in the current iteration and returning a scalar.
        lr_multipliers (scalar or callable, optional): Step size for the dual updates.
            Defaults to None. If no step size is given for `lr_multipliers`, then
            `lr_func` is also used for `lr_multipliers`.
        linear_op_solver (callable, optional): This is a function which outputs
            the solution to `x = Ax + b` when given a callable linear operator
            representing the matrix-vector product `Ax` and an array `b`. If
            `None` is given, then a simple fixed point iteration solver is used.
        solve_order (str, optional): Specifies how the updates for each player are solved for.
            Should be one of

            - 'both' (default): Solves the linear system for each player (eq. 3 of Schaefer 2019)
            - 'yx' : Solves for the player behind `y` then updates `x`
            - 'xy' : Solves for the player behind `x` then updates `y`
            - 'alternate': Solves for `x` update `y`, next iteration solves for y and update `x`

            Defaults to 'both'

    Returns:
        tuple: Triple of callables  `(lagrange_init, lagrange_update, get_params)`
    """
    def neg_lagrangian(*args, **kwargs):
        return -lagrangian(*args, **kwargs)

    cga_init, cga_update, cga_get_params = cga.cga(
        step_size_f=lr_func,
        step_size_g=lr_func if lr_multipliers is None else lr_multipliers,
        f=lagrangian,
        g=neg_lagrangian,
        linear_op_solver=linear_op_solver or cg.fixed_point_solve,
        solve_order=solve_order
    )

    def lagrange_init(lagrange_params):
        return cga_init(lagrange_params)

    def lagrange_update(i, grads, opt_state, *args, **kwargs):
        """Update the optimization state of the Lagrangian.

        Args:
            i: iteration step
            grads: tuple of pytrees where the first element is a pytree of the
                gradients of the Lagrangian with respect to the parameters and
                the seconds is a pytree of the gradients with respect to the
                Lagrangian multipliers.
            opt_state: the packed optimization state returned by the previous
                call to this method or from the first call to `lagrange_init`.

        Returns:
            An new packed optimization state with the updated parameters and
            Lagrange multipliers.
        """
        params_grad, multipliers_grad = grads
        grads = (params_grad, tree_util.tree_map(lax.neg, multipliers_grad))
        return cga_update(i, grads, opt_state, *args, **kwargs)

    def get_params(opt_state):
        return cga_get_params(opt_state)

    return lagrange_init, lagrange_update, get_params


def cga_ecp(
        objective, equality_constraints, initial_values, lr_func, lr_multipliers=None,
        linear_op_solver=None, solve_order='alternating', max_iter=500,
        convergence_test=default_convergence_test, batched_iter_size=1,
):
    """Use CGA to solve a nonlinear equality-constrained program of the form:

    max f(x, θ) subject to h(x, θ) = 0 .

    We form the lagrangian L(x, θ, λ) = f(x, θ) - λ^⊤ h(x, θ) and try to find a saddle-point in:

    max_{x, θ} min_λ L(x, θ, λ)

    Args:
        objective (callable): Binary callable with signature `f(x, θ)`
        equality_constraints (callble): Binary callable with signature `h(x, θ)`
        initial_values (tuple): Tuple of initial values `(x_0, θ_0)`
        lr_func (scalar or callable): The step size used by CGA for `f`. This can be a scalar or
            a callable taking in the current iteration and returning a scalar.
        lr_multipliers (scalar or callable, optional): Step size for the dual updates.
            Defaults to None. If no step size is given for `lr_multipliers`, then
            `lr_func` is also used for `lr_multipliers`.
        linear_op_solver (callable, optional): This is a function which outputs
            the solution to `x = Ax + b` when given a callable linear operator
            representing the matrix-vector product `Ax` and an array `b`. If
            `None` is given, then a simple fixed point iteration solver is used. Used to solve for
            the matrix inverses in the CGA algorithm
        solve_order (str, optional): Specifies how the updates for each player are solved for.
            Should be one of

            - 'both' (default): Solves the linear system for each player (eq. 3 of Schaefer 2019)
            - 'yx' : Solves for the player behind `y` then updates `x`
            - 'xy' : Solves for the player behind `x` then updates `y`
            - 'alternate': Solves for `x` update `y`, next iteration solves for y and update `x`

            Defaults to 'both'
        max_iter (int): Maximum number of outer iterations. Defaults to 500.
        convergence_test (callable): Binary callable with signature `callback(new_state, old_state)`
            where `new_state` and `old_state` are nested tuples of the form `((x_k, θ_k),  λ_k)`
            The default convergence test returns `true` if all elements of the tuple have not
            changed within some tolerance.
        batched_iter_size (int, optional):  The number of iterations to be
            unrolled and executed per iterations of the `while_loop` op for the forward iteration
            and the fixed-point adjoint iteration. Defaults to 1.

    Returns:
        fax.loop.FixedPointSolution: A named tuple containing the solution `(x, θ)` as as the
            `value` attribute, `converged` (a bool indicating whether convergence was achieved),
            `iterations` (the number of iterations used), and `previous_value`
            (the value of the solution on the previous iteration). The previous value satisfies
            `sol.value=func(sol.previous_value)` and allows us to log the size
            of the last step if desired.
    """
    def _objective(variables):
        return -objective(*variables)

    def _equality_constraints(variables):
        return -equality_constraints(*variables)

    init_mult, lagrangian, _ = make_lagrangian(_objective, _equality_constraints)
    lagrangian_variables = init_mult(initial_values)

    if lr_multipliers is None:
        lr_multipliers = lr_func

    opt_init, opt_update, get_params = cga_lagrange_min(
        lagrangian, lr_func, lr_multipliers, linear_op_solver, solve_order)

    @jit
    def update(i, opt_state):
        grad_fn = grad(lagrangian, (0, 1))
        grads = grad_fn(*get_params(opt_state))
        return opt_update(i, grads, opt_state)

    solution = fixed_point_iteration(init_x=opt_init(lagrangian_variables),
                                     func=update,
                                     convergence_test=jit(convergence_test),
                                     max_iter=max_iter,
                                     batched_iter_size=batched_iter_size,
                                     unroll=False)
    return solution._replace(
        value=get_params(solution.value)[0],
        previous_value=get_params(solution.previous_value)[0],
    )


def slsqp_ecp(objective, equality_constraints, initial_values, max_iter=500, ftol=1e-6):
    """Interface to the Sequential Least Squares Programming in scipy.optimize.minimize

    The SLSQP approach is described in:

    Kraft, D. A software package for sequential quadratic programming. 1988.
    DFVLR-FB 88-28, DLR German Aerospace Center  Institute for Flight Mechanics, Koln, Germany.

    Args:
        objective (callable): Binary callable with signature `f(x, θ)`
        equality_constraints (callble): Binary callable with signature `h(x, θ)`
        initial_values (tuple): Tuple of initial values `(x_0, θ_0)`
        max_iter (int): Maximum number of outer iterations. Defaults to 500.
        ftol (float, optional): Tolerance in the value of the objective for the stopping criterion.
            Defaults to 1e-6.

    Returns:
        ConstrainedSolution: A namedtuple with fields 'value', 'iterations' and 'converged'
    """
    flat_initial_values, unravel = ravel_pytree(initial_values)

    @jit
    def _objective(variables):
        unraveled = unravel(variables)
        return -objective(*unraveled)

    @jit
    def _equality_constraints(variables):
        return np.ravel(equality_constraints(*unravel(variables)))

    @jit
    def gradfun_objective(variables):
        return grad(_objective)(variables)

    @jit
    def jacobian_constraints(variables):
        return jacrev(_equality_constraints)(variables)

    options = {'maxiter': max_iter, 'ftol': ftol}
    constraints = ({'type': 'eq', 'fun': _equality_constraints, 'jac': jacobian_constraints})
    solution = minimize(_objective, flat_initial_values, method='SLSQP',
                        constraints=constraints, options=options, jac=gradfun_objective)

    return ConstrainedSolution(
        value=unravel(solution.x), iterations=solution.nit, converged=solution.success)
