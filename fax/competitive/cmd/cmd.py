import collections
from fax import math
import jax
import jax.numpy as jnp
from jax import tree_util, jacfwd, random, grad, jvp, jit
from jax.scipy.sparse import linalg
from jax.scipy import linalg as scipy_linalg
# from cmd_helper import DP_pd, DP_inv_pd, inv_D2P_pd, D2P_pd, id_func
from functools import partial

BregmanPotential = collections.namedtuple("BregmanPotential", ["DP", "DP_inv", "D2P", "inv_D2P"])


# AugmentedDP = collections.namedtuple("AugmentedDP", ["DP_primal", "DP_eq", "DP_ineq"])
# AugmentedDPinv = collections.namedtuple("AugmentedDPinv", ["DPinv_primal","DPinv_eq","DPinv_ineq"])
# AugmentedD2P = collections.namedtuple("AugmentedD2P", ["D2P_primal", "D2P_eq", "D2P_ineq"])
# AugmentedD2Pinv = collections.namedtuple("AugmentedD2Pinv", ["D2Pinv_primal","D2Pinv_eq","D2Pinv_ineq"])


def id_func(x):
    return lambda u: jnp.dot(jnp.identity(x.shape[0]), u)

@jit
def hvp(f, primals, tangents):
    return jvp(grad(f), primals, tangents)[1]

@jit
def breg_bound(vec, lb=-1.0, ub=1.0, *args, **kwargs):
    return jnp.sum((- vec + ub) * jnp.log(- vec + ub) + (vec - lb) * jnp.log(vec - lb))


DP_bound = jax.grad(breg_bound, 0)

@jit
def DP_inv_bound(vec, lb=-1.0, ub=1.0):
    return (ub * jnp.exp(vec) + lb) / (1 + jnp.exp(vec))

@jit
def D2P_bound(vec, lb=-1.0, ub=1.0):
    def out(u):
        return jvp(lambda x: DP_bound(x, lb, ub), (vec,), (u,))[1]

    return out

@jit
def inv_D2P_bound(vec, lb=-1.0, ub=1.0):
    if len(jnp.shape(vec)) <= 1:
        def out(u):
            return jnp.dot(jnp.diag(1 / ((1 / (ub - vec)) + (1 / (vec - lb)))), u)
    else:
        def out(u):
            return (1 / ((1 / (ub - vec)) + (1 / (vec - lb)))) * u
    return out


bound_breg = BregmanPotential(DP_bound, DP_inv_bound, D2P_bound, inv_D2P_bound)

@jit
def DP_hand(vec, nx):
    temp = jnp.reshape(vec, (nx, nx))
    return (-jnp.linalg.inv(temp).T + temp).reshape(nx ** 2, 1)

@jit
def matrix_DP_pd(M):
    return -jnp.linalg.slogdet(M)[1]

@jit
def vector_DP_pd(v):
    return jnp.dot(v, jnp.log(v))

@jit
def DP_pd(v):
    m = len(jnp.shape(v))
    if m == 1:
        out = grad(lambda x: jnp.dot(x, jnp.log(x)))(v)
    else:
        out = grad(lambda M: -jnp.linalg.slogdet(M)[1])(v)
    return out

@jit
def vector_DP_inv_pd(v):
    return jnp.exp(v - jnp.ones_like(v))

@jit
def DP_inv_pd(v):
    m = len(jnp.shape(v))
    if m == 1:
        out = vector_DP_inv_pd(v)
    else:
        out = -scipy_linalg.inv(v).T
    return out

@jit
def D2P_pd(v):
    m = len(jnp.shape(v))
    if m == 1:
        def out(u):
            return hvp(vector_DP_pd, (v,), (u,))
    else:
        def out(u):
            return hvp(matrix_DP_pd, (v,), (u,))
    return out

@jit
def inv_D2P_pd(v):
    m = len(jnp.shape(v))
    if m == 1:
        def out(u):
            return jnp.dot(jnp.diag(v), u)
    else:
        def out(u):
            return jnp.dot(jnp.linalg.matrix_power(v, 2).T, u)
    return out

@jit
def D2P_l2(v):
    return lambda x: x

@jit
def default_func(x, *args, **kwargs):
    return None


# TODO: bake in step size to the bregman potential definitions instead of in cmd loop
default_breg = BregmanPotential(lambda v: jax.tree_map(lambda x: x, v),
                                lambda v: jax.tree_map(lambda x: x, v), D2P_l2, D2P_l2)


def make_pd_bregman(step_size=1e-5):
    return BregmanPotential(DP_pd, DP_inv_pd, D2P_pd, inv_D2P_pd)


def make_bound_breg(lb=-1.0, ub=1.0):
    def breg_bound_internal(lb, ub, *args, **kwargs):
        return lambda vec: jnp.sum(
            (- vec + ub) * jnp.log(- vec + ub) + (vec - lb) * jnp.log(vec - lb))

    def DP_bound_internal(lb, ub):
        return jax.grad(breg_bound_internal(lb, ub))

    def DP_inv_bound_internal(lb, ub):
        return lambda vec: (ub * jnp.exp(vec) + lb) / (1 + jnp.exp(vec))

    def D2P_bound_internal(lb, ub):
        def out(vec):
            return lambda u: jvp(DP_bound_internal(lb, ub), (vec,), (u,))[1]

        return out

    def inv_D2P_bound_internal(lb, ub):
        def out(vec):
            if len(jnp.shape(vec)) <= 1:
                return lambda u: jnp.dot(jnp.diag(1 / ((1 / (ub - vec)) + (1 / (vec - lb)))), u)
            else:
                print("hey there")
                return lambda u: (1 / ((1 / (ub - vec)) + (1 / (vec - lb)))) * u

        return out

    return BregmanPotential(DP_bound_internal(lb, ub), DP_inv_bound_internal(lb, ub),
                            D2P_bound_internal(lb, ub), inv_D2P_bound_internal(lb, ub))


# usage: hessian_xy((min_P,max_P))(max_P)
def make_mixed_jvp(f, first_args, second_args, opposite=False):
    """Make a mixed jacobian-vector product function
    Args:
        f (callable): Binary callable with signature f(x,y)
        first_args (numpy.ndarray): First arguments to f
        second_args (numpy.ndarray): Second arguments to f
        opposite (bool, optional): Take Dyx if False, Dxy if True. Defaults to
            False.
    Returns:
        callable: Unary callable 'jvp(v)' taking a numpy.ndarray as input.
    """
    if opposite is not True:
        given = second_args
        gradfun = jit(jax.grad(f, 0))

        def frozen_grad(y):
            return gradfun(first_args, y)
    else:
        given = first_args
        gradfun = jit(jax.grad(f, 1))

        def frozen_grad(x):
            return gradfun(x, second_args)

    return jax.linearize(frozen_grad, given)[1]


def make_lagrangian(obj_func, breg_min=default_breg, breg_max=default_breg,
                    min_inequality_constraints=default_func,
                    min_equality_constraints=default_func, max_inequality_constraints=default_func,
                    max_equality_constraints=default_func):
    """Transform the original constrained minimax problem with parametric inequalities into another minimax problem with only set constraints

    Args:
        obj_func (callable): multivariate callable with signature `f(x,y, *args, **kwargs)`
        breg_min (Named tuples of callable): Tuple of unary callables with signature
                                            'BregmanPotential = collections.namedtuple("BregmanPotential", ["DP", "DP_inv", "D2P","D2P_inv"])'
                                            where DP and DP_inv are unary callables with signatures
                                            `DP(x,*args, **kwargs)`,'DP_inv(x,*arg,**kwarg)' and
                                            D2P, D2P_inv are function of functions
                                            (Given an x, returning linear transformation function
                                            that can take in another vector to output hessian-vector product).
        breg_max (Named tuples of callable): Tuple of unary callables
        min_inequality_constraints (callable): Unary callable with signature `h(x, *args, **kwargs)`
        min_equality_constraints (callable): Unary callable with signature `h(x, *args, **kwargs)`
        max_inequality_constraints (callable): Unary callable with signature `g(y, *args, **kwargs)`
        max_equality_constraints (callable): Unary callable with signature `g(y, *args, **kwargs)`

    Returns:
        tuple: callables (init_multipliers, lagrangian, breg_min_aug, breg_max_aug)
    """

    def init_multipliers(params_min, params_max=None, key=random.PRNGKey(1), *args, **kwargs):
        """Initialize multipliers for equality and inequality constraints for both players

        Args:
          params_min: initialized input to the equality and ineuqality constraints for min player, 'x'
          params_max: initialized input to the equality and ineuqality constraints for max player, 'y'

        Returns:
            min_augmented (tuple): initialized min player with signature (original_min_param, multipliers_eq_max, multipliers_ineq_max)
            max_augmented (tuple): initialized max player with signature (original_max_param, multipliers_eq_min, multipliers_ineq_min)
        """

        # Equality constraints' shape for both min and max player
        h_min = jax.eval_shape(min_equality_constraints, params_min, *args, **kwargs)
        h_max = jax.eval_shape(max_equality_constraints, params_max, *args, **kwargs)
        # Make multipliers for equality constraints associated with both players, ie, '\lambda_x' and '\lambda_y'
        multipliers_eq_min = tree_util.tree_map(lambda x: random.normal(key, x.shape), h_min)
        multipliers_eq_max = tree_util.tree_map(lambda x: random.normal(key, x.shape), h_max)

        # Inequality constraints' shape for both min and max player
        g_min = jax.eval_shape(min_inequality_constraints, params_min, *args,
                               **kwargs)  # should be a tuple
        g_max = jax.eval_shape(max_inequality_constraints, params_max, *args,
                               **kwargs)  # should be a tuple
        # Make multipliers for the constraints associated with both players, ie, '\mu_x' and '\mu_y'
        multipliers_ineq_min = tree_util.tree_map(lambda x: random.normal(key, x.shape), g_min)
        multipliers_ineq_max = tree_util.tree_map(lambda x: random.normal(key, x.shape), g_max)

        min_augmented = (params_min, multipliers_eq_max, multipliers_ineq_max)
        max_augmented = (params_max, multipliers_eq_min, multipliers_ineq_min)

        return min_augmented, max_augmented


    def lagrangian(minPlayer, maxPlayer):
        obj_portion = obj_func(*[x for x in [minPlayer[0], maxPlayer[0]] if x is not None])
        min_eq_portion = 0
        max_eq_portion = 0
        min_ineq_portion = 0
        max_ineq_portion = 0

        if maxPlayer[1] is None:
            pass
        else:
            min_eq_portion = math.pytree_dot(min_equality_constraints(minPlayer[0]), maxPlayer[1])

        if minPlayer[1] is None:
            pass
        else:
            max_eq_portion = math.pytree_dot(max_equality_constraints(maxPlayer[0]), minPlayer[1])

        if maxPlayer[2] is None:
            pass
        else:
            min_ineq_portion = math.pytree_dot(min_inequality_constraints(minPlayer[0]),
                                               maxPlayer[2])

        if minPlayer[2] is None:
            pass
        else:
            max_ineq_portion = math.pytree_dot(max_inequality_constraints(maxPlayer[0]),
                                               minPlayer[2])

        # out = obj_func(minPlayer[0], maxPlayer[0]) +\
        #      math.pytree_dot(min_equality_constraints(minPlayer[0]), maxPlayer[1]) +\
        #      math.pytree_dot(max_equality_constraints(maxPlayer[0]), minPlayer[1]) +\
        #      math.pytree_dot(min_inequality_constraints(minPlayer[0]), maxPlayer[2]) +\
        #      math.pytree_dot(max_inequality_constraints(maxPlayer[0]), minPlayer[2])
        return obj_portion + min_eq_portion + max_eq_portion + min_ineq_portion + max_ineq_portion

    # DP_eq_min = lambda v: jax.tree_map(lambda x: x, v)
    # DP_ineq_min = lambda v: jax.tree_map(DP_pd, v)
    min_augmented_DP = (breg_min.DP, lambda v: jax.tree_map(lambda x: x, v),
                        lambda v: jax.tree_map(DP_pd, v))  # [breg_min.DP, DP_eq_min, DP_ineq_min]

    # DP_inv_eq_min = lambda v: jax.tree_map(lambda x: x, v)
    # DP_inv_ineq_min = lambda v: jax.tree_map(DP_inv_pd,v)
    min_augmented_DP_inv = (breg_min.DP_inv, lambda v: jax.tree_map(lambda x: x, v),
                            lambda v: jax.tree_map(DP_inv_pd,
                                                   v))  # [breg_min.DP_inv, DP_inv_eq_min, DP_inv_ineq_min]

    # D2P_eq_min = D2P_l2
    # D2P_ineq_min = lambda v: jax.tree_map(D2P_pd,v)
    min_augmented_D2P = (breg_min.D2P, D2P_l2, lambda v: jax.tree_map(D2P_pd,
                                                                      v))  # [breg_min.D2P, D2P_eq_min, D2P_ineq_min]

    # inv_D2P_eq_min = D2P_l2
    # inv_D2P_ineq_min = lambda v: jax.tree_map(inv_D2P_pd, v)
    min_augmented_D2P_inv = (breg_min.inv_D2P, D2P_l2, lambda v: jax.tree_map(inv_D2P_pd,
                                                                              v))  # [breg_min.inv_D2P, inv_D2P_eq_min, inv_D2P_ineq_min]

    # DP_eq_max = lambda x: x
    # DP_ineq_max = lambda v: jax.tree_map(DP_pd, v)
    max_augmented_DP = (breg_max.DP, lambda v: jax.tree_map(lambda x: x, v),
                        lambda v: jax.tree_map(DP_pd, v))  # [breg_min.DP, DP_eq_min, DP_ineq_min]

    # DP_inv_eq_min = lambda v: jax.tree_map(lambda x: x, v)
    # DP_inv_ineq_min = lambda v: jax.tree_map(DP_inv_pd,v)
    max_augmented_DP_inv = (breg_max.DP_inv, lambda v: jax.tree_map(lambda x: x, v),
                            lambda v: jax.tree_map(DP_inv_pd,
                                                   v))  # [breg_min.DP_inv, DP_inv_eq_min, DP_inv_ineq_min]

    # D2P_eq_max = D2P_l2
    # D2P_ineq_max = lambda v: jax.tree_map(D2P_pd,v)
    max_augmented_D2P = (breg_max.D2P, D2P_l2, lambda v: jax.tree_map(D2P_pd,
                                                                      v))  # [breg_min.D2P, D2P_eq_min, D2P_ineq_min]

    # inv_D2P_eq_max =D2P_l2
    # inv_D2P_ineq_max = lambda v: jax.tree_map(inv_D2P_pd, v)
    max_augmented_D2P_inv = (breg_max.inv_D2P, D2P_l2, lambda v: jax.tree_map(inv_D2P_pd,
                                                                              v))  # [breg_max.inv_D2P, inv_D2P_eq_max, inv_D2P_ineq_max]

    return init_multipliers, lagrangian, BregmanPotential(min_augmented_DP, min_augmented_DP_inv,
                                                          min_augmented_D2P,
                                                          min_augmented_D2P_inv), BregmanPotential(
        max_augmented_DP, max_augmented_DP_inv, max_augmented_D2P, max_augmented_D2P_inv)


CMDState = collections.namedtuple("CMDState", "minPlayer maxPlayer minPlayer_dual maxPlayer_dual")
UpdateState = collections.namedtuple("UpdateState", "del_min del_max")
_tree_apply = partial(jax.tree_multimap, lambda f, x: f(x))


def updates(prev_state, eta_min, eta_max, hessian_xy=None, hessian_yx=None, grad_min=None,
            grad_max=None, breg_min=default_breg, breg_max=default_breg, objective_func=None):
    """Equation (4). Given current position (prev_state), compute the updates (del_x,del_y) to the players in cmd algorithm for next position.

    Args:
        prev_state (Named tuples of vectors): The current position of the players given by tuple
                                             with signature 'CMDState(minPlayer maxPlayer minPlayer_dual maxPlayer_dual)'
        breg_min (Named tuples of callable): Tuple of unary callables with signature
                                            'BregmanPotential = collections.namedtuple("BregmanPotential", ["DP", "DP_inv", "D2P","D2P_inv"])'
                                            where DP and DP_inv are unary callables with signatures
                                            `DP(x,*args, **kwargs)`, 'DP_inv(x,*arg,**kwarg)' and
                                            D2P, D2P_inv are function of functions
                                            (Given an x, returning linear transformation function
                                            that can take in another vector to output hessian-vector product).
        breg_max (Named tuples of callable): Tuple of unary callables as 'breg_min'.
        eta_min (scalar): User specified step size for min player. Default 1e-4.
        eta_max (scalar): User specified step size for max player. Default 1e-4.
        hessian_xy (callable): The (estimated) mixed hessian of the current positions of the players, represented in a matrix-vector operator from jax.jvp
        hessian_xy (callable): The (estimated) mixed hessian of the current positions of the players, represented in a matrix-vector operator from jax.jvp
        grad_min (vector): The (estimated) gradient of the cost function w.r.t. the max player parameters at current position.
        grad_max(vector): The (estimated) gradient of the cost function w.r.t. the max player parameters at current position.
    Returns:
        UpdateState(del_min, del_max), a named tuple for the updates
    """
    if objective_func is not None:
        # grad_min_func = jit(jacfwd(objective_func, 0))
        # grad_max_func = jit(jacfwd(objective_func, 1))
        # H_xy_func = jit(jacfwd(grad_min, 1))
        # H_yx_func =jit(jacfwd(grad_max, 0))

        # Compute current gradient for min and max players
        grad_min = jit(jacfwd(objective_func, 0))(prev_state.minPlayer, prev_state.maxPlayer)
        grad_max = jit(jacfwd(objective_func, 1))(prev_state.minPlayer, prev_state.maxPlayer)

        # Define the mixed hessian-vector product linear operator at current position
        def hessian_xy(tangent):
            return make_mixed_jvp(objective_func, prev_state.minPlayer, prev_state.maxPlayer)(
                tangent)

        def hessian_yx(tangent):
            return make_mixed_jvp(objective_func, prev_state.minPlayer, prev_state.maxPlayer, True)(
                tangent)

    def linear_opt_min(min_tree):
        temp = hessian_yx(min_tree)  # returns max_tree type
        temp1 = _tree_apply(_tree_apply(breg_max.inv_D2P, prev_state.maxPlayer),
                            temp)  # returns max_tree type
        temp2 = hessian_xy(temp1)  # returns min_tree type
        temp3 = tree_util.tree_map(lambda x: eta_max * x, temp2)  # still min_tree type
        temp4 = _tree_apply(_tree_apply(breg_min.D2P, prev_state.minPlayer),
                            min_tree)  # also returns min_tree type
        temp5 = tree_util.tree_map(lambda x: 1 / eta_min * x, temp4)
        print("linear operator being called! - min")
        return tree_util.tree_multimap(lambda x, y: x + y, temp3, temp5)  # min_tree type

    def linear_opt_max(max_tree):
        temp = hessian_xy(max_tree)
        temp1 = _tree_apply(_tree_apply(breg_min.inv_D2P, prev_state.minPlayer), temp)
        temp2 = hessian_yx(temp1)  # returns max_tree type
        temp3 = tree_util.tree_map(lambda x: eta_min * x, temp2)  # max_tree type
        temp4 = _tree_apply(_tree_apply(breg_max.D2P, prev_state.maxPlayer), max_tree)
        temp5 = tree_util.tree_map(lambda x: 1 / eta_max * x, temp4)  # max_tree type
        print("linear operator being called! - max")
        return tree_util.tree_multimap(lambda x, y: x + y, temp3, temp5)  # max_tree type

    # calculate the vectors in equation (4)
    temp = hessian_xy(_tree_apply(_tree_apply(breg_max.inv_D2P, prev_state.maxPlayer), grad_max))
    temp2 = tree_util.tree_map(lambda x: eta_max * x, temp)
    vec_min = tree_util.tree_multimap(lambda arr1, arr2: arr1 + arr2, grad_min, temp2)

    # temp = _tree_apply(hessian_yx, _tree_apply(_tree_apply(breg_min.inv_D2P, prev_state.minPlayer), grad_min))
    temp = hessian_yx(_tree_apply(_tree_apply(breg_min.inv_D2P, prev_state.minPlayer), grad_min))
    temp2 = tree_util.tree_map(lambda x: eta_min * x, temp)
    vec_max = tree_util.tree_multimap(lambda x, y: x - y, grad_max, temp2)

    update_min, status_min = linalg.cg(linear_opt_min, vec_min, maxiter=1000, tol=1e-6)
    update_min = tree_util.tree_multimap(lambda x: -x, update_min)
    update_max, status_max = linalg.cg(linear_opt_max, vec_max, maxiter=1000, tol=1e-6)

    return UpdateState(update_min, update_max)


def cmd_step(prev_state, updates, breg_min=default_breg, breg_max=default_breg):
    """Equation (2). Take in the previous player positions and update to the next player position. Return a 1-step cmd update.

    Args:
        prev_state (Named tuples of vectors): The current position of the players given by tuple
                                             with signature 'CMDState(minPlayer maxPlayer minPlayer_dual maxPlayer_dual)'
        updates (Named tuples of vectors): The updates del_x,del_y computed from updates(...) with signature 'UpdateState(del_min, del_max)'
        breg_min (Named tuples of callable): Tuple of unary callables with signature
                                            'BregmanPotential = collections.namedtuple("BregmanPotential", ["DP", "DP_inv", "D2P","D2P_inv"])'
                                            where DP and DP_inv are unary callables with signatures
                                            `DP(x,*args, **kwargs)`,'DP_inv(x,*arg,**kwarg)' and
                                            D2P, D2P_inv are function of functions
                                            (Given an x, returning linear transformation function
                                            that can take in another vector to output hessian-vector product).
        breg_max (Named tuples of callable): Tuple of unary callables as 'breg_min'.

    Returns:
        Named tuple: the states of the players at current iteration - CMDState
    """
    temp_min = _tree_apply(_tree_apply(breg_min.D2P, prev_state.minPlayer), updates.del_min)
    temp_max = _tree_apply(_tree_apply(breg_max.D2P, prev_state.maxPlayer), updates.del_max)

    dual_min = tree_util.tree_multimap(lambda x, y: x + y, prev_state.minPlayer_dual, temp_min)
    dual_max = tree_util.tree_multimap(lambda x, y: x + y, prev_state.maxPlayer_dual, temp_max)

    minP = _tree_apply(breg_min.DP_inv, dual_min)
    maxP = _tree_apply(breg_max.DP_inv, dual_max)

    return CMDState(minP, maxP, dual_min, dual_max)
