import collections
from fax import math
import jax
import jax.numpy as jnp
from jax import tree_util
from jax.scipy.sparse import linalg
from cmd_helper import DP_pd, DP_inv_pd, inv_D2P_pd, D2P_pd, id_func
from functools import partial

BregmanPotential = collections.namedtuple("BregmanPotential", ["DP", "DP_inv", "D2P", "inv_D2P"])
# AugmentedDP = collections.namedtuple("AugmentedDP", ["DP_primal", "DP_eq", "DP_ineq"])
# AugmentedDPinv = collections.namedtuple("AugmentedDPinv", ["DPinv_primal","DPinv_eq","DPinv_ineq"])
# AugmentedD2P = collections.namedtuple("AugmentedD2P", ["D2P_primal", "D2P_eq", "D2P_ineq"])
# AugmentedD2Pinv = collections.namedtuple("AugmentedD2Pinv", ["D2Pinv_primal","D2Pinv_eq","D2Pinv_ineq"])


def default_constraint(x=None):
    return 0


def make_lagrangian(obj_func, breg_min, breg_max, min_inequality_constraints=default_constraint,
                    min_equality_constraints=default_constraint, max_inequality_constraints=default_constraint, max_equality_constraints=default_constraint):
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
        tuple: callables (lagrangian, breg_min_aug, breg_max_aug, init_multipliers)
    """

    def init_multipliers(params_min, params_max, *args, **kwargs):
        """Initialize multipliers for equality and inequality constraints for both players

        Args:
          params_min: input to the equality and ineuqality constraints for min player, 'x'
          params_max: input to the equality and ineuqality constraints for max player, 'y'

        Returns:
            params_min (pytree): input to the equality and ineuqality constraints for min player, 'x'
            params_max (pytree): input to the equality and ineuqality constraints for max player, 'y'
            multipliers_eq_min (pytree): The initialized Lagrangian multiplier for the equality constraint for min player
            multipliers_eq_max (pytree): The initialized Lagrangian multiplier for the equality constraint for max player
            multipliers_ineq_min (pytree): The initialized Lagrangian multiplier for the inequality constraint for min player
            multipliers_ineq_max (pytree): The initialized Lagrangian multiplier for the inequality constraint for max player
        """
        # Equality constraints' shape for both min and max player
        h_min = jax.eval_shape(min_equality_constraints, params_min, *args, **kwargs)
        h_max = jax.eval_shape(max_equality_constraints, params_max, *args, **kwargs)
        # Make multipliers for equality constraints associated with both players, ie, '\lambda_x' and '\lambda_y'
        multipliers_eq_min = tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), h_min)
        multipliers_eq_max = tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), h_max)

        # Inequality constraints' shape for both min and max player
        g_min = jax.eval_shape(min_inequality_constraints, params_min, *args, **kwargs)  # should be a tuple
        g_max = jax.eval_shape(max_inequality_constraints, params_max, *args, **kwargs)  # should be a tuple
        # Make multipliers for the constraints associated with both players, ie, '\mu_x' and '\mu_y'
        multipliers_ineq_min = tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), g_min)
        multipliers_ineq_max = tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), g_max)

        return params_min, params_max, multipliers_eq_min, multipliers_eq_max, multipliers_ineq_min, multipliers_ineq_max

    def lagrangian(params_min, params_max, multipliers_eq_min=None, multipliers_eq_max=None,
                   multipliers_ineq_min=None, multipliers_ineq_max=None, *args, **kwargs):
        """Generate the Lagrangian of the original optimization

        Args:
            params_min: input to the equality and inequality constraints for min player, 'x'
            params_max: input to the equality and inequality constraints for max player, 'y'
            multipliers_eq_min: The Lagrangian multiplier for the equality constraint for min player
            multipliers_eq_max: The Lagrangian multiplier for the equality constraint for max player
            multipliers_ineq_min: The Lagrangian multiplier for the inequality constraint for min player
            multipliers_ineq_max: The Lagrangian multiplier for the inequality constraint for max player
        Returns:
            The Lagrangian function (callable) of the original problem with some currently assigned Lagrangian multipliers.
        """
        h_min = min_equality_constraints(params_min, *args, **kwargs)  # This is a pytree
        h_max = max_equality_constraints(params_max, *args, **kwargs)  # This returns a pytree
        g_min = min_inequality_constraints(params_min, *args, **kwargs)  # This returns a pytree
        g_max = max_inequality_constraints(params_max, *args, **kwargs)  # This returns a pytree

        return obj_func(params_min, params_max, *args, **kwargs) + math.pytree_dot(multipliers_eq_min, h_min) + math.pytree_dot(multipliers_eq_max, h_max) + math.pytree_dot(multipliers_ineq_min, g_min) + math.pytree_dot(multipliers_ineq_max, g_max)

    def breg_min_aug():
        """Augment the original min player's Bregman divergence structure with Lagrangian multipliers from the max player's (in)equalities.

        Returns:
            Named tuple of augmented Bregman divergence containing functions: pytree -> pytree.
        """
        # DP_eq_min = lambda x: x
        # DP_ineq_min = lambda v: jax.tree_map(DP_pd, v)
        min_augmented_DP = [breg_min.DP, lambda x: x, lambda v: jax.tree_map(DP_pd, v)]  # [breg_min.DP, DP_eq_min, DP_ineq_min]

        # DP_inv_eq_min = lambda v: jax.tree_map(lambda x: x, v)
        # DP_inv_ineq_min = lambda v: jax.tree_map(DP_inv_pd,v)
        min_augmented_DP_inv = [breg_min.DP_inv, lambda v: jax.tree_map(lambda x: x, v), lambda v: jax.tree_map(DP_inv_pd, v)]  # [breg_min.DP_inv, DP_inv_eq_min, DP_inv_ineq_min]

        # D2P_eq_min = lambda v: jax.tree_map(id_func, v)
        # D2P_ineq_min = lambda v: jax.tree_map(D2P_pd,v)
        min_augmented_D2P = [breg_min.D2P, lambda v: jax.tree_map(id_func, v), lambda v: jax.tree_map(D2P_pd, v)]  # [breg_min.D2P, D2P_eq_min, D2P_ineq_min]

        # inv_D2P_eq_min = lambda v: jax.tree_map(lambda x:x, v)
        # inv_D2P_ineq_min = lambda v: jax.tree_map(inv_D2P_pd, v)
        min_augmented_D2P_inv = [breg_min.inv_D2P, lambda v: jax.tree_map(lambda x:x, v), lambda v: jax.tree_map(inv_D2P_pd, v)]  # [breg_min.inv_D2P, inv_D2P_eq_min, inv_D2P_ineq_min]

        return BregmanPotential(min_augmented_DP, min_augmented_DP_inv, min_augmented_D2P, min_augmented_D2P_inv)

    def breg_max_aug():
        # DP_eq_max = lambda x: x
        # DP_ineq_max = lambda v: jax.tree_map(DP_pd, v)
        max_augmented_DP = [breg_max.DP, lambda x: x, lambda v: jax.tree_map(DP_pd, v)]  # [breg_min.DP, DP_eq_min, DP_ineq_min]

        # DP_inv_eq_min = lambda v: jax.tree_map(lambda x: x, v)
        # DP_inv_ineq_min = lambda v: jax.tree_map(DP_inv_pd,v)
        max_augmented_DP_inv = [breg_max.DP_inv, lambda v: jax.tree_map(lambda x: x, v), lambda v: jax.tree_map(DP_inv_pd, v)]  # [breg_min.DP_inv, DP_inv_eq_min, DP_inv_ineq_min]

        # D2P_eq_max = lambda v: jax.tree_map(id_func, v)
        # D2P_ineq_max = lambda v: jax.tree_map(D2P_pd,v)
        max_augmented_D2P = [breg_max.D2P, lambda v: jax.tree_map(id_func, v), lambda v: jax.tree_map(D2P_pd, v)]  # [breg_min.D2P, D2P_eq_min, D2P_ineq_min]

        # inv_D2P_eq_max = lambda v: jax.tree_map(lambda x:x, v)
        # inv_D2P_ineq_max = lambda v: jax.tree_map(inv_D2P_pd, v)
        max_augmented_D2P_inv = [breg_max.inv_D2P, lambda v: jax.tree_map(lambda x:x, v), lambda v: jax.tree_map(inv_D2P_pd, v)]  # [breg_max.inv_D2P, inv_D2P_eq_max, inv_D2P_ineq_max]

        return BregmanPotential(max_augmented_DP, max_augmented_DP_inv, max_augmented_D2P, max_augmented_D2P_inv)

    return lagrangian, breg_min_aug, breg_max_aug, init_multipliers


CMDState = collections.namedtuple("CMDState", "minPlayer maxPlayer minPlayer_dual maxPlayer_dual")
UpdateState = collections.namedtuple("UpdateState", "del_min del_max")
_tree_apply = partial(jax.tree_multimap, lambda f, x: f(x))


def updates(prev_state, breg_min, breg_max, eta_min, eta_max, hessian_xy, hessian_yx, J_min, J_max):
    """Equation (4). Given current position (prev_state), compute the updates (del_x,del_y) to the players in CMD algorithm for next position.

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
        J_min (vector): The (estimated) gradient of the cost function w.r.t. the max player parameters at current position.
        J_max(vector): The (estimated) gradient of the cost function w.r.t. the max player parameters at current position.
    Returns:
        UpdateState(del_min, del_max), a named tuple for the updates
    """

    def linear_opt_min(v):
        temp = _tree_apply(hessian_yx, v)
        temp1 = _tree_apply(breg_max.inv_D2P, temp)
        temp2 = _tree_apply(hessian_xy, temp1)
        temp3 = tree_util.tree_map(lambda x: 1 / eta_max * x, temp2)
        temp4 = _tree_apply(breg_min.D2P, v)
        temp5 = tree_util.tree_map(lambda x: 1 / eta_min * x, temp4)
        return tree_util.tree_map(lambda x, y: x + y, temp3, temp5)

    def linear_opt_max(v):
        temp = _tree_apply(hessian_xy, v)
        temp1 = _tree_apply(breg_min.inv_D2P, temp)
        temp2 = _tree_apply(hessian_yx, temp1)
        temp3 = tree_util.tree_map(lambda x: 1 / eta_min * x, temp2)
        temp4 = _tree_apply(breg_max.D2P, v)
        temp5 = tree_util.tree_map(lambda x: 1 / eta_max * x, temp4)
        return tree_util.tree_map(lambda x, y: x + y, temp3, temp5)

    # calculate the vectors in equation (4)
    temp = _tree_apply(hessian_xy, _tree_apply(breg_max.inv_D2P, J_max))
    temp2 = tree_util.tree_map(lambda x: eta_max * x, temp)
    vec_min = tree_util.tree_map(lambda x, y: x + y, J_min, temp2)

    temp = _tree_apply(hessian_yx, _tree_apply(breg_min.inv_D2P, J_min))
    temp2 = tree_util.tree_map(lambda x: eta_min * x, temp)
    vec_max = tree_util.tree_map(lambda x, y: x + y, J_max, temp2)

    update_min, status_min = linalg.cg(linear_opt_min, vec_min, max_iter=1000, tol=1e-6)
    update_max, status_max = linalg.cg(linear_opt_max, vec_max, max_iter=1000, tol=1e-6)
    return UpdateState(update_min, update_max)


def cmd_step(prev_state, updates, breg_min, breg_max):
    """Equation (2). Take in the previous player positions and update to the next player position. Return a 1-step CMD update.

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
    temp_min = _tree_apply(breg_min.D2P, updates.del_min)
    temp_max = _tree_apply(breg_max.D2P, updates.del_max)

    dual_min = tree_util.tree_map(lambda x, y: x + y, prev_state.minPlayer_dual, temp_min)
    dual_max = tree_util.tree_map(lambda x, y: x + y, prev_state.maxPlayer_dual, temp_max)

    minP = _tree_apply(breg_min.DP_inv, dual_min)
    maxP = _tree_apply(breg_max.DP_inv, dual_max)

    return CMDState(minP, maxP, dual_min, dual_max)
