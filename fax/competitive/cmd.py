import collections
from functools import partial
import operator

import jax
import jax.numpy as jnp
from jax import lax
from jax import tree_util
from jax.experimental import optimizers
from jax.scipy.sparse import linalg

def make_lagrangian(obj_func, breg_min, breg_max, min_inequality_constraints,
                    min_equality_constraints, max_inequality_constraints, max_equality_constraints):
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
        tuple: callables (lagrangian, breg_min_lagrangian, breg_max_lagrangian, init_multipliers, get_params)
    """

    def init_multipliers(params_min, params_max, *args, **kwargs):
        """Initialize multipliers for euqality and ineuqality constraints for both players

        Args:
          params_min: input to the equality and ineuqality constraints for min player, 'x'
          params_min: input to the equality and ineuqality constraints for max player, 'y'
        """
        # Equality constraints' shape for both min and max player
        h_min = jax.eval_shape(min_equality_constraints, params_min, *args, **kwargs)
        h_max = jax.eval_shape(max_equality_constraints, params_max, *args, **kwargs)
        # Make multipliers for equality constraints associated with both players, ie, '\lambda_x' and '\lambda_y'
        multipliers_eq_min = tree_util.tree_map(lambda x: np.zeros(x.shape, x.dtype), h_min)
        multipliers_eq_max = tree_util.tree_map(lambda x: np.zeros(x.shape, x.dtype), h_max)

        # Inequality constraints' shape for both min and max player
        g_min = jax.eval_shape(min_inequality_constraints, params_min, *args, **kwargs)
        g_max = jax.eval_shape(max_inequality_constraints, params_max, *args, **kwargs)
        # Make multipliers for the constraints associated with both players, ie, '\mu_x' and '\mu_y'
        multipliers_ineq_min = tree_util.tree_map(lambda x: np.zeros(x.shape, x.dtype), g_min)
        multipliers_ineq_max = tree_util.tree_map(lambda x: np.zeros(x.shape, x.dtype), g_max)

        return params_min, params_max, multipliers_eq_min, multipliers_eq_max, multipliers_ineq_min, multipliers_ineq_max

    def lagrangian(params_min, params_max, multipliers_eq_min, multipliers_eq_max,
                   multipliers_ineq_min, multipliers_ineq_max, *args, **kwargs):
        """Generate the Lagrangian of the original optimization

        Args:
            params_min: input to the equality and ineuqality constraints for min player, 'x'
            params_min: input to the equality and ineuqality constraints for max player, 'y'
        """
        h_min = min_equality_constraints(params_min, *args, **kwargs)
        h_max = max_equality_constraints(params_max, *args, **kwargs)
        g_min = min_inequality_constraints(params_min, *args, **kwargs)
        g_max = max_inequality_constraints(params_max, *args, **kwargs)

        return func(params, *args, **kwargs) + math.pytree_dot(multipliers_eq_min,
                                                               h_min) + math.pytree_dot(
            multipliers_eq_max, h_max) + math.pytree_dot(multipliers_ineq_min,
                                                         g_min) + math.pytree_dot(
            multipliers_ineq_max, g_max)

    def breg_min_lagrangian(params_min, multipliers_eq_max, multipliers_ineq_max, *args, **kwargs):
        sign, logdet = jnp.linalg.slogdet(multipliers_ineq_max)
        return breg_min(params_min, *args, **kwargs) + 0.5 * jnp.norm(
            multipliers_eq_max) ** 2 + jnp.dot(multipliers_ineq_max, jnp.log(multipliers_ineq_max))

    def breg_max_lagrangian(params_max, multipliers_eq_min, multipliers_ineq_min, *args, **kwargs):
        return breg_max(params_max, *args, **kwargs) + 0.5 * jnp.norm(
            multipliers_eq_min) ** 2 + jnp.dot(multipliers_ineq_min, jnp.log(multipliers_ineq_min))

    def get_params(opt_state):
        return opt_state[0]

    return lagrangian, breg_min_lagrangian, breg_max_lagrangian, init_multipliers, get_params





CMDState = collections.namedtuple("CMDState", "minPlayer maxPlayer minPlayer_dual maxPlayer_dual")
UpdateState = collections.namedtuple("UpdateState","del_min del_max")





def updates(prev_state,breg_min,breg_max, eta_min, eta_max, hessian_xy, hessian_yx, J_min, J_max):
    """Given current position (prev_state), compute the updates (del_x,del_y) to the players in CMD algorithm for next position.

    Args:
        prev_state (Named tuples of vectors): The current position of the players given by tuple
                                             with signature 'CMDState(minPlayer maxPlayer minPlayer_dual maxPlayer_dual)'
        breg_min (Named tuples of callable): Tuple of unary callables with signature
                                            'BregmanPotential = collections.namedtuple("BregmanPotential", ["DP", "DP_inv", "D2P","D2P_inv"])'
                                            where DP and DP_inv are unary callables with signatures
                                            `DP(x,*args, **kwargs)`,'DP_inv(x,*arg,**kwarg)' and
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

    def temp_min(eta_min):
        return lambda x: 1/eta_min * breg_min.D2P(prev_state)(x)
    def temp_max(eta_max):
        return lambda x: 1/eta_max * breg_max.D2P(prev_state)(x)

    linear_opt_min = lambda v: temp_min(eta_min)(v) + eta_max * hessian_xy(breg_max.D2P_inv(prev_state)(hessian_yx(v)))
    linear_opt_max = lambda v: temp_max(eta_max)(v) + eta_min * hessian_yx(breg_min.D2P_inv(prev_state)(hessian_xy(v)))

    vec_min = J_min + eta_max * hessian_xy(breg_max.D2P_inv(J_max))
    vec_max = J_max - eta_min * hessian_yx(breg_min.D2P_inv(J_min))

    update_min,status_min = linalg.cg(linear_opt_min,vec_min,max_iter=1000,tol=1e-8)
    update_max,status_max =  linalg.cg(linear_opt_max,vec_max,max_iter=1000,tol=1e-8)
    return UpdateState(update_min, update_max)


def cmd_step(prev_state,updates, breg_min,breg_max):
    """Take in the previous player positions and update to the next player position. Return a 1-step CMD update.

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
    dual_min = breg_min.DP(prev_state.minP) - updates.del_min
    dual_max = breg_max.DP(prev_state.maxP) + updates.del_max

    minP = breg_min.DP_inv(dual_min)
    maxP = breg_max.DP_inv(dual_max)

    return CMDState(minP, maxP,dual_min,dual_max)

    """Take in an objective function that is likely the Lagrangian of an optimization problem with
        Bregman potential on both min and max player and return a 1-step CMD update.

    Args:
        lagrangian (callable): multivariate callable with signature `L(params_min, params_max, multipliers_eq_min, multipliers_eq_max,
                   multipliers_ineq_min, multipliers_ineq_max, *args, **kwargs)`
        breg_min (Named tuples of callable): Tuple of unary callables with signature
                                            'BregmanPotential = collections.namedtuple("BregmanPotential", ["DP", "DP_inv", "D2P","D2P_inv"])'
                                            where DP and DP_inv are unary callables with signatures
                                            `DP(x,*args, **kwargs)`,'DP_inv(x,*arg,**kwarg)' and
                                            D2P, D2P_inv are function of functions
                                            (Given an x, returning linear transformation function
                                            that can take in another vector to output hessian-vector product).
        breg_max (Named tuples of callable): Tuple of unary callables
        eta_min (scalar): User specified step size for min player. Default 1e-4.
        eta_max (scalar): User specified step size for max player. Default 1e-4.

    Returns:
        Named tuple: the states of the players at current iteration - CMDState
    """