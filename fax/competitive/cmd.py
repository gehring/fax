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





CMDState = collections.namedtuple("CMDState", "iter minP maxP del_x del_y")

def compose_matrix_func(A,fun):
    f = lambda x: jnp.dot(A,x)
    g = lambda x: jnp.dot(jnp.transpose(A),x)
    return lambda x: f(A(g(x)))


def updates(init_state,breg_min,breg_max, eta_min, eta_max, mixed_hessian, J_min, J_max):
    linear_opt_min = 1/eta_min * breg_min.D2P(init_state) - eta_max * compose_matrix_func(mixed_hessian,breg_max.D2P_inv(init_state))
    linear_opt_max = 1/eta_max * breg_max.D2P(init_state) - eta_min * compose_matrix_func(jnp.transpose(mixed_hessian),breg_min.D2P_inv(init_state))

    vec_min = J_min + eta_max * jnp.dot(mixed_hessian, breg_max.D2P_inv(J_max))
    vec_max = J_max - eta_min * jnp.dot(jnp.transpose(mixed_hessian), breg_min.D2P_inv(J_min))

    update_min,status_min = linalg.cg(linear_opt_min,vec_min,max_iter=1000,tol=1e-8)
    update_max,status_max =  linalg.cg(linear_opt_max,vec_max,max_iter=1000,tol=1e-8)
    return update_min, update_max





def cmd_step(init_state, lagrangian,breg_min,breg_max, eta_min = 1e-4, eta_max = 1e-4):
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



    minP_prev = init_state.minP
    maxP_prev = init_state.maxP

    return CMDState(iter_num, minP,maxP,del_min,del_max)