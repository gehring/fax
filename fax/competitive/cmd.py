import collections
from functools import partial
import operator

import jax
import jax.numpy as jnp


def make_lagrangian(obj_func, breg_min, breg_max, min_inequality_constraints,
                    min_equality_constraints, max_inequality_constraints, max_equality_constraints):
    """Transform the original constrained minimax problem with parametric inequalities into another minimax problem with only set constraints

    Args:
        obj_func (callable): multivariate callable with signature `f(x,y, *args, **kwargs)`
        breg_min (callable): Unary callable with signature `Px(x,*args, **kwargs)`
        breg_max (callable): Unary callable with signature `Py(y,*args, **kwargs)`
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
