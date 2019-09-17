from jax import tree_util
import jax.numpy as np

from fax import math


def make_lagrangian(func, equality_constraints):

    def init_multipliers(params):
        h = equality_constraints(params)
        multipliers = tree_util.tree_map(np.zeros_like, h)
        return params, multipliers

    def lagrangian(params, multipliers, *args, **kwargs):
        h = equality_constraints(params, *args, **kwargs)
        return -func(params, *args, **kwargs) + math.pytree_dot(multipliers, h)

    def get_params(opt_state):
        return opt_state[0]

    return init_multipliers, lagrangian, get_params


def get_multipliers(opt_state):
    return opt_state[1]
