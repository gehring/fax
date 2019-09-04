from jax import lax
from jax import numpy as np
from jax import tree_util


def pytree_dot(x, y):
    partial_dot = tree_util.tree_multimap(np.dot, x, y)
    return tree_util.tree_reduce(lax.add, partial_dot)
