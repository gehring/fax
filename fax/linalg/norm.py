import jax
import jax.numpy as jnp
import jax.tree_util


def tree_vdot(x, y):
    return sum(jax.tree_leaves(jax.tree_multimap(jnp.vdot, x, y)))


def tree_l2_squared(x):
    """Compute the squared l_2 vector norm of a pytree's leaves.

    If a leaf is a multidimensional array, it is treated as a flat vector.

    Args:
        x: a pytree of arrays

    Returns:
        the squared l_2 norm of the flat vector corresponding to concatenating
        the flattened arrays in `x`'s leaves.
    """
    return tree_vdot(x, x)


def tree_l2_norm(x):
    """Compute the l_2 vector norm of a pytree's leaves.

    This function is equivalent to: `sqrt(tree_l2_squared(x))`

     Args:
        x: a pytree of arrays

    Returns:
        the l_2 norm of the flat vector corresponding to concatenating the
        flattened arrays in `x`'s leaves.
    """
    return jnp.sqrt(tree_l2_squared(x))
