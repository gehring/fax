import functools

from jax import tree_util, lax, numpy as np

division = functools.partial(tree_util.tree_multimap, lax.div)
add = functools.partial(tree_util.tree_multimap, lax.add)
sub = functools.partial(tree_util.tree_multimap, lax.sub)
mul = functools.partial(tree_util.tree_multimap, lax.mul)
square = functools.partial(tree_util.tree_map, lax.square)


def division_constant(constant):
    def divide(a):
        return tree_util.tree_multimap(lambda _a: _a / constant, a)

    return divide


def multiply_constant(constant):
    return functools.partial(mul, constant)


def expand_like(a, b):
    return a * np.ones(b.shape, b.dtype)


def make_exp_smoothing(beta):
    def exp_smoothing(state, var):
        return multiply_constant(beta)(state) + multiply_constant((1 - beta))(var)

    return exp_smoothing
