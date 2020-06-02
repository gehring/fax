from jax import tree_util, lax, numpy as np


def division_constant(constant):
    def divide(a):
        return tree_util.tree_multimap(lambda _a: _a / constant, a)

    return divide


def multiply_constant(constant):
    def multiply(a):
        return tree_util.tree_multimap(lambda _a: _a * constant, a)

    return multiply


division = lambda _a, _b: tree_util.tree_multimap(lambda _a, _b: _a / _b, _a, _b)
add = lambda _a, _b: tree_util.tree_multimap(lambda _a, _b: _a + _b, _a, _b)
sub = lambda _a, _b: tree_util.tree_multimap(lambda _a, _b: _a - _b, _a, _b)


def mul(_a, _b):
    return tree_util.tree_multimap(lax.mul, _a, _b)


def expand_like(a, b):
    return a * np.ones(b.shape, b.dtype)


square = lambda _a: tree_util.tree_map(np.square, _a)
