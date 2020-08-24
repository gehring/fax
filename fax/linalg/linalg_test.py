import hypothesis.extra.numpy

import pytest

import numpy as np

from fax import linalg

import jax
import jax.test_util
import jax.numpy as jnp

from jax.config import config
config.update("jax_enable_x64", True)


@pytest.fixture(scope="module", params=[linalg.gmres])
def dense_solver(request):
    linear_solver = request.param

    @jax.jit
    def _solve(A, b):
        return linear_solver(lambda x: jnp.dot(A, x), b, tol=1e-6)[0]

    return _solve


@hypothesis.settings(deadline=None)
@hypothesis.given(
    hypothesis.extra.numpy.arrays(
        np.float_, (20, 20), elements=hypothesis.strategies.floats(0, 1),
        unique=True),
    hypothesis.extra.numpy.arrays(
        np.float_, 20, elements=hypothesis.strategies.floats(1e-2, 1)))
def test_linear_solve(dense_solver, A, b):
    hypothesis.assume(np.linalg.det(A) > 1e-3)
    jax.test_util.check_close(dense_solver(A, b), np.linalg.solve(A, b),
                              rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("solver", [linalg.gmres])
def test_pytree_input(solver):
    factor = 2.0

    def linear_op(xs):
        return jax.tree_map(lambda x: x*factor, xs)

    bvec = (0.3 * np.ones((2, 3)), [1., 2., {"foo": 4.}])
    bvec = jax.tree_map(lambda x: jnp.array(x), bvec)

    xsol = solver(linear_op, bvec)[0]
    jax.tree_multimap(
        lambda b, x: np.testing.assert_allclose(b/factor, x, rtol=1e-5),
        bvec,
        xsol,
    )
