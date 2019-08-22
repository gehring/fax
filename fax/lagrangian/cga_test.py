from absl.testing import absltest

import hypothesis.extra.numpy

import numpy as onp
from numpy import testing

from fax import converge
from fax.lagrangian import cga

import jax
from jax import random
import jax.numpy as np
import jax.test_util
from jax.config import config
config.update("jax_enable_x64", True)


class CGATest(jax.test_util.JaxTestCase):

    @hypothesis.settings(max_examples=100, deadline=5000.)
    @hypothesis.given(
        hypothesis.extra.numpy.arrays(
            onp.float, (2, 3), elements=hypothesis.strategies.floats(0.1, 1)),
    )
    def testSimpleTwoPlayer(self, amat):
        def f(x, y):
            return x.T @ amat @ y

        def g(x, y):
            return -f(x, y)

        eta = 0.5
        rtol = atol = 1e-10
        max_iter = 10000

        def convergence_test(x_new, x_old):
            return converge.max_diff_test(x_new, x_old, rtol, atol)

        rng = random.PRNGKey(42)
        rng_x, rng_y = random.split(rng)
        init_vals = (random.uniform(rng_x, shape=(2,)),
                     random.uniform(rng_y, shape=(3,)))

        solution = cga.cga_iteration(init_vals, eta, f, g, convergence_test,
                                     max_iter)
        testing.assert_allclose(jax.tree_map(onp.zeros_like, solution.value),
                                solution.value)


if __name__ == "__main__":
    absltest.main()
