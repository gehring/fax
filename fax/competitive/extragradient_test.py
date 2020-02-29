import hypothesis.extra.numpy
import jax.numpy as np
import jax.test_util
import numpy as onp
from absl.testing import absltest
from jax import random
from jax.config import config

from fax import converge
from fax.competitive import extragradient

config.update("jax_enable_x64", True)


class CGATest(jax.test_util.JaxTestCase):
    @hypothesis.settings(max_examples=10, deadline=5000.)
    @hypothesis.given(
        hypothesis.extra.numpy.arrays(
            onp.float, (2, 3), elements=hypothesis.strategies.floats(0.1, 1)),
    )
    def testEgSimpleTwoPlayer(self, amat):
        amat = amat + np.eye(*amat.shape)

        def f(x, y):
            return y.T @ amat @ x + np.dot(x, x)

        rng = random.PRNGKey(0)  # pyrandom.randint(0, 2 ** 32 - 1))
        rng_x, rng_y = random.split(rng)
        init_vals = (random.uniform(rng_x, shape=(amat.shape[1],)),
                     random.uniform(rng_y, shape=(amat.shape[0],)))

        step_size = 1e-1
        rtol = atol = 1e-12
        max_iter = 5000

        def convergence_test(x_new, x_old):
            return converge.max_diff_test(x_new, x_old, rtol, atol)

        solution = extragradient.extra_gradient_iteration(
            init_vals, step_size, step_size, f, convergence_test, max_iter)

        self.assertAllClose(
            solution.value[0],
            np.zeros_like(solution.value[0]),
            rtol=1e-8, atol=1e-8, check_dtypes=True)

    @hypothesis.settings(max_examples=10, deadline=5000.)
    @hypothesis.given(
        hypothesis.extra.numpy.arrays(
            onp.float, (2, 3), elements=hypothesis.strategies.floats(0.1, 1)),
    )
    def testBatchedIteration(self, amat):
        amat = amat + np.eye(*amat.shape)

        def f(x, y):
            return y.T @ amat @ x + np.dot(x, x)

        step_size = 1e-1
        rtol = atol = 1e-12
        max_iter = 5000
        rng = random.PRNGKey(0)
        rng_x, rng_y = random.split(rng)
        init_vals = (random.uniform(rng_x, shape=(amat.shape[1],)),
                     random.uniform(rng_y, shape=(amat.shape[0],)))

        def convergence_test(x_new, x_old):
            return converge.max_diff_test(x_new, x_old, rtol, atol)

        solution = extragradient.extra_gradient_iteration(
            init_vals, step_size, step_size, f, convergence_test, max_iter
            , batched_iter_size=10)

        self.assertAllClose(
            solution.value[0],
            np.zeros_like(solution.value[0]),
            rtol=1e-8, atol=1e-8, check_dtypes=True)

    @hypothesis.settings(max_examples=10, deadline=5000.)
    @hypothesis.given(
        hypothesis.extra.numpy.arrays(
            onp.float, (2, 3), elements=hypothesis.strategies.floats(0.1, 1)),
    )
    def testUnroll(self, amat):
        amat = amat + np.eye(*amat.shape)

        def f(x, y):
            return y.T @ amat @ x + np.dot(x, x)

        step_size = 1e-1
        rtol = atol = 1e-12
        max_iter = 5000
        rng = random.PRNGKey(0)
        rng_x, rng_y = random.split(rng)
        init_vals = (random.uniform(rng_x, shape=(amat.shape[1],)),
                     random.uniform(rng_y, shape=(amat.shape[0],)))

        def convergence_test(x_new, x_old):
            return converge.max_diff_test(x_new, x_old, rtol, atol)

        solution = extragradient.extra_gradient_iteration(
            init_vals, step_size, step_size, f, convergence_test, max_iter
            , unroll=True)

        self.assertAllClose(
            solution.value[0],
            np.zeros_like(solution.value[0]),
            rtol=1e-8, atol=1e-8, check_dtypes=True)

if __name__ == "__main__":
    absltest.main()
