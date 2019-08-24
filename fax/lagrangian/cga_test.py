from absl.testing import absltest
from absl.testing import parameterized

import hypothesis.extra.numpy

import numpy as onp

from fax import converge
from fax.lagrangian import cga

import jax
import jax.numpy as np
from jax import random
import jax.test_util

from jax.config import config
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)


class CGATest(jax.test_util.JaxTestCase):

    # @hypothesis.settings(max_examples=100, deadline=5000.)
    # @hypothesis.given(
    #     hypothesis.extra.numpy.arrays(
    #         onp.float, (2, 3), elements=hypothesis.strategies.floats(0.1, 1)),
    # )
    @parameterized.parameters(
        {"fullmatrix": True},
        {"fullmatrix": False},
    )
    def testCGASimpleTwoPlayer(self, fullmatrix, amat=onp.full((2, 3), 0.1)):
        amat = amat + np.eye(2, 3)

        def f(x, y):
            return x.T @ amat @ y

        def g(x, y):
            return -f(x, y)

        eta = 0.1
        num_iter = 10000

        rng = random.PRNGKey(42)
        rng_x, rng_y = random.split(rng)
        init_vals = (random.uniform(rng_x, shape=(2,)),
                     random.uniform(rng_y, shape=(3,)))

        if fullmatrix:
            cga_init, cga_update, get_params = cga.full_solve_cga(
                step_size_f=eta,
                step_size_g=eta,
                f=f,
                g=g,
            )
        else:
            cga_init, cga_update, get_params = cga.cga(
                step_size_f=eta,
                step_size_g=eta,
                f=f,
                g=g,
            )
        grad_yg = jax.grad(g, 1)
        grad_xf = jax.grad(f, 0)

        @jax.jit
        def step(i, opt_state):
            x, y = get_params(opt_state)[:2]
            grads = (grad_xf(x, y), grad_yg(x, y))
            return cga_update(i, grads, opt_state)

        opt_state = cga_init(init_vals)
        for i in range(num_iter):
            opt_state = step(i, opt_state)

        # [-0.03819215 -0.03819215  0.45830577]

        final_values = get_params(opt_state)[:2]
        self.assertAllClose(jax.tree_map(np.zeros_like, final_values),
                            final_values, check_dtypes=True)

    @hypothesis.settings(max_examples=100, deadline=5000.)
    @hypothesis.given(
        hypothesis.extra.numpy.arrays(
            onp.float, (2, 3), elements=hypothesis.strategies.floats(0.1, 1)),
    )
    def testCGAIterationSimpleTwoPlayer(self, amat):
        self.skipTest("until cga is fixed")
        amat = amat + np.eye(2, 3)
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

        solution = cga.cga_iteration(init_vals, f, g, convergence_test,
                                     max_iter, eta)
        self.assertAllClose(jax.tree_map(np.zeros_like, solution.value),
                            solution.value, check_dtypes=True)


if __name__ == "__main__":
    absltest.main()
