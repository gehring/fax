import hypothesis.extra.numpy
import hypothesis.strategies
import jax.numpy as np
import jax.test_util
import numpy as onp
from absl.testing import absltest
from jax import random
from jax.config import config

import fax
from fax import converge
from fax.competitive import extragradient

config.update("jax_enable_x64", True)


class CGATest(jax.test_util.JaxTestCase):
    stop_criterion_params = dict(rtol=1e-12, atol=1e-12)
    convergence_params = dict(rtol=1e-6, atol=1e-6, check_dtypes=True)

    @hypothesis.settings(max_examples=10, deadline=5000.)
    @hypothesis.given(
        hypothesis.extra.numpy.arrays(
            onp.float, (2, 3), elements=hypothesis.strategies.floats(0.1, 1)),
    )
    def testEgSimpleTwoPlayer(self, amat):
        step_size = 1e-1
        max_iter = 1000
        amat = amat + np.eye(*amat.shape)

        def function(x, y):
            return y.T @ amat @ x + np.dot(x, x)

        rng = random.PRNGKey(0)
        rng_x, rng_y = random.split(rng)
        initial_values = (random.uniform(rng_x, shape=(amat.shape[1],)), random.uniform(rng_y, shape=(amat.shape[0],)))

        def convergence_test(x_new, x_old):
            return converge.max_diff_test(x_new, x_old, **CGATest.stop_criterion_params)

        optimizer_init, optimizer_update, optimizer_get_params = extragradient.rprop_extragradient_optimizer(
            step_size_x=step_size,
            step_size_y=step_size,
        )
        grad_x = jax.grad(function, 0)
        grad_y = jax.grad(function, 1)
        body = lambda i, x: optimizer_update(i, (grad_x, grad_y), x)

        solution = fax.loop.fixed_point_iteration(
            init_x=optimizer_init(initial_values),
            func=body,
            convergence_test=convergence_test,
            max_iter=max_iter,
            get_params=optimizer_get_params,
        )
        x, y = solution.value
        # final_val = function(*solution.value)
        # print(x, y, final_val)
        print(x - np.zeros_like(x))
        self.assertAllClose(x, np.zeros_like(x), **CGATest.convergence_params)


if __name__ == "__main__":
    absltest.main()
