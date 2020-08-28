import functools
import operator
from absl.testing import absltest

from fax import test_util
from fax.implicit import root
from fax.implicit import twophase

import numpy as np

import jax
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)


# TODO: make proper root solving tests
class RootSolveTest(test_util.FixedPointTestCase):

    def make_solver(self, param_func):
        def func(x, params):
            return jax.tree_multimap(operator.sub, x, param_func(params)(x))

        def solve(x, params):
            return root.root_solve(
                func,
                x,
                params,
                solver=functools.partial(twophase.default_solver(), param_func),
            )
        return solve

    def testSecondOrder(self):
        step_size = 0.1

        def sqrt_step(target):
            def _sqrt_step(x):
                return x + step_size * (target - x**2)
            return _sqrt_step

        solver = self.make_solver(sqrt_step)

        test_input = jnp.array([0.1, 0.5, 2., 10.])
        out = jax.vmap(jax.grad(jax.grad(lambda x: solver(x, x))))(test_input)
        np.testing.assert_allclose(out, -0.25 * test_input**(-3/2))


if __name__ == "__main__":
    absltest.main()
