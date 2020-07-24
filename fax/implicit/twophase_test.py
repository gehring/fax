from absl.testing import absltest

from fax import test_util
from fax.implicit import twophase

from jax.config import config
config.update("jax_enable_x64", True)


class TwoPhaseOpsTest(test_util.FixedPointTestCase):

    def make_solver(self, param_func):
        def solve(x, params):
            return twophase.two_phase_solve(param_func, x, params)
        return solve


if __name__ == "__main__":
    absltest.main()
