from absl.testing import absltest

from fax import test_util
from fax.implicit import twophase

from jax.config import config
config.update("jax_enable_x64", True)


class TwoPhaseOpsTest(test_util.FixedPointTestCase):

    def make_solver(self, param_func):
        return twophase.two_phase_solver(
            param_func=param_func,
            default_rtol=1e-10,
            default_atol=1e-10,
            default_max_iter=10000,
        )


if __name__ == "__main__":
    absltest.main()
