from absl.testing import absltest

import hypothesis.extra.numpy

import numpy as onp

from fax.lagrangian import cg

import jax.numpy as np
import jax.test_util

from jax.config import config
config.update("jax_enable_x64", True)


class CGTest(jax.test_util.JaxTestCase):

    @hypothesis.settings(max_examples=500, deadline=5000.)
    @hypothesis.given(
        hypothesis.extra.numpy.arrays(
            onp.float, (3, 3), elements=hypothesis.strategies.floats(0.1, 1)),
        hypothesis.extra.numpy.arrays(
            onp.float, (3,), elements=hypothesis.strategies.floats(0.1, 1)),
    )
    def testSolveSimpleCase(self, amat, bvec):
        amat = np.eye(*amat.shape) + amat
        amat = amat.T @ amat

        solution = cg.conjugate_gradient_solve(lambda x: np.dot(amat, x), bvec,
                                               np.zeros_like(bvec))

        self.assertTrue(solution.converged)
        self.assertAllClose(np.linalg.solve(amat, bvec), solution.value,
                            check_dtypes=True)


if __name__ == "__main__":
    absltest.main()