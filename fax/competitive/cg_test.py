from absl.testing import absltest

import hypothesis.extra.numpy

import numpy as onp

from fax.competitive import cg

import jax.numpy as np
import jax.test_util

from jax.config import config
config.update("jax_enable_x64", True)


class CGTest(jax.test_util.JaxTestCase):

    @hypothesis.settings(max_examples=200, deadline=5000.)
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
                            check_dtypes=True,
                            atol=1e-10,
                            rtol=1e-5)

    @hypothesis.settings(max_examples=100, deadline=5000.)
    @hypothesis.given(
        hypothesis.extra.numpy.arrays(
            onp.float, (3, 3), elements=hypothesis.strategies.floats(0.1, 1)),
        hypothesis.extra.numpy.arrays(
            onp.float, (3,), elements=hypothesis.strategies.floats(0.1, 1)),
    )
    def testTupleSolveSimpleCase(self, amat, bvec):
        amat = np.eye(*amat.shape) + amat
        amat = amat.T @ amat

        split_at = 2
        tuple_bvec = tuple(np.split(bvec, (split_at,), axis=0))

        def linear_op(x):
            return np.dot(amat, x)

        def tuple_linear_op(x):
            return tuple(np.split(linear_op(np.concatenate(x)), (split_at,)))

        init_tuple = (np.zeros((split_at,)),
                      np.zeros((bvec.shape[0] - split_at,)))

        tuple_sol = cg.conjugate_gradient_solve(tuple_linear_op, tuple_bvec,
                                                init_tuple)

        solution = cg.conjugate_gradient_solve(linear_op, bvec,
                                               np.zeros_like(bvec))

        # check if tuple type is preserved
        self.assertTrue(isinstance(tuple_sol.value, tuple))

        # check if output is the same for tuple inputs vs a single array
        self.assertAllClose(np.concatenate(tuple_sol.value),
                            solution.value,
                            check_dtypes=True,
                            rtol=1e-8,
                            atol=1e-8)

        # the number of iterations done should be the same
        self.assertEqual(tuple_sol.iterations, solution.iterations)


if __name__ == "__main__":
    absltest.main()
