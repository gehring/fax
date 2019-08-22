from absl.testing import absltest
from absl.testing import parameterized

import numpy as onp
from numpy import testing

from fax import converge
from fax import loop
from fax import test_util

import jax
import jax.numpy as np
from jax import test_util as jtu
from jax.config import config
config.update("jax_enable_x64", True)


class LoopTest(jtu.JaxTestCase):
    def testFixedPointIteration(self):
        mat_size = 5
        rtol = atol = 1e-10
        max_steps = 5000

        matrix = test_util.generate_stable_matrix(mat_size, eps=1e-2)
        offset = onp.random.rand(5)
        init_x = np.zeros_like(offset)

        def convergence_test(x_new, x_old):
            return converge.max_diff_test(x_new, x_old, rtol, atol)

        def step(x_old):
            return test_util.ax_plus_b(x_old, matrix, offset)

        sol = loop.fixed_point_iteration(
            init_x=init_x,
            func=step,
            convergence_test=convergence_test,
            max_iter=max_steps,
            batched_iter_size=1,
        )

        true_sol = test_util.solve_ax_b(matrix, offset)

        testing.assert_allclose(sol.value, true_sol, rtol=1e-5, atol=1e-5)
        self.assertTrue(sol.converged)
        self.assertLess(sol.iterations, max_steps)
        testing.assert_allclose(sol.value, sol.previous_value, rtol=1e-5,
                                atol=1e-5)

    def testFixedPointDiverge(self):
        rtol = atol = 1e-10
        max_steps = 10

        init_x = np.zeros(())

        def convergence_test(x_new, x_old):
            return converge.max_diff_test(x_new, x_old, rtol, atol)

        def step(x_old):
            return x_old + 1

        sol = loop.fixed_point_iteration(
                init_x=init_x,
                func=step,
                convergence_test=convergence_test,
                max_iter=max_steps,
                batched_iter_size=1,
            )
        self.assertFalse(sol.converged)
        self.assertEqual(sol.iterations, max_steps)

    @parameterized.parameters(
        {"unroll": True},
        {"unroll": False},
    )
    def testBatchedLoop(self, unroll):
        max_steps = 10

        def step(x):
            return x - 1

        init_x = np.zeros(())
        batched_sol = loop.fixed_point_iteration(
            init_x=init_x,
            func=step,
            convergence_test=lambda *args: False,
            max_iter=max_steps,
            batched_iter_size=5,
            unroll=unroll,
        )

        loop_sol = loop.fixed_point_iteration(
            init_x=init_x,
            func=step,
            convergence_test=lambda *args: False,
            max_iter=max_steps,
            batched_iter_size=1,
            unroll=unroll,
        )

        testing.assert_array_equal(batched_sol, loop_sol)

    def testUnrollFixedpointLoop(self):
        max_steps = 10

        def step(x):
            return x - 1

        init_x = np.zeros(())
        scan_sol = loop.fixed_point_iteration(
            init_x=init_x,
            func=step,
            convergence_test=lambda *args: False,
            max_iter=max_steps,
            batched_iter_size=1,
            unroll=True,
        )

        loop_sol = loop.fixed_point_iteration(
            init_x=init_x,
            func=step,
            convergence_test=lambda *args: False,
            max_iter=max_steps,
            batched_iter_size=1,
            unroll=False,
        )

        testing.assert_array_equal(scan_sol, loop_sol)

    def testJITUnrollFixedpointLoop(self):
        max_steps = 10

        def step(x):
            return x - 1

        init_x = np.zeros(())

        @jax.jit
        def run_unrolled(x):
            return loop.fixed_point_iteration(
                init_x=x,
                func=step,
                convergence_test=lambda *args: False,
                max_iter=max_steps,
                batched_iter_size=1,
                unroll=True,
            )

        scan_sol = run_unrolled(init_x)

        @jax.jit
        def run_loop(x):
            return loop.fixed_point_iteration(
                init_x=x,
                func=step,
                convergence_test=lambda *args: False,
                max_iter=max_steps,
                batched_iter_size=1,
                unroll=False,
            )
        loop_sol = run_loop(init_x)

        testing.assert_array_equal(scan_sol, loop_sol)

    def testUnrollGrad(self):
        max_steps = 10

        def step(x):
            return x - 1

        init_x = np.zeros(())

        def run_unrolled(x):
            return loop.fixed_point_iteration(
                init_x=x,
                func=step,
                convergence_test=lambda *args: False,
                max_iter=max_steps,
                batched_iter_size=1,
                unroll=True,
            ).value
        jax.grad(run_unrolled)(init_x)

    def testBatchedRaise(self):
        max_steps = 10

        def step(x):
            return x - 1

        def neg_batch():
            init_x = np.zeros(())
            return loop.fixed_point_iteration(
                init_x=init_x,
                func=step,
                convergence_test=lambda *args: False,
                max_iter=max_steps,
                batched_iter_size=0,
            )

        def oversize_batch():
            init_x = np.zeros(())
            return loop.fixed_point_iteration(
                init_x=init_x,
                func=step,
                convergence_test=lambda *args: False,
                max_iter=max_steps,
                batched_iter_size=max_steps + 1,
            )

        def none_divisible_batch():
            init_x = np.zeros(())
            return loop.fixed_point_iteration(
                init_x=init_x,
                func=step,
                convergence_test=lambda *args: False,
                max_iter=max_steps,
                batched_iter_size=3,
            )

        self.assertRaises(ValueError, neg_batch)
        self.assertRaises(ValueError, oversize_batch)
        self.assertWarns(UserWarning, none_divisible_batch)

    def testNoneMaxIter(self):
        max_steps = None

        def step(x):
            return x + 1

        init_x = np.zeros(())
        loop_sol = loop.fixed_point_iteration(
            init_x=init_x,
            func=step,
            convergence_test=lambda i, *args: i > 10,
            max_iter=max_steps,
            batched_iter_size=3,
        )
        self.assertIsNotNone(loop_sol)
        loop_sol.value.block_until_ready()

    def testUnrolledLoop(self):
        max_steps = 11

        def step(x):
            return x - 1

        init_x = np.zeros(())

        batched_x, batched_x_old = loop.unrolled(
            init_x=init_x,
            func=step,
            num_iter=max_steps,
            return_last_two=True,
        )

        single_batched_x = loop.unrolled(
            init_x=init_x,
            func=step,
            num_iter=max_steps,
            return_last_two=False,
        )

        loop_x, loop_x_old = (0., 0.)
        for _ in range(max_steps):
            loop_x, loop_x_old = step(loop_x), loop_x

        testing.assert_array_equal(batched_x, single_batched_x)
        testing.assert_array_equal(batched_x, loop_x)
        testing.assert_array_equal(batched_x_old, loop_x_old)


if __name__ == "__main__":
    absltest.main()
