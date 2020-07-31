from absl.testing import absltest
from absl.testing import parameterized

import numpy as onp
from numpy import testing

from fax import converge
from fax import loop
from fax import test_util

import jax
import jax.numpy as np
import jax.test_util
from jax.config import config
config.update("jax_enable_x64", True)


class LoopTest(jax.test_util.JaxTestCase):

    @parameterized.parameters(
        {"unroll": True, "jit": False},
        {"unroll": False, "jit": False},
        {"unroll": True, "jit": True},
        {"unroll": False, "jit": True},
    )
    def testFixedPointIteration(self, unroll, jit):
        mat_size = 5
        rtol = atol = 1e-10
        max_steps = 500

        matrix = test_util.generate_stable_matrix(mat_size, eps=1e-1)
        offset = onp.random.rand(5)
        init_x = np.zeros_like(offset)

        def convergence_test(x_new, x_old):
            return converge.max_diff_test(x_new, x_old, rtol, atol)

        def step(x_old):
            return test_util.ax_plus_b(x_old, matrix, offset)

        def solve(x):
            return loop.fixed_point_iteration(
                init_x=x,
                func=step,
                convergence_test=convergence_test,
                max_iter=max_steps,
                batched_iter_size=1,
                unroll=unroll,
            )

        if jit:
            solve = jax.jit(solve)

        sol = solve(init_x)

        true_sol = test_util.solve_ax_b(matrix, offset)

        testing.assert_allclose(sol.value, true_sol, rtol=1e-5, atol=1e-5)
        self.assertTrue(sol.converged)
        self.assertLessEqual(sol.iterations, max_steps)
        testing.assert_allclose(sol.value, sol.previous_value, rtol=1e-5,
                                atol=1e-5)

    @parameterized.parameters(
        {"unroll": True},
        {"unroll": False},
    )
    def testFixedPointDiverge(self, unroll):
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
                unroll=unroll,
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

    @parameterized.parameters(
        {"unroll": True},
        {"unroll": False},
    )
    def testTermination(self, unroll):
        if unroll:
            self.skipTest((
                "Can't terminate early when unrolling until `jax.lax.cond` "
                "supports differentiation."))
        max_steps = 10

        def step(x):
            return x - 1

        init_x = np.zeros(())
        term_sol = loop.fixed_point_iteration(
            init_x=init_x,
            func=step,
            convergence_test=lambda x, *args: x <= -5,
            max_iter=max_steps,
            batched_iter_size=1,
            unroll=unroll,
        )

        last_i, fixed_value = loop.unrolled(
            0,
            init_x=init_x,
            func=step,
            num_iter=5,
            return_last_two=False,
        )

        self.assertEqual(fixed_value, term_sol.value)
        self.assertEqual(last_i, term_sol.iterations)

    def testUnrollFixedpointLoop(self):
        max_steps = 10

        def step(x):
            return x - 1

        init_x = np.zeros(())
        unroll_sol = loop.fixed_point_iteration(
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

        testing.assert_array_equal(unroll_sol, loop_sol)

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

        unroll_sol = run_unrolled(init_x)

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

        testing.assert_array_equal(unroll_sol, loop_sol)

        jax.grad(lambda x: run_unrolled(x).value)(init_x)

    @parameterized.parameters(
        {"jit": False},
        {"jit": True},
    )
    def testUnrollGrad(self, jit):
        max_steps = 10

        def step(x):
            return x*0.1

        def converge_test(x_new, x_old):
            return np.max(x_new - x_old) < 1e-3

        init_x = np.ones(())

        def run_unrolled(x):
            return loop.fixed_point_iteration(
                init_x=x,
                func=step,
                convergence_test=converge_test,
                max_iter=max_steps,
                batched_iter_size=1,
                unroll=True,
            ).value

        grad_fun = jax.grad(run_unrolled)
        if jit:
            grad_fun = jax.jit(grad_fun)
        grad_fun(init_x)

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

        batched_i, batched_x, batched_x_old = loop.unrolled(
            0,
            init_x=init_x,
            func=step,
            num_iter=max_steps,
            return_last_two=True,
        )

        single_i, single_batched_x = loop.unrolled(
            0,
            init_x=init_x,
            func=step,
            num_iter=max_steps,
            return_last_two=False,
        )

        loop_x, loop_x_old = (0., 0.)
        for i in range(max_steps):
            loop_x, loop_x_old = step(loop_x), loop_x

        testing.assert_array_equal(batched_x, single_batched_x)
        testing.assert_array_equal(batched_x, loop_x)
        testing.assert_array_equal(batched_x_old, loop_x_old)

        self.assertEqual(i + 1, single_i)
        self.assertEqual(i + 1, batched_i)


def _fixedpoint_iteration_solver(unroll,
                                 param_func,
                                 default_rtol=1e-10,
                                 default_atol=1e-10,
                                 default_max_iter=200,
                                 default_batched_iter_size=1):

        def fixed_point_iteration_solver(init_x, params):
            rtol, atol = converge.adjust_tol_for_dtype(default_rtol,
                                                       default_atol,
                                                       init_x.dtype)

            def convergence_test(x_new, x_old):
                return converge.max_diff_test(x_new, x_old, rtol, atol)

            func = param_func(params)
            sol = loop.fixed_point_iteration(
                init_x=init_x,
                func=func,
                convergence_test=convergence_test,
                max_iter=default_max_iter,
                batched_iter_size=default_batched_iter_size,
                unroll=unroll,
            )

            return sol.value
        return fixed_point_iteration_solver


class UnrolledFixedPointIterationTest(test_util.FixedPointTestCase):

    def make_solver(self, param_func):
        return _fixedpoint_iteration_solver(unroll=True, param_func=param_func)


if __name__ == "__main__":
    absltest.main()
