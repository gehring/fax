from absl.testing import absltest
from absl.testing import parameterized

import hypothesis.extra.numpy

import numpy as onp

from fax import converge
from fax.lagrangian import cg
from fax.lagrangian import cga

import jax
import jax.numpy as np
from jax import random
import jax.test_util

from jax.config import config
config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", True)


class CGATest(jax.test_util.JaxTestCase):

    @parameterized.parameters(
        {"fullmatrix": False, "conj_grad": True},
        {"fullmatrix": False, "conj_grad": False},
        {"fullmatrix": True, "conj_grad": False},
    )
    @hypothesis.settings(max_examples=100, deadline=5000.)
    @hypothesis.given(
        hypothesis.extra.numpy.arrays(
            onp.float, (2, 3), elements=hypothesis.strategies.floats(0.1, 1)),
    )
    def testCGASimpleTwoPlayer(self, fullmatrix, conj_grad, amat):
        amat = amat + np.eye(*amat.shape)

        def f(x, y):
            return x.T @ amat @ y + np.dot(y, y)

        def g(x, y):
            return -f(x, y)

        linear_op_solver = None
        if conj_grad:
            linear_op_solver = cg.fixed_point_solve

        eta = 0.1
        num_iter = 3000

        rng = random.PRNGKey(42)
        rng_x, rng_y = random.split(rng)
        init_vals = (random.uniform(rng_x, shape=(amat.shape[0],)),
                     random.uniform(rng_y, shape=(amat.shape[1],)))

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
                linear_op_solver=linear_op_solver,
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

        final_values = get_params(opt_state)[:2]
        self.assertAllClose(jax.tree_map(np.zeros_like, final_values),
                            final_values, check_dtypes=True)

    @parameterized.parameters(
        {"fullmatrix": False, "conj_grad": True},
        {"fullmatrix": False, "conj_grad": False},
        {"fullmatrix": True, "conj_grad": False},
    )
    @hypothesis.settings(max_examples=100, deadline=5000.)
    @hypothesis.given(
        hypothesis.extra.numpy.arrays(
            onp.float, (2, 3), elements=hypothesis.strategies.floats(0.1, 1)),
    )
    def testCGAIterationSimpleTwoPlayer(self, fullmatrix, conj_grad, amat):
        amat = np.array([[0.32651018, 0.1, 0.32651018],
                         [0.32651018, 0.32651018, 0.32651018]])
        amat = amat + np.eye(*amat.shape)

        def f(x, y):
            return x.T @ amat @ y + np.dot(y, y)

        def g(x, y):
            return -f(x, y)

        eta = 0.1
        rtol = atol = 1e-7
        max_iter = 3000

        def convergence_test(x_new, x_old):
            return converge.max_diff_test(x_new, x_old, rtol, atol)

        linear_op_solver = None
        if conj_grad:
            linear_op_solver = cg.fixed_point_solve

        rng = random.PRNGKey(42)
        rng_x, rng_y = random.split(rng)
        init_vals = (random.uniform(rng_x, shape=(amat.shape[0],)),
                     random.uniform(rng_y, shape=(amat.shape[1],)))

        solution = cga.cga_iteration(init_vals, f, g, convergence_test,
                                     max_iter, eta, use_full_matrix=fullmatrix,
                                     linear_op_solver=linear_op_solver)
        self.assertAllClose(jax.tree_map(np.zeros_like, solution.value),
                            solution.value, check_dtypes=True)


if __name__ == "__main__":
    absltest.main()
