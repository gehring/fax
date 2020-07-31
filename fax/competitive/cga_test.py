import random as pyrandom

from absl.testing import absltest
from absl.testing import parameterized

import hypothesis.extra.numpy

import numpy as onp

import jax
from jax import random
from jax import tree_util
import jax.numpy as np
import jax.test_util
from jax.config import config

from fax import converge
from fax.competitive import cga

config.update("jax_enable_x64", True)


def _tree_concatentate(x):
    return tree_util.tree_reduce(lambda a, b: np.concatenate((a, b)), x)


class CGATest(jax.test_util.JaxTestCase):

    @parameterized.parameters(
        {"fullmatrix": False, "conj_grad": True, "order": "simultaneous"},
        {"fullmatrix": False, "conj_grad": True, "order": "alternating"},
        {"fullmatrix": False, "conj_grad": True, "order": "xy"},
        {"fullmatrix": False, "conj_grad": True, "order": "yx"},
        {"fullmatrix": False, "conj_grad": False, "order": "simultaneous"},
        {"fullmatrix": False, "conj_grad": False, "order": "alternating"},
        {"fullmatrix": False, "conj_grad": False, "order": "xy"},
        {"fullmatrix": False, "conj_grad": False, "order": "yx"},
        {"fullmatrix": True, "conj_grad": False, "order": None},
    )
    @hypothesis.settings(max_examples=10, deadline=5000.)
    @hypothesis.given(
        hypothesis.extra.numpy.arrays(
            onp.float, (2, 3), elements=hypothesis.strategies.floats(0.1, 1)),
    )
    def testCGASimpleTwoPlayer(self, fullmatrix, conj_grad, order, amat):
        amat = amat + np.eye(*amat.shape)

        def f(x, y):
            return x.T @ amat @ y + np.dot(y, y)

        def g(x, y):
            return -f(x, y)

        linear_op_solver = None
        if conj_grad:
            linear_op_solver = cga.cg_fixed_point_solve

        eta = 0.1
        num_iter = 3000

        # The hypothesis package takes care of setting the seed of python's
        # random package.
        rng = random.PRNGKey(pyrandom.randint(0, 2 ** 32 - 1))
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
                solve_order=order
            )
        grad_yg = jax.grad(g, 1)
        grad_xf = jax.grad(f, 0)

        @jax.jit
        def step(opt_state):
            x, y = get_params(opt_state)
            grads = (grad_xf(x, y), grad_yg(x, y))
            return cga_update(grads, opt_state)

        opt_state = cga_init(init_vals)
        for i in range(num_iter):
            opt_state = step(opt_state)
            if (np.linalg.norm(opt_state.delta_x) < 1e-8
                    and np.linalg.norm(opt_state.delta_y) < 1e-8):
                break

        final_values = get_params(opt_state)
        self.assertAllClose(
            jax.tree_map(np.zeros_like, final_values),
            final_values,
            check_dtypes=True,
            atol=1e-5,
            rtol=1e-5,
        )

    def testCGASolveOrder(self):
        rng = random.PRNGKey(398513)

        rng, amat_key = random.split(rng)
        amat = random.uniform(amat_key, (2, 3))
        amat = amat + np.eye(*amat.shape)

        def f(x, y):
            return x.T @ amat @ y + np.dot(y, y)

        def g(x, y):
            return -f(x, y)

        linear_op_solver = cga.cg_fixed_point_solve

        eta_f = 0.1
        eta_g = 0.5
        rng_x, rng_y = random.split(rng)
        init_vals = (random.uniform(rng_x, shape=(amat.shape[0],)),
                     random.uniform(rng_y, shape=(amat.shape[1],)))

        grad_yg = jax.grad(g, 1)
        grad_xf = jax.grad(f, 0)

        def two_steps(params, order):
            cga_init, cga_update, get_params = cga.cga(
                step_size_f=eta_f,
                step_size_g=eta_g,
                f=f,
                g=g,
                linear_op_solver=linear_op_solver,
                solve_order=order,
            )
            opt_state = cga_init(params)

            def step(opt_state):
                x, y = get_params(opt_state)
                grads = (grad_xf(x, y), grad_yg(x, y))
                return cga_update(grads, opt_state)

            return get_params(step(step(opt_state)))

        orders = ["simultaneous", "alternating", "xy", "yx"]
        results = {order: two_steps(init_vals, order) for order in orders}

        for order, result in results.items():
            if order != "simultaneous":
                print("Comparing {} and {}".format("simultaneous", order))
                for a, b in zip(results["simultaneous"], result):
                    self.assertArraysAllClose(a, b, check_dtypes=True)

    @parameterized.parameters(
        {"fullmatrix": False, "conj_grad": True, "order": "simultaneous"},
        {"fullmatrix": False, "conj_grad": True, "order": "alternating"},
        {"fullmatrix": False, "conj_grad": True, "order": "xy"},
        {"fullmatrix": False, "conj_grad": True, "order": "yx"},
        {"fullmatrix": False, "conj_grad": False, "order": "simultaneous"},
        {"fullmatrix": False, "conj_grad": False, "order": "alternating"},
        {"fullmatrix": False, "conj_grad": False, "order": "xy"},
        {"fullmatrix": False, "conj_grad": False, "order": "yx"},
        {"fullmatrix": True, "conj_grad": False, "order": None}
    )
    @hypothesis.settings(max_examples=10, deadline=5000.)
    @hypothesis.given(
        hypothesis.extra.numpy.arrays(
            onp.float, (2, 3), elements=hypothesis.strategies.floats(0.1, 1)),
    )
    def testCGAIterationSimpleTwoPlayer(self, fullmatrix, conj_grad, order, amat):
        amat = amat + np.eye(*amat.shape)

        def f(x, y):
            return x.T @ amat @ y + np.dot(y, y)

        def g(x, y):
            return -f(x, y)

        eta = 0.1
        rtol = atol = 1e-8
        max_iter = 3000

        def convergence_test(x_new, x_old):
            return converge.max_diff_test(x_new, x_old, rtol, atol)

        linear_op_solver = None
        if conj_grad:
            linear_op_solver = cga.cg_fixed_point_solve

        # The hypothesis package takes care of setting the seed of python's
        # random package.
        rng = random.PRNGKey(pyrandom.randint(0, 2**32 - 1))
        rng_x, rng_y = random.split(rng)
        init_vals = (random.uniform(rng_x, shape=(amat.shape[0],)),
                     random.uniform(rng_y, shape=(amat.shape[1],)))

        solution = cga.cga_iteration(init_vals, f, g, convergence_test,
                                     max_iter, eta, use_full_matrix=fullmatrix,
                                     linear_op_solver=linear_op_solver, solve_order=order)
        self.assertAllClose(
            jax.tree_map(np.zeros_like, solution),
            solution,
            check_dtypes=True,
            atol=1e-5,
            rtol=1e-5,
        )

    @parameterized.parameters(
        {"fullmatrix": False, "conj_grad": True, "order": "simultaneous"},
        {"fullmatrix": False, "conj_grad": True, "order": "alternating"},
        {"fullmatrix": False, "conj_grad": True, "order": "xy"},
        {"fullmatrix": False, "conj_grad": True, "order": "yx"},
        {"fullmatrix": False, "conj_grad": False, "order": "simultaneous"},
        {"fullmatrix": False, "conj_grad": False, "order": "alternating"},
        {"fullmatrix": False, "conj_grad": False, "order": "xy"},
        {"fullmatrix": False, "conj_grad": False, "order": "yx"},
        {"fullmatrix": True, "conj_grad": False, "order": None}
    )
    @hypothesis.settings(max_examples=10, deadline=5000.)
    @hypothesis.given(
        hypothesis.extra.numpy.arrays(
            onp.float, (2, 3), elements=hypothesis.strategies.floats(0.1, 1)),
    )
    def testTupleCGA(self, fullmatrix, conj_grad, order, amat):
        if fullmatrix:
            self.skipTest((
                "PyTree inputs are not supported by the full-matrix "
                "implementation of CGA."))

        amat = amat + np.eye(*amat.shape)

        def f(x, y):
            return x.T @ amat @ y + np.dot(y, y)

        def g(x, y):
            return -f(x, y)

        def tuple_f(x, y):
            assert isinstance(x, tuple)
            assert isinstance(y, tuple)
            x = np.concatenate(x)
            y = _tree_concatentate(y)
            return f(x, y)

        def tuple_g(x, y):
            assert isinstance(x, tuple)
            assert isinstance(y, tuple)
            x = np.concatenate(x)
            y = _tree_concatentate(y)
            return g(x, y)

        eta = 0.1
        rtol = atol = 1e-8
        max_iter = 3000

        def convergence_test(x_new, x_old):
            return converge.max_diff_test(x_new, x_old, rtol, atol)

        linear_op_solver = None
        if conj_grad:
            linear_op_solver = cga.cg_fixed_point_solve

        # The hypothesis package takes care of setting the seed of python's
        # random package.
        rng = random.PRNGKey(pyrandom.randint(0, 2 ** 32 - 1))
        rng_x, rng_y = random.split(rng)

        init_vals = (random.uniform(rng_x, shape=(amat.shape[0],)),
                     random.uniform(rng_y, shape=(amat.shape[1],)))

        tuple_y = np.split(init_vals[1], (1,))
        tuple_y = (tuple_y[0], tuple(np.split(tuple_y[1], (1,))))
        init_tuple_vals = (tuple(np.split(init_vals[0], (1,))),
                           tuple_y)

        tuple_sol = cga.cga_iteration(init_tuple_vals, tuple_f, tuple_g,
                                      convergence_test, max_iter, eta,
                                      use_full_matrix=fullmatrix,
                                      linear_op_solver=linear_op_solver,
                                      solve_order=order)

        solution = cga.cga_iteration(init_vals, f, g, convergence_test,
                                     max_iter, eta, use_full_matrix=fullmatrix,
                                     linear_op_solver=linear_op_solver,
                                     solve_order=order)

        # check if tuple type is preserved
        self.assertTrue(isinstance(tuple_sol[0], tuple))
        self.assertTrue(isinstance(tuple_sol[1], tuple))

        # check if tuple type is preserved
        self.assertTrue(isinstance(tuple_sol, tuple))

        # check if output is the same for tuple inputs vs a single array
        self.assertAllClose(
            tuple((_tree_concatentate(x) for x in tuple_sol)),
            solution,
            check_dtypes=True,
            rtol=1e-8,
            atol=1e-8,
        )


if __name__ == "__main__":
    absltest.main()
