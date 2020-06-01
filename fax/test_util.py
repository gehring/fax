from typing import Callable

import hypothesis.extra.numpy
import hypothesis.strategies
import jax
import jax.numpy as np
import jax.scipy
import jax.test_util
import numpy as onp
from numpy import testing


def generate_stable_matrix(size, eps=1e-2):
    """Generate a random matrix who's singular values are less than 1 - `eps`.

    Args:
        size (int): The size of the matrix. The dimensions of the matrix will be
            `size`x`size`.
        eps (float): A float between 0 and 1. The singular values will be no
            larger than 1 - `eps`.

    Returns:
        A `size`x`size` matrix with singular values less than 1 - `eps`.
    """
    mat = onp.random.rand(size, size)
    return make_stable(mat, eps)


def make_stable(matrix, eps):
    u, s, vt = onp.linalg.svd(matrix)
    s = onp.clip(s, 0, 1 - eps)
    return u.dot(s[:, None] * vt)


def ax_plus_b(xvec, amat, bvec):
    """Compute Ax + b using tensorflow.

    Args:
        xvec: A vector which will be multiplied by `amat`.
        amat: A matrix to be used to multiply the vector `xvec`.
        bvec: A vector to add to the matrix-vector product `amat` x `xvec`.

    Returns:
        A vector equal to the matrix-vector product `amat` x `xvec` plus `bvec`.
    """
    return np.tensordot(amat, xvec, 1) + bvec


def solve_ax_b(amat, bvec):
    """Solve for the fixed point x = Ax + b.


    Args:
        amat: A contractive matrix.
        bvec: The vector offset.

    Returns:
        A vector `x` such that x = Ax + b.
    """
    matrix = np.eye(amat.shape[0]) - amat
    return np.linalg.solve(matrix, bvec)


def solve_grad_ax_b(amat, bvec):
    """Solve for the gradient of the fixed point x = Ax + b.

    Args:
        amat: A contractive matrix.
        bvec: The vector offset.

    Returns:
        3-D array: The partial derivative of the fixed point for a given element
            of `amat`.
        2-D array: The partial derivative of the fixed point for a given element
            of `bvec`.
    """
    matrix = np.eye(amat.shape[0]) - amat
    grad_bvec = np.linalg.solve(matrix.T, np.ones(matrix.shape[0]))
    grad_matrix = grad_bvec[:, None] * np.linalg.solve(matrix, bvec)[None, :]
    return grad_matrix, grad_bvec


def param_ax_plus_b(params):
    matrix, offset = params
    return lambda i, x: ax_plus_b(x, matrix, offset)


class FixedPointTestCase(jax.test_util.JaxTestCase):

    def make_solver(self, param_func):
        del param_func
        raise NotImplementedError

    @hypothesis.settings(max_examples=100, deadline=5000.)
    @hypothesis.given(
        hypothesis.extra.numpy.arrays(
            onp.float, (5, 5), elements=hypothesis.strategies.floats(0, 1)),
        hypothesis.extra.numpy.arrays(
            onp.float, 5, elements=hypothesis.strategies.floats(0, 1)),
    )
    def testSimpleContraction(self, matrix, offset):
        matrix = make_stable(matrix, eps=1e-1)
        x0 = np.zeros_like(offset)

        solver = self.make_solver(param_ax_plus_b)
        self.assertSimpleContraction(solver, x0, matrix, offset)

    @hypothesis.settings(max_examples=100, deadline=5000.)
    @hypothesis.given(
        hypothesis.extra.numpy.arrays(
            onp.float, (5, 5), elements=hypothesis.strategies.floats(0.1, 1)),
        hypothesis.extra.numpy.arrays(
            onp.float, 5, elements=hypothesis.strategies.floats(0.1, 1)),
    )
    def testJVP(self, matrix, offset):
        matrix = make_stable(matrix, eps=1e-1)
        x0 = np.zeros_like(offset)
        solver = self.make_solver(param_ax_plus_b)

        f = lambda *args: solver(*args).value
        f_vjp = lambda *args: jax.vjp(f, *args)
        jax.test_util.check_vjp(f, f_vjp, (x0, (matrix, offset)),
                                rtol=1e-4, atol=1e-4)

    def assertSimpleContraction(self, solver, x0, matrix, offset):
        true_sol = solve_ax_b(matrix, offset)
        sol = solver(x0, (matrix, offset))

        testing.assert_allclose(sol.value, true_sol, rtol=1e-5, atol=1e-5)

    def testGradient(self):
        """
        Test gradient on the fixed point of Ax + b = x.
        """
        mat_size = 10
        matrix = generate_stable_matrix(mat_size, 1e-1)
        offset = onp.random.rand(mat_size)
        x0 = np.zeros_like(offset)

        solver = self.make_solver(param_ax_plus_b)

        def loss(x, params): return np.sum(solver(x, params).value)

        jax.test_util.check_grads(
            loss,
            (x0, (matrix, offset),),
            order=1,
            modes=["rev"],
            atol=1e-4,
            rtol=1e-4,
        )
        self.assertSimpleContractionGradient(loss, x0, matrix, offset)

    def assertSimpleContractionGradient(self, loss, x0, matrix, offset):
        grad_matrix, grad_offset = jax.grad(loss, 1)(x0, (matrix, offset))

        true_grad_matrix, true_grad_offset = solve_grad_ax_b(matrix, offset)

        testing.assert_allclose(grad_matrix, true_grad_matrix,
                                rtol=1e-5, atol=1e-5)
        testing.assert_allclose(grad_offset, true_grad_offset,
                                rtol=1e-5, atol=1e-5)


def constrained_opt_problem(n) -> (Callable, Callable, np.array, float):
    def func(params):
        return params[0]

    def equality_constraints(params):
        return np.sum(params ** 2) - 1

    optimal_solution = np.array([1.] + [0.] * (n - 1))

    optimal_value = -1.
    return func, equality_constraints, optimal_solution, optimal_value


def dot_product_minimization(v):
    """Problem: find a u such that np.dot(u, v) is maximum, subject to np.linalg.norm(u) = 1.
    """

    def func(u):
        return np.dot(u, v)

    def equality_constraints(u):
        return np.linalg.norm(u) - 1

    optimal_solution = -(1. / np.linalg.norm(v)) * v
    optimal_value = np.dot(optimal_solution, v)

    return func, equality_constraints, optimal_solution, optimal_value


