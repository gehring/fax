import collections
import itertools
import math
import zipfile
from typing import Callable

import hypothesis.extra.numpy
import hypothesis.strategies
import jax
import jax.numpy as np
import jax.scipy
import jax.test_util
import numpy as onp
from numpy import testing

_basic_math_context = {}
for m in dir(math):
    try:
        _basic_math_context[m] = np.__getattribute__(m)
    except AttributeError:
        pass


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

    Args:
        n (integer): Number of components for the fixed vector `v`
    """

    def func(u):
        return np.dot(u, v)

    def equality_constraints(u):
        return np.linalg.norm(u) - 1

    optimal_solution = -(1. / np.linalg.norm(v)) * v
    optimal_value = np.dot(optimal_solution, v)

    return func, equality_constraints, optimal_solution, optimal_value


def get_list(rows):
    param_list = []
    skipped = []
    for row in rows:
        row_text = row.lstrip()
        if not row_text:
            continue

        if row_text.startswith("!"):
            skipped.append(row_text)
            continue

        if {">=", "<=", "<", ">"}.intersection(row_text):
            raise NotImplementedError("no inequalities")

        if row_text[0].isupper() and row.replace(" ", "").isalpha():
            assert row_text.startswith("End")
            return param_list, skipped
        else:
            param_list.append(row_text)
    raise ValueError


def get_struct(rows):
    struct = {}
    skipped = []
    for row in rows:
        # print(row)
        if not row:
            continue

        if row[0] == '!' or row[0] == '#':
            skipped.append(row)
            continue

        row_text = row.lstrip()
        if row_text == "End Model":
            continue

        if row_text[0].isupper():
            struct[row_text], skipped_ = get_struct(rows)
            skipped.extend(skipped_)
        else:
            params, skipped_ = get_list(itertools.chain([row], rows))
            skipped.extend(skipped_)
            return params, skipped
    return struct, skipped


def parse_apm(text):
    # assert text.count("Model") == 2
    if "Intermediates" in text:
        return
    if "does not exist" in text:
        return
    rows = iter(text.splitlines())
    try:
        struct, skipped = get_struct(rows)
    except NotImplementedError:
        return

    assert len(struct) == 1

    for model, model_struct in struct.items():
        closure = {}
        var_sizes = collections.defaultdict(int)
        for obj in model_struct["Variables"]:
            if obj == "obj":
                continue

            variable, value = obj.split("=")
            var, size = variable.split("[")
            size, _ = size.split("]")
            if ":" in size:
                size = max(int(s) for s in size.split(":"))
            else:
                size = int(size)

            var_sizes[var] = max(var_sizes[var], size)

        for k, v in var_sizes.items():
            closure[k] = np.zeros(v)

        # print(closure)
        for obj in model_struct["Equations"]:
            variable, equation = (o.strip() for o in obj.split("="))

            if "obj" in variable:
                # By default we maximize here.
                equation = "-(" + equation + ")"

                cost_function = text_to_code(variable, equation, closure)
                print(cost_function)
                exec(cost_function, _basic_math_context, closure)

    assert "obj" in closure and len(closure) > 1
    # print("================================================================")
    for idx, comment in enumerate(skipped):
        if "! best known objective =" in comment:
            _, optimal_solution = comment.split("=")

            optimal_solution = eval(optimal_solution.strip(), {}, _basic_math_context)
            optimal_solution = -np.array(float(optimal_solution))
            break
    else:
        raise ValueError("No solution found")
    del skipped[idx]

    (model_name, model_struct), = list(struct.items())
    func = closure['obj']

    state_space = closure['x']
    constraints = []

    for equation in model_struct['Equations']:
        lhs, rhs = equation.split("=")
        if lhs.strip() != 'obj':
            if not set(rhs.strip()).difference({'0', '.', ','}):
                lhs = f"{lhs} - {rhs}"

            constraint_variable = f"h{len(constraints)}"
            closure = {"x": state_space.copy(), }

            # TODO: split in two steps, load and dump python ascii code + exec
            cost_function = text_to_code(constraint_variable, lhs, closure)
            # print("Costraint:", cost_function)
            exec(cost_function, {}, closure)
            constraints.append(closure[constraint_variable])

    if not constraints or len(constraints) > 1:
        # print("SKIPPING", constraints)
        return

    def equality_constraints(params):
        if len(constraints) > 1:
            raise NotImplementedError

        constraint, = constraints
        return constraint(params)

    # optimal_solution = np.array([1.] + [0.] * (n - 1))
    # optimal_value = -1.

    # maximise fx - lambda
    return func, equality_constraints, optimal_solution, state_space, model_name


def text_to_code(variable, equation, closure):
    cost_function = equation.replace("^", "**")
    seq = []
    for a in cost_function.split("]"):
        if "[" not in a:
            seq.append(a)
        else:
            rest, num = a.split("[")
            b = f"{rest}[{int(num) - 1}"
            seq.append(b)
    cost_function = "]".join(seq)
    scope = ", ".join(k for k in closure.keys() if not k.startswith("__"))
    cost_function_ = f"{variable} = lambda {scope}: {cost_function}"
    return cost_function_


def load_HockSchittkowski_models():
    with zipfile.ZipFile('/home/esac/research/fax/fax/hs.zip') as test_archive:
        for test_case_path in test_archive.filelist:
            with test_archive.open(test_case_path) as test_case:
                retr = parse_apm(test_case.read().decode('utf-8'))
                if retr is not None:
                    yield retr
