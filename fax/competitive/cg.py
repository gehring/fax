from functools import partial

from jax import lax
from jax import tree_util

from fax import converge
from fax import loop
from fax import math


def cg_step(a_lin_op, i, current_state):
    del i
    x, r, r_sqr, p = current_state
    amat_p = a_lin_op(p)

    alpha = r_sqr / math.pytree_dot(p, amat_p)
    x_new = tree_util.tree_multimap(lambda x, p: x + alpha * p, x, p)
    r_new = tree_util.tree_multimap(lambda r, amat_p: r - alpha * amat_p,
                                    r, amat_p)

    r_new_sqr = math.pytree_dot(r_new, r_new)
    beta = r_new_sqr/r_sqr
    p_new = tree_util.tree_multimap(lambda r_new, p: r_new + beta * p, r_new, p)
    return x_new, r_new, r_new_sqr, p_new


def conjugate_gradient_solve(linear_op, bvec, init_x, max_iter=1000,
                             atol=1e-10):
    dtype = converge.tree_smallest_float_dtype(bvec)
    _, atol = converge.adjust_tol_for_dtype(0., atol=atol, dtype=dtype)
    init_r = tree_util.tree_multimap(lax.sub, bvec, linear_op(init_x))
    init_p = init_r
    init_r_sqr = math.pytree_dot(init_r, init_r)

    def convergence_test(state_new, state_old):
        del state_old
        return state_new[2] < atol

    solution = loop.fixed_point_iteration(
        (init_x, init_r, init_r_sqr, init_p),
        func=partial(cg_step, linear_op),
        convergence_test=convergence_test,
        max_iter=max_iter
    )
    return solution._replace(
        value=solution.value[0],
        previous_value=solution.value[0],
    )


def fixed_point_solve(linear_op, bvec, init_x, max_iter=1000,
                      atol=1e-10):
    return conjugate_gradient_solve(
        linear_op=lambda x: tree_util.tree_multimap(lax.sub, x, linear_op(x)),
        bvec=bvec,
        init_x=init_x,
        max_iter=max_iter,
        atol=atol,
    )
