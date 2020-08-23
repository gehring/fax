"""
Implementation of GMRES
"""
import functools
import operator

from jax import lax
import jax.numpy as jnp
import jax.ops
import jax.scipy as jsp
import jax.tree_util

_dot = functools.partial(
    jax.tree_multimap,
    functools.partial(jnp.dot, precision=lax.Precision.HIGHEST),
)
_add = functools.partial(jax.tree_multimap, jnp.add)
_sub = functools.partial(jax.tree_multimap, jnp.subtract)


def _vdot_tree(x, y):
    return sum(jax.tree_leaves(jax.tree_multimap(jnp.vdot, x, y)))


def _project_on_columns(A, v):
    v_proj = jax.tree_multimap(
        lambda X, y: jnp.tensordot(X.T.conj(), y, axes=X.ndim - 1),
        A,
        v,
    )
    return jax.tree_util.tree_reduce(operator.add, v_proj)


def _normalize(x, return_norm=False):
    norm = jnp.sqrt(_vdot_tree(x, x))
    normalized_x = jax.tree_map(lambda v: v / norm, x)
    if return_norm:
        return normalized_x, norm
    else:
        return normalized_x


def _identity(x):
    return x


def _iterative_classical_gram_schmidt(Q, x, iterations=2):
    """Orthogonalize x against the columns of Q."""
    # "twice is enough"
    # http://slepc.upv.es/documentation/reports/str1.pdf

    # This assumes that Q's leaves all have the same dimension in the last
    # axis.
    r = jnp.zeros((jax.tree_leaves(Q)[0].shape[-1]))
    q = x

    for _ in range(iterations):
        h = _project_on_columns(Q, q)
        q = _sub(q, jax.tree_map(lambda X: jnp.dot(X, h), Q))
        r = _add(r, h)
    return q, r


def arnoldi_iteration(A, b, n, M=None):
    # https://en.wikipedia.org/wiki/Arnoldi_iteration#The_Arnoldi_iteration
    if M is None:
        M = _identity
    q = _normalize(b)
    Q = jax.tree_map(
        lambda x: jnp.pad(x[..., None], ((0, 0),) * x.ndim + ((0, n),)),
        q,
    )
    H = jnp.zeros((n, n + 1), jnp.result_type(*jax.tree_leaves(b)))

    def step(carry, k):
        Q, H = carry
        q = jax.tree_map(lambda x: x[..., k], Q)
        v = A(M(q))
        v, h = _iterative_classical_gram_schmidt(Q, v, iterations=1)
        v, v_norm = _normalize(v, return_norm=True)
        Q = jax.tree_multimap(lambda X, y: X.at[..., k + 1].set(y), Q, v)
        h = h.at[k + 1].set(v_norm)
        H = H.at[k, :].set(h)
        return (Q, H), None

    (Q, H), _ = lax.scan(step, (Q, H), jnp.arange(n))
    return Q, H


@jax.jit
def lstsq(a, b):
    # slightly faster than jnp.linalg.lstsq
    return jsp.linalg.solve(_dot(a.T, a), _dot(a.T, b), sym_pos=True)


def _gmres(A, b, x0, n, M, residual=None):
    # https://www-users.cs.umn.edu/~saad/Calais/PREC.pdf
    Q, H = arnoldi_iteration(A, b, n, M)
    if residual is None:
        residual = _sub(b, A(x0))
    beta = jnp.sqrt(_vdot_tree(residual, residual))
    e1 = jnp.concatenate([jnp.ones((1,), beta.dtype), jnp.zeros((n,), beta.dtype)])
    y = lstsq(H.T, beta * e1)

    dx = M(jax.tree_map(lambda X: jnp.dot(X[..., :-1], y), Q))
    x = _add(x0, dx)
    return x


def _gmres_solve(A, b, x0, *, tol, atol, restart, maxiter, M):
    bs = _vdot_tree(b, b)
    atol2 = jnp.maximum(jnp.square(tol) * bs, jnp.square(atol))
    num_restarts = maxiter // restart

    def cond_fun(value):
        x, residual, k = value
        sqr_error = _vdot_tree(residual, residual)
        return (sqr_error > atol2) & (k < num_restarts)

    def body_fun(value):
        x, residual, k = value
        x = _gmres(A, b, x, restart, M, residual)
        residual = _sub(b, A(x))
        return x, residual, k + 1

    residual = _sub(b, A(x0))
    if num_restarts:
        x, residual, _ = jax.lax.while_loop(
            cond_fun, body_fun, (x0, residual, 0))
    else:
        x = x0

    k = maxiter % restart
    sqr_error = _vdot_tree(residual, residual)
    if k > 0:
        x_final = jax.lax.cond(
            sqr_error > atol2,
            true_fun=lambda values: _gmres(A, b, values[0], k, M, values[1]),
            false_fun=lambda values: values[0],
            operand=(x, residual),
        )
    else:
        x_final = x
    return x_final


def gmres(A, b, x0=None, *, tol=1e-5, atol=0.0, restart=20, maxiter=None,
          M=None):
    if x0 is None:
        x0 = jax.tree_map(jnp.zeros_like, b)
    if M is None:
        M = _identity

    size = sum(bi.size for bi in jax.tree_leaves(b))
    if maxiter is None:
        maxiter = 10 * size  # copied from scipy
    if restart > size:
        restart = size

    if jax.tree_structure(x0) != jax.tree_structure(b):
        raise ValueError(
            'x0 and b must have matching tree structure: '
            f'{jax.tree_structure(x0)} vs {jax.tree_structure(b)}')

    b, x0 = jax.device_put((b, x0))

    def _solve(A, b):
        return _gmres_solve(A, b, x0, tol=tol, atol=atol, maxiter=maxiter,
                            restart=restart, M=M)

    x = jax.lax.custom_linear_solve(A, b, solve=_solve, transpose_solve=_solve)
    info = None
    return x, info
