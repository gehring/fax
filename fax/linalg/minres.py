import functools
from typing import NamedTuple

import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental import host_callback

_add = functools.partial(jax.tree_multimap, jnp.add)
_sub = functools.partial(jax.tree_multimap, jnp.subtract)


def _scalar_mul(scalar, x):
    return jax.tree_map(lambda v: scalar * v, x)


def _tree_vdot(v, u):
    vdots = jax.tree_multimap(jnp.vdot, v, u)
    return sum(jax.tree_leaves(vdots))


def _identity(x):
    return x


def _shapes(pytree):
  return map(jnp.shape, jax.tree_leaves(pytree))


class _MINRESNorms(NamedTuple):
    A: float
    y: float
    epsa: float
    epsx: float
    epsr: float
    qr: float
    r: float


class _ResidualInfo(NamedTuple):
    epsilon: float
    delta: float
    dbar: float
    gbar: float
    root: float


class _Rotation(NamedTuple):
    cs: float
    sn: float
    gamma: float

    def next_rotation(self, beta, gbar, eps):
        gamma = jnp.sqrt(gbar ** 2 + beta ** 2)
        gamma = jnp.maximum(gamma, eps)
        cs = gbar / gamma
        sn = beta / gamma
        return _Rotation(
            cs=cs,
            sn=sn,
            gamma=gamma,
        )

    def apply(self, alpha, beta, residual_info):
        # Apply rotation to get
        #   [delta epsilon] = [cs  sn][dbar   0  ]
        #   [gbar   dbar  ]   [sn -cs][alpha beta].
        delta = self.cs * residual_info.dbar + self.sn * alpha
        gbar = self.sn * residual_info.dbar - self.cs * alpha
        epsilon = self.sn * beta
        dbar = - self.cs * beta
        root = jnp.sqrt(gbar ** 2 + dbar ** 2)
        return _ResidualInfo(epsilon, delta, dbar, gbar, root)


class _MINRESInfo(NamedTuple):
    beta: float
    prev_beta: float
    tnorm2: float
    phi: float
    phibar: float
    gmin: float
    gmax: float

    def Acond(self):
        return self.gmax / self.gmin

    def update(self,
               alpha: float,
               beta: float,
               rotation: _Rotation):
        tnorm2 = self.tnorm2 + alpha ** 2 + self.beta ** 2 + beta ** 2
        phi = rotation.cs * self.phibar
        phibar = rotation.sn * self.phibar
        gmin = jnp.minimum(self.gmin, rotation.gamma)
        gmax = jnp.maximum(self.gmax, rotation.gamma)
        return _MINRESInfo(
            beta=beta,
            prev_beta=self.beta,
            tnorm2=tnorm2,
            phi=phi,
            phibar=phibar,
            gmin=gmin,
            gmax=gmax,
        )

    def norms(self, x, eps, tol):
        # compute various norms
        Anorm = jnp.sqrt(self.tnorm2)
        ynorm = jnp.sqrt(_tree_vdot(x, x))
        epsa = Anorm * eps
        epsx = Anorm * ynorm * eps
        epsr = Anorm * ynorm * tol

        return _MINRESNorms(
            A=Anorm,
            y=ynorm,
            epsa=epsa,
            epsx=epsx,
            epsr=epsr,
            qr=self.phibar,
            r=self.phibar,
        )


def _convergence_tests(norms: _MINRESNorms, residual_info: _ResidualInfo):
    test1 = jnp.where(
        (norms.y == 0) | (norms.A == 0),
        jnp.inf,
        norms.r / (norms.A * norms.y),
    )
    test2 = jnp.where(
        norms.A == 0,
        jnp.inf,
        residual_info.root / norms.A,
    )
    return test1, test2


def _solve_status(i: int,
                  norms: _MINRESNorms,
                  residual_info: _ResidualInfo,
                  info: _MINRESInfo,
                  maxiter: int,
                  beta1: float,
                  eps: float,
                  tol: float):
    test1, test2 = _convergence_tests(norms, residual_info)
    t1 = 1 + test1
    t2 = 1 + test2

    status = 0
    status = jnp.where(t2 <= 1, 2, status)
    status = jnp.where(t1 <= 1, 1, status)

    status = jnp.where(i >= maxiter, 6, status)
    status = jnp.where(info.Acond() >= 0.1 / eps, 4, status)
    status = jnp.where(norms.epsx >= beta1, 3, status)

    status = jnp.where(test2 <= tol, 2, status)
    status = jnp.where(test1 <= tol, 1, status)

    status = jnp.where(
        (i == 1) & (info.prev_beta / beta1 <= 10 * eps),
        -1,
        status,
    )

    return status


def _minres(A, b, x0, M, maxiter, tol, shift=None, callback=None,
            status_callback=None):

    dtype = jnp.result_type(*jax.tree_leaves(x0))
    eps = jnp.finfo(dtype).eps

    r1 = _sub(b, A(x0))
    y = M(r1)

    beta1_sqr = _tree_vdot(r1, y)
    beta1 = jnp.sqrt(beta1_sqr)

    def cond(state):
        status, *_ = state
        return status == 0

    def body(state):
        _, x, i, y, r1, r2, w, w2, prev_res_info, rotation, prev_info = state

        i = i + 1
        v = _scalar_mul(jnp.reciprocal(prev_info.beta), y)
        y = A(v)
        if shift is not None:
            y = _sub(y, _scalar_mul(shift, v))

        y = jnp.where(
            i >= 2,
            _sub(y, _scalar_mul(prev_info.beta / prev_info.prev_beta, r1)),
            y,
        )

        alpha = _tree_vdot(v, y)
        y = _sub(y, _scalar_mul(alpha / prev_info.beta, r2))

        r1 = r2
        r2 = y
        y = M(r2)

        beta_sqr = _tree_vdot(r2, y)
        beta = jnp.sqrt(beta_sqr)

        res_info = rotation.apply(alpha, beta, prev_res_info)
        next_rotation = rotation.next_rotation(beta, res_info.gbar, eps)
        next_info = prev_info.update(alpha, beta, next_rotation)

        w1 = w2
        w2 = w
        w = _sub(v, _scalar_mul(prev_res_info.epsilon, w1))
        w = _sub(w, _scalar_mul(res_info.delta, w2))
        w = _scalar_mul(jnp.reciprocal(next_rotation.gamma), w)
        x = _add(x, _scalar_mul(next_info.phi, w))

        if callback is not None:
            x = host_callback.id_tap(callback, x)

        norms = next_info.norms(x, eps, tol)
        status = _solve_status(
            i, norms, res_info, next_info, maxiter, beta1, eps, tol)

        return (status, x, i, y, r1, r2, w, w2, prev_res_info, rotation,
                prev_info)

    init_res_info = _ResidualInfo(
        epsilon=0.,
        delta=0.,
        dbar=0.,
        gbar=0.,
        root=0.,
    )
    init_rot = _Rotation(
        cs=-1.,
        sn=0.,
        gamma=0.,
    )
    init_info = _MINRESInfo(
        beta=beta1,
        prev_beta=0.,
        tnorm2=0.,
        phi=0.,
        phibar=beta1,
        gmin=jnp.finfo(dtype).max,
        gmax=jnp.zeros((), dtype=dtype),
    )
    init_vals = (
        0,  # status
        x0,
        0,  # iteration count
        y,
        r1,
        r1,  # r2
        jax.tree_map(jnp.zeros_like, x0),  # w
        jax.tree_map(jnp.zeros_like, x0),  # w2
        init_res_info,
        init_rot,
        init_info,
    )
    status, x, *_ = jax.lax.while_loop(cond, body, init_vals)
    # TODO: pass down the status
    return x


def minres(A, b, x0=None, tol=1e-5, maxiter=None, M=None, callback=None):
    if x0 is None:
        x0 = jax.tree_map(jnp.zeros_like, b)

    b, x0 = jax.device_put((b, x0))

    if maxiter is None:
        size = sum(bi.size for bi in jax.tree_leaves(b))
        maxiter = 5 * size  # copied from scip

    if M is None:
        M = _identity

    if jax.tree_structure(x0) != jax.tree_structure(b):
        raise ValueError(
            'x0 and b must have matching tree structure: '
            f'{jax.tree_structure(x0)} vs {jax.tree_structure(b)}')

    if _shapes(x0) != _shapes(b):
        raise ValueError(
            'arrays in x0 and b must have matching shapes: '
            f'{_shapes(x0)} vs {_shapes(b)}')

    def minres_solve(A, b):
        return _minres(A, b, x0=x0, tol=tol, maxiter=maxiter, M=M,
                       callback=callback)

    # real-valued positive-definite linear operators are symmetric
    real_valued = lambda x: not issubclass(x.dtype.type, np.complexfloating)
    symmetric = all(map(real_valued, jax.tree_leaves(b)))
    x = jax.lax.custom_linear_solve(A, b,
                                    solve=minres_solve,
                                    transpose_solve=minres_solve,
                                    symmetric=symmetric)
    info = None
    return x, info
