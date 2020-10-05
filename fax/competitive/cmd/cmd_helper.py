import jax.numpy as np
import jax
from jax import grad, jvp
from jax.scipy import linalg
from jax import random
from functools import partial


# DP helper functions
def DP_hand(vec, nx):
    temp = np.reshape(vec, (nx, nx))
    return (-np.linalg.inv(temp).T + temp).reshape(nx**2, 1)


# matrix PD Potential: P(M) = -logdet(M) + 1/2*norm(M)^2
# Vector PD Potential: P(x) = xlog(x)
def matrix_DP_pd(M):
    return -np.linalg.slogdet(M)[1]


def vector_DP_pd(v):
    return np.dot(v, np.log(v))


def DP_pd(v):
    m = len(np.shape(v))
    if m == 1:
        out = grad(lambda x: np.dot(x, np.log(x)))(v)
    else:
        out = grad(lambda M: -np.linalg.slogdet(M)[1])(v)
    return out


# DP_inv helper functions

def vector_DP_inv_pd(v):
    return np.exp(v - np.ones_like(v))


def DP_inv_pd(v):
    m = len(np.shape(v))
    if m == 1:
        out = vector_DP_inv_pd(v)
    else:
        out = -linalg.inv(v).T
    return out

# def matrix_DP_inv_pd(Y):
#     # Y = (Y+Y.T)/2
#     nx,_ = np.shape(Y)
#     s,U = linalg.eigh(Y)
#     s_x = np.empty_like(s)
#     for i in range(len(s)):
#         y = float(np.real(np.roots([1,-s[i],-1])[np.roots([1,-s[i],-1])>0]))
#         print(y)
#         s_x = index_update(s_x, i, y)
#         print(s_x)
#     TT = (U @ np.diag(s_x) @ linalg.inv(U))
#     b, _ = np.linalg.eig(TT.reshape((nx, nx)))
#     print(b)
#     return TT


# D2P helper functions
def id_func(x):
    return lambda u: np.dot(np.identity(x.shape[0]), u)


def hvp(f, primals, tangents):
    return jvp(grad(f), primals, tangents)[1]


def D2P_pd(v):
    m = len(np.shape(v))
    if m == 1:
        def out(u):
            return hvp(vector_DP_pd, (v,), (u,))
    else:
        def out(u):
            return hvp(matrix_DP_pd, (v,), (u,))
    return out


# inv_D2P helper functions

def inv_D2P_pd(v):
    m = len(np.shape(v))
    if m == 1:
        def out(u):
            return np.dot(np.diag(v), u)
    else:
        def out(u):
            return np.dot(np.linalg.matrix_power(v, 2).T, u)
    return out

# Testing #
#
# DP_inv_eq_min = lambda v: jax.tree_map(lambda x: x, v)
# DP_inv_ineq_min = lambda v: jax.tree_map(DP_inv_pd, v)
#
# min_augmented_DP = (lambda x: x, lambda v: jax.tree_map(DP_pd, v))  # [breg_min.DP, DP_eq_min, DP_ineq_min]
# min_augmented_DP_inv = (DP_inv_eq_min, DP_inv_ineq_min)  # [breg_min.DP_inv, DP_inv_eq_min, DP_inv_ineq_min]
#
# D2P_eq_min = lambda v: jax.tree_map(id_func, v)
# D2P_ineq_min = lambda v: jax.tree_map(D2P_pd, v)
# min_augmented_D2P = ( D2P_eq_min, D2P_ineq_min)
#
# # inv_D2P_eq_min = lambda v: jax.tree_map(lambda x: x, v)
# inv_D2P_eq_min = lambda v: jax.tree_map(id_func, v)
# inv_D2P_ineq_min = lambda v: jax.tree_map(inv_D2P_pd, v)
# min_augmented_D2P_inv = (inv_D2P_eq_min, inv_D2P_ineq_min)
#
#
# key1 = random.PRNGKey(0)
# key = random.PRNGKey(1)
# x1 = np.array([1., 2., 3.,4., 5.])
# x2 = random.normal(key1, (5,))
# W1 = random.normal(key, (3,3))
# W2 = random.normal(key1, (3,3))
#
# x = ((x1,x2), (x1,x1,x2))
# W = ((W1,W2), (W1,W1,W2))
#
# print(jax.tree_multimap(lambda f, x: f(x), min_augmented_DP, x))
# print(DP_pd(x1))
# print(DP_pd(x2))
#
#
# Check if the inv(D2P) match the closed form.
# print(inv_D2P_pd(W1)(np.identity(W1.shape[0])))
# print(np.linalg.matrix_power(W1,2).T)
#
# print(inv_D2P_pd(x2)(x1))
# print(np.dot(np.diag(x2),x1))
