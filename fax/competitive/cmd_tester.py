import jax.numpy as np
from jax import random, grad, jacfwd
# from jax.scipy import linalg
from cmd_helper import DP_pd, DP_inv_pd, D2P_pd, inv_D2P_pd
from cmd import make_lagrangian, updates, cmd_step
import collections
from lq_game_helper import proj, gradient, Df_lambda, Df_L, Df_lambda_L, Df_L_lambda
import jax.ops
import pickle

BregmanPotential = collections.namedtuple("BregmanPotential", ["DP", "DP_inv", "D2P", "inv_D2P"])
CMDState = collections.namedtuple("CMDState", "minPlayer maxPlayer minPlayer_dual maxPlayer_dual")
UpdateState = collections.namedtuple("UpdateState", "del_min del_max")

key1 = random.PRNGKey(0)
key = random.PRNGKey(1)
x1 = np.array([1., 2., 3., 4., 5.])
x2 = random.normal(key1, (5, ))
W1 = random.normal(key, (3, 3))
W2 = random.normal(key1, (3, 3))

x = [(x1, x2), (x1, x1, x2)]
W = [(W1, W2), (W1, W1, W2)]


# CMD main algorithm test with single variables, scalar example from CMD paper
"""
def obj_func(x, y):
    return 2 * x * y - (1 - y) ** 2

breg_min = BregmanPotential(DP_pd, DP_inv_pd, D2P_pd, inv_D2P_pd)
breg_max = BregmanPotential(DP_pd, DP_inv_pd, D2P_pd, inv_D2P_pd)
# initialize states
x_init = -random.normal(key1, (1, ))
y_init = -random.normal(key, (1, ))
x_init_dual = DP_pd(x_init)
y_init_dual = DP_pd(y_init)
prev_state = CMDState(x_init, y_init, x_init_dual, y_init_dual)

# Compute hessians and gradients
grad_min = jacfwd(obj_func,0)
grad_max = jacfwd(obj_func,1)
H_xy = jacfwd(grad_min,1)
H_yx =jacfwd(grad_max,0)

J_min = grad_min(prev_state.minPlayer, prev_state.maxPlayer).flatten()
J_max = grad_max(prev_state.minPlayer, prev_state.maxPlayer).flatten()
hessian_xy = lambda v: H_xy(prev_state.minPlayer, prev_state.maxPlayer).flatten()*v
hessian_yx = lambda v: H_yx(prev_state.minPlayer, prev_state.maxPlayer).flatten()*v

# Main CMD algorithm
for t in range(1000):
    delta = updates(prev_state, 0.001, 0.001, hessian_xy, hessian_yx, J_min, J_max, breg_min, breg_max)
    print(prev_state)
    new_state = cmd_step(prev_state, delta, breg_min, breg_max)
    print(new_state)
    prev_state = new_state

    J_min = grad_min(prev_state.minPlayer, prev_state.maxPlayer).flatten()
    J_max = grad_max(prev_state.minPlayer, prev_state.maxPlayer).flatten()
    hessian_xy = lambda v: H_xy(prev_state.minPlayer, prev_state.maxPlayer).flatten() * v
    hessian_yx = lambda v: H_yx(prev_state.minPlayer, prev_state.maxPlayer).flatten() * v

print(prev_state)
print(new_state)
"""


# CMD main algorithm test with single variables, vector example from CMD paper
"""
n = 3
A = random.normal(key, (n, n))
b = (A[:,0] + A[:,1]) / 2 + 0.01

def obj_func(x, y):
    return np.linalg.norm(np.dot(A,x)-b) ** 2 + y * (np.dot(np.ones_like(b), x) - 1)

breg_min = BregmanPotential(DP_pd, DP_inv_pd, D2P_pd, inv_D2P_pd)

# initialize states
x_init = random.normal(key1, (n, ))
y_init = random.normal(key, (1, ))
x_init = jax.ops.index_update(x_init, x_init<0.01,0.01)
x_init_dual = DP_pd(x_init)
y_init_dual = y_init
prev_state = CMDState(x_init, y_init, x_init_dual, y_init_dual)

# Compute hessians and gradients
grad_min = jacfwd(obj_func,0)
grad_max = jacfwd(obj_func,1)
H_xy = jacfwd(grad_min,1)
H_yx =jacfwd(grad_max,0)

J_min = grad_min(prev_state.minPlayer, prev_state.maxPlayer).reshape(n,)
J_max = grad_max(prev_state.minPlayer, prev_state.maxPlayer).reshape(1,)
hessian_xy = lambda v: np.dot(H_xy(prev_state.minPlayer, prev_state.maxPlayer).reshape(n,1), v)
hessian_yx = lambda v: np.dot(H_yx(prev_state.minPlayer, prev_state.maxPlayer).reshape(1,n), v)

# Main CMD algorithm
for t in range(1000):
    delta = updates(prev_state, 0.01, 0.1, hessian_xy, hessian_yx, J_min, J_max, breg_min)
    print(prev_state)
    new_state = cmd_step(prev_state, delta, breg_min)
    print(new_state)
    prev_state = new_state

    J_min = grad_min(prev_state.minPlayer, prev_state.maxPlayer).reshape(n,)
    J_max = grad_max(prev_state.minPlayer, prev_state.maxPlayer).reshape(1,)
    hessian_xy = lambda v: np.dot(H_xy(prev_state.minPlayer, prev_state.maxPlayer).reshape(n, 1), v)
    hessian_yx = lambda v: np.dot(H_yx(prev_state.minPlayer, prev_state.maxPlayer).reshape(1, n), v)

print(prev_state)
print(new_state)
"""


# CMD main algorithm test with structured variables, vector example from RRL paper
# using the sampling functions from RRL repository.
# The min player has two variables, K and Lambda. The max player has a single variable L.

# Problem Parameters
A = np.array([[1,1],[0,1]])
B = np.array([[0],[1]])
C = np.array([[0.5],[1]])
nx,nu = B.shape
_,nw = C.shape
T = 15
Ru = np.eye(nu)
Rw = 20*np.eye(nw)
Q = np.eye(nx)
q = 0.01
e,_ = np.linalg.eig(Q)
l_max = (np.min(e) - q) / Rw
safeguard = 2


# Bregmen Potential definitions
def D2P_l2(v):
    return lambda x: x
breg_min = BregmanPotential([lambda x: x, DP_pd], [lambda x: x, DP_inv_pd], [D2P_l2, D2P_pd], [D2P_l2, inv_D2P_pd])

# Initialization of variables
K = 0.01 * random.normal(key1, (nu, nx))
Lambda = 0.01 * random.normal(key1, (nx, nx))
Lambda = np.eye(nx)+ Lambda + Lambda.T
Lambda = proj(Lambda,2)

x = [K, Lambda] # min player
y = 0.01 * random.normal(key, (nu, nx)) # max player L
dual_x = [K, DP_pd(Lambda)]
dual_y = y
prev_state = CMDState(x, y, dual_x, dual_y)

# Get gradients and hessians
DK,DL,DKL = gradient(50,100,A,B,C,Q,Ru,Rw,prev_state.minPlayer[0],y,T)
DfLambda = Df_lambda(prev_state.minPlayer[1],y,Q,q,Rw,nx)
DfL = Df_L(prev_state.minPlayer[1],y,Q,q,Rw,nx)
DfLambdaL = Df_lambda_L(prev_state.minPlayer[1],y,Q,q,Rw,nx)
DfLLambda = Df_L_lambda(prev_state.minPlayer[1],y,Q,q,Rw,nx)

J_max = DL + DfL
J_min = [DK, DfLambda]
# hessian_xy = lambda v: [np.matmul(DKL, v.T).T, np.tensordot(DfLambdaL, DL)]
def hessian_xy_generator(DKL,DfLambdaL):
    def hessian_xy(max_tree):
        return [np.matmul(DKL, max_tree.T).T, np.tensordot(DfLambdaL, max_tree)] # returns minPlayer structure
    return hessian_xy

# hessian_yx = lambda K_var,Lambda_var : np.matmul(DKL.T,K_var.T).T + np.tensordot(DfLLambda,Lambda_var)
def hessian_yx_generator(DKL,DfLLambda):
    def hessian_yx(min_tree):
        K_var = min_tree[0]
        Lambda_var = min_tree[1]
        return np.matmul(DKL.T,K_var.T).T + np.tensordot(DfLLambda,Lambda_var) # returns maxPlayer structure
    return hessian_yx

# Main CMD algorithm
state_list = []
state_list.append(prev_state)
minPlayer_list_1 = [prev_state.minPlayer[0][0][0]]
minPlayer_list_2 = [prev_state.minPlayer[0][0][1]]
maxPlayer_list_1 = [prev_state.maxPlayer[0][0]]
maxPlayer_list_2 = [prev_state.maxPlayer[0][1]]


for t in range(2000):
    delta = updates(prev_state, 2e-5, 2e-5, hessian_xy_generator(DKL,DfLambdaL), hessian_yx_generator(DKL,DfLLambda), J_min, J_max, breg_min)
    new_state = cmd_step(prev_state, delta, breg_min)

    # Saving data
    state_list.append(new_state)
    minPlayer_list_1.append(new_state.minPlayer[0][0][0])
    minPlayer_list_2.append(new_state.minPlayer[0][0][1])
    maxPlayer_list_1.append( new_state.maxPlayer[0][0])
    maxPlayer_list_2.append(new_state.maxPlayer[0][1])

    if t%20 ==0:
        print("-------------------",t,"---------------")
        print(prev_state)
        print(new_state)
        np.save("minPlayer_list_1",minPlayer_list_1)
        np.save("minPlayer_list_2",minPlayer_list_2)
        np.save("maxPlayer_list_1",maxPlayer_list_1)
        np.save("maxPlayer_list_2",maxPlayer_list_2)

        with open('state_list.pkl', 'wb') as f:
            pickle.dump(state_list, f)


    prev_state = new_state

    DK, DL, DKL = gradient(50, 200, A, B, C, Q, Ru, Rw, prev_state.minPlayer[0], y, T)
    DfLambda = Df_lambda(prev_state.minPlayer[1], y, Q, q, Rw, nx)
    DfL = Df_L(prev_state.minPlayer[1], y, Q, q, Rw, nx)
    DfLambdaL = Df_lambda_L(prev_state.minPlayer[1], y, Q, q, Rw, nx)
    DfLLambda = Df_L_lambda(prev_state.minPlayer[1], y, Q, q, Rw, nx)
    J_max = DL + DfL
    J_min = [DK, DfLambda]




print(prev_state)
print(new_state)



# Test lagrangian making portion, skip for now
""""
def obj_func(x, y):
    return 2 * x * y - (1 - y) ** 2


breg_min = BregmanPotential(DP_pd, DP_inv_pd, D2P_pd, inv_D2P_pd)
breg_max = BregmanPotential(DP_pd, DP_inv_pd, D2P_pd, inv_D2P_pd)

lagrangian, breg_min_aug, breg_max_aug, init_multipliers = make_lagrangian(obj_func, breg_min, breg_max)
print(lagrangian(1., 2., obj_func))
"""

