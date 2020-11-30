import jax.numpy as np
from jax import random, grad, jacfwd
import jaxlib
from jax.scipy import linalg
from cmd_helper import DP_pd, DP_inv_pd, D2P_pd, inv_D2P_pd
from cmd import make_lagrangian, updates, cmd_step, _tree_apply, make_bound_breg
import collections
from lq_game_helper import proj, gradient, Df_lambda, Df_L, Df_lambda_L, Df_L_lambda
import jax.ops
import pickle
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
print(jax.__version__)
print(jaxlib.__version__)
from jax.config import config
config.update("jax_enable_x64", True)

BregmanPotential = collections.namedtuple("BregmanPotential", ["DP", "DP_inv", "D2P", "inv_D2P"])
CMDState = collections.namedtuple("CMDState", "minPlayer maxPlayer minPlayer_dual maxPlayer_dual")
UpdateState = collections.namedtuple("UpdateState", "del_min del_max")

key1 = random.PRNGKey(0)
key = random.PRNGKey(1)
x1 = jnp.array([1., 2., 3., 4., 5.])
x2 = random.normal(key1, (5, ))
W1 = random.normal(key, (3, 3))
W2 = random.normal(key1, (3, 3))

x = [(x1, x2), (x1, x1, x2)]
W = [(W1, W2), (W1, W1, W2)]


DP_inv_eq_min = lambda v: jax.tree_map(lambda x: x, v)
DP_inv_ineq_min = lambda v: jax.tree_map(DP_inv_pd, v)

min_augmented_DP = (lambda x: x, lambda v: jax.tree_map(DP_pd, v))  # [breg_min.DP, DP_eq_min, DP_ineq_min]
min_augmented_DP_inv = (DP_inv_eq_min, DP_inv_ineq_min)  # [breg_min.DP_inv, DP_inv_eq_min, DP_inv_ineq_min]

D2P_eq_min = lambda v: jax.tree_map(id_func, v)
D2P_ineq_min = lambda v: jax.tree_map(D2P_pd, v)
min_augmented_D2P = ( D2P_eq_min, D2P_ineq_min)

# inv_D2P_eq_min = lambda v: jax.tree_map(lambda x: x, v)
inv_D2P_eq_min = lambda v: jax.tree_map(id_func, v)
inv_D2P_ineq_min = lambda v: jax.tree_map(inv_D2P_pd, v)
min_augmented_D2P_inv = (inv_D2P_eq_min, inv_D2P_ineq_min)


key1 = random.PRNGKey(0)
key = random.PRNGKey(1)
x1 = jnp.array([1., 2., 3.,4., 5.])
x2 = random.normal(key1, (5,))
W1 = random.normal(key, (3,3))
W2 = random.normal(key1, (3,3))

x = ((x1,x2), (x1,x1,x2))
W = ((W1,W2), (W1,W1,W2))

print(jax.tree_multimap(lambda f, x: f(x), min_augmented_DP, x))
print(DP_pd(x1))
print(DP_pd(x2))


# Check if the inv(D2P) match the closed form.
print(inv_D2P_pd(W1)(jnp.identity(W1.shape[0])))
print(jnp.linalg.matrix_power(W1,2).T)

print(inv_D2P_pd(x2)(x1))
print(jnp.dot(jnp.diag(x2),x1))


# cmd main algorithm test with single variables, scalar example from cmd paper
""""
def obj_func(x, y):
    return 2 * x * y - (1 - y) ** 2

breg_min = pd_bregman()
breg_max = pd_bregman()
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

gradient_min = grad_min(prev_state.minPlayer, prev_state.maxPlayer).flatten()
gradient_max = grad_max(prev_state.minPlayer, prev_state.maxPlayer).flatten()
hessian_xy = lambda v: H_xy(prev_state.minPlayer, prev_state.maxPlayer).flatten()*v
hessian_yx = lambda v: H_yx(prev_state.minPlayer, prev_state.maxPlayer).flatten()*v

# Main cmd algorithm
for t in range(1000):
    delta = updates(prev_state, 0.001, 0.001, hessian_xy, hessian_yx, gradient_min, gradient_max, breg_min, breg_max)
    print(prev_state)
    new_state = cmd_step(prev_state, delta, breg_min, breg_max)
    print(new_state)
    prev_state = new_state

    gradient_min = grad_min(prev_state.minPlayer, prev_state.maxPlayer).flatten()
    gradient_max = grad_max(prev_state.minPlayer, prev_state.maxPlayer).flatten()
    hessian_xy = lambda v: H_xy(prev_state.minPlayer, prev_state.maxPlayer).flatten() * v
    hessian_yx = lambda v: H_yx(prev_state.minPlayer, prev_state.maxPlayer).flatten() * v

print(prev_state)
print(new_state)
"""


# cmd main algorithm test with single variables, vector example from cmd paper
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

grad_min = grad_min(prev_state.minPlayer, prev_state.maxPlayer).reshape(n,)
grad_max = grad_max(prev_state.minPlayer, prev_state.maxPlayer).reshape(1,)
hessian_xy = lambda v: np.dot(H_xy(prev_state.minPlayer, prev_state.maxPlayer).reshape(n,1), v)
hessian_yx = lambda v: np.dot(H_yx(prev_state.minPlayer, prev_state.maxPlayer).reshape(1,n), v)

# Main cmd algorithm
for t in range(1000):
    delta = updates(prev_state, 0.01, 0.1, hessian_xy, hessian_yx, grad_min, grad_max, breg_min)
    print(prev_state)
    new_state = cmd_step(prev_state, delta, breg_min)
    print(new_state)
    prev_state = new_state

    grad_min = grad_min(prev_state.minPlayer, prev_state.maxPlayer).reshape(n,)
    grad_max = grad_max(prev_state.minPlayer, prev_state.maxPlayer).reshape(1,)
    hessian_xy = lambda v: np.dot(H_xy(prev_state.minPlayer, prev_state.maxPlayer).reshape(n, 1), v)
    hessian_yx = lambda v: np.dot(H_yx(prev_state.minPlayer, prev_state.maxPlayer).reshape(1, n), v)

print(prev_state)
print(new_state)
"""


# cmd main algorithm test with structured variables, vector example from RRL paper
# using the sampling functions from RRL repository.
# The min player has two variables, K and Lambda. The max player has a single variable L.

# Problem Parameters
A = jnp.array([[1,1],[0,1]])
B = jnp.array([[0],[1]])
C = jnp.array([[0.5],[1]])
nx,nu = B.shape
_,nw = C.shape
T = 15
Ru = jnp.eye(nu)
Rw = 20*jnp.eye(nw)
Q = jnp.eye(nx)
q = 0.01
e,_ = jnp.linalg.eig(Q)
l_max = (jnp.min(e) - q) / Rw
safeguard = 2


# Bregmen Potential definitions
def D2P_l2(v):
    return lambda x: x
breg_min = BregmanPotential((lambda x: x, DP_pd), (lambda x: x, DP_inv_pd), (D2P_l2, D2P_pd), (D2P_l2, inv_D2P_pd))

# Initialization of variables
K = 0.001 * random.normal(key1, (nu, nx))
Lambda = 0.001 * random.normal(key1, (nx, nx))
Lambda = jnp.eye(nx)+ Lambda + Lambda.T
Lambda = proj(Lambda,2)

x = (K, Lambda) # min player
y = 0.01 * random.normal(key, (nu, nx)) # max player L
dual_x = (K, DP_pd(Lambda))
dual_y = y
prev_state = CMDState(x, y, dual_x, dual_y)

# # Get gradients and hessians
# print("-- about to compute gradients --")
# DK,DL,DKL = gradient(50,100,A,B,C,Q,Ru,Rw,prev_state.minPlayer[0],y,T)
# print("--done with gradient computation--")
# DfLambda = Df_lambda(prev_state.minPlayer[1],y,Q,q,Rw,nx)
# DfL = Df_L(prev_state.minPlayer[1],y,Q,q,Rw,nx)
# DfLambdaL = Df_lambda_L(prev_state.minPlayer[1],y,Q,q,Rw,nx)
# DfLLambda = Df_L_lambda(prev_state.minPlayer[1],y,Q,q,Rw,nx)
#
# grad_max = DL + DfL
# grad_min = (DK, DfLambda)
# # hessian_xy = lambda v: [np.matmul(DKL, v.T).T, np.tensordot(DfLambdaL, DL)]
def hessian_xy_generator(DKL,DfLambdaL):
    def hessian_xy(max_tree):
        return (np.matmul(DKL, max_tree.T).T, np.tensordot(DfLambdaL, max_tree)) # returns minPlayer structure
    return hessian_xy

# hessian_yx = lambda K_var,Lambda_var : np.matmul(DKL.T,K_var.T).T + np.tensordot(DfLLambda,Lambda_var)
def hessian_yx_generator(DKL,DfLLambda):
    def hessian_yx(min_tree):
        K_var = min_tree[0]
        Lambda_var = min_tree[1]
        return np.matmul(DKL.T,K_var.T).T + np.tensordot(DfLLambda,Lambda_var) # returns maxPlayer structure
    return hessian_yx

# Main cmd algorithm
state_list = []
state_list.append(prev_state)
minPlayer_list_1 = [prev_state.minPlayer[0][0][0]]
minPlayer_list_2 = [prev_state.minPlayer[0][0][1]]
maxPlayer_list_1 = [prev_state.maxPlayer[0][0]]
maxPlayer_list_2 = [prev_state.maxPlayer[0][1]]

# @jit
# def jit_cmd(prev_state):
#     DK, DL, DKL= gradient(50, 100, A, B, C, Q, Ru, Rw, prev_state.minPlayer[0], prev_state.maxPlayer, T)
#     DfLambda = Df_lambda(prev_state.minPlayer[1], prev_state.maxPlayer, Q, q, Rw, nx)
#     DfL = Df_L(prev_state.minPlayer[1], prev_state.maxPlayer, Q, q, Rw, nx)
#     DfLambdaL = Df_lambda_L(prev_state.minPlayer[1], prev_state.maxPlayer, Q, q, Rw, nx)
#     DfLLambda = Df_L_lambda(prev_state.minPlayer[1], prev_state.maxPlayer, Q, q, Rw, nx)
#     grad_max = DL + DfL
#     grad_min = (DK, DfLambda)
#
#     delta = updates(prev_state, 2e-4, 4e-3, hessian_xy_generator(DKL, DfLambdaL),
#             hessian_yx_generator(DKL, DfLLambda), grad_min, grad_max, breg_min)
#     return cmd_step(prev_state, delta, breg_min)

eta_x = 1e-4
eta_y = 1e-3

# @jit
def jit_cmd(prev_state, DK, DL, DKL):
    def hessian_xy_generator(DKL, DfLambdaL):
        def hessian_xy(max_tree):
            # print('hessian_xy called!')
            return (np.matmul(DKL, max_tree.T).T,
                    np.tensordot(DfLambdaL, max_tree))  # returns minPlayer structure

        return hessian_xy

    # hessian_yx = lambda K_var,Lambda_var : np.matmul(DKL.T,K_var.T).T + np.tensordot(DfLLambda,Lambda_var)
    def hessian_yx_generator(DKL, DfLLambda):
        def hessian_yx(min_tree):
            K_var = min_tree[0]
            Lambda_var = min_tree[1]
            # print('hessian_yx called!')
            return np.matmul(DKL.T, K_var.T).T + np.tensordot(DfLLambda,
                                                              Lambda_var)  # returns maxPlayer structure

        return hessian_yx

    DfLambda = Df_lambda(prev_state.minPlayer[1], prev_state.maxPlayer, Q, q, Rw, nx)
    DfL = Df_L(prev_state.minPlayer[1], prev_state.maxPlayer, Q, q, Rw, nx)
    DfLambdaL = Df_lambda_L(prev_state.minPlayer[1], prev_state.maxPlayer, Q, q, Rw, nx)
    DfLLambda = Df_L_lambda(prev_state.minPlayer[1], prev_state.maxPlayer, Q, q, Rw, nx)
    grad_max = DL + DfL
    grad_min = (DK, DfLambda)

    delta = updates(prev_state, eta_x, eta_y, hessian_xy_generator(DKL, DfLambdaL),
            hessian_yx_generator(DKL, DfLLambda), grad_min, grad_max, breg_min=breg_min,
                    precond_b_min=False, precond_b_max=False)
    return cmd_step(prev_state, delta, breg_min=breg_min), delta, grad_min
# ----------------------------------------------

infile = open('state_list.pkl','rb')
state_list = pickle.load(infile)
infile.close()
prev_state = state_list[-1]
prev_state = jax.tree_map(lambda x:jnp.float64(x), prev_state)

print('------brgin-------- starting with: ', prev_state)
for t in range(3000):
    # print("-- about to compute gradients --")
    # DK, DL, DKL = jit_gradient(prev_state) #gradient(50, 100, A, B, C, Q, Ru, Rw, prev_state.minPlayer[0], y, T)
    # print("--done with gradient computation--")
    # DfLambda = Df_lambda(prev_state.minPlayer[1], prev_state.maxPlayer, Q, q, Rw, nx)
    # DfL = Df_L(prev_state.minPlayer[1], prev_state.maxPlayer, Q, q, Rw, nx)
    # DfLambdaL = Df_lambda_L(prev_state.minPlayer[1], prev_state.maxPlayer, Q, q, Rw, nx)
    # DfLLambda = Df_L_lambda(prev_state.minPlayer[1], prev_state.maxPlayer, Q, q, Rw, nx)
    # grad_max = DL + DfL
    # grad_min = (DK, DfLambda)
    # delta = jit_updates(prev_state, DKL, DfLambdaL, DfLLambda, grad_min, grad_max) #updates(prev_state, 2e-5, 2e-5, hessian_xy_generator(DKL,DfLambdaL), hessian_yx_generator(DKL,DfLLambda), grad_min, grad_max, breg_min)


    DK, DL, DKL= gradient(50, 200, A, B, C, Q, Ru, Rw, prev_state.minPlayer[0], prev_state.maxPlayer, T)
    new_state,del_, grad_min = jit_cmd(prev_state, DK, DL, DKL)



    # Saving data
    state_list.append(new_state)
    minPlayer_list_1.append(new_state.minPlayer[0][0][0])
    minPlayer_list_2.append(new_state.minPlayer[0][0][1])
    maxPlayer_list_1.append( new_state.maxPlayer[0][0])
    maxPlayer_list_2.append(new_state.maxPlayer[0][1])

    if t%1 ==0:
        print("-------------------",t,"---------------")
        print("K ",new_state.minPlayer[0])
        # print('grad_min ', grad_min)
        print('delta_min ', del_.del_min)
        print("L ", new_state.maxPlayer)
        np.save("minPlayer_list_1",minPlayer_list_1)
        np.save("minPlayer_list_2",minPlayer_list_2)
        np.save("maxPlayer_list_1",maxPlayer_list_1)
        np.save("maxPlayer_list_2",maxPlayer_list_2)

        with open('state_list.pkl', 'wb') as f:
            pickle.dump(state_list, f)


    prev_state = new_state



print(new_state)



p1 = plt.figure(1)
plt.subplot(121)
plt.plot(minPlayer_list_1,label = 'CMD')
plt.legend()

plt.subplot(122)
plt.plot(minPlayer_list_2,label = 'CMD')
plt.legend()
plt.title('K')

p2 = plt.figure(2)
plt.subplot(121)
plt.plot(maxPlayer_list_1, label = 'CMD')
plt.legend()


plt.subplot(122)
plt.plot(maxPlayer_list_2,label = 'CMD')
plt.title('L')
plt.legend()
plt.show()

# # Test lagrangian making portion, works!
# def obj_func(x, y):
#     return (2 * x * y - (1 - y) ** 2)[0]
#
#
# breg_min = BregmanPotential(DP_pd, DP_inv_pd, D2P_pd, inv_D2P_pd)
# breg_max = BregmanPotential(DP_pd, DP_inv_pd, D2P_pd, inv_D2P_pd)
#
# init_multipliers, lagrangian, breg_min_aug, breg_max_aug = make_lagrangian(obj_func, breg_min, breg_max)
# min_P, max_P = init_multipliers(np.array([1.,]),np.array([2.,]))
# dual_min = _tree_apply(breg_min_aug.DP,min_P)
# dual_max = _tree_apply(breg_max_aug.DP,max_P)
# prev_state = CMDState(min_P,max_P, dual_min, dual_max)
# updates(prev_state,1e-4, 1e-4, breg_min=breg_min_aug, breg_max = breg_max_aug, objective_func=lagrangian)



"""
horizon = 10                              # how many unit time we simulate for
num_control_intervals = 20                # how many intervals of control
step_size = horizon/num_control_intervals # how long to hold each control value

control_bounds = np.empty((num_control_intervals, 2))
control_bounds[:] = [-0.75, 1.0]
# (^ this can stay an onp array)

x0 = jnp.array([0., 1.]) # start state
xf = jnp.array([0., 0.]) # end state

# Dynamics function
@jit
def f(x, u):
  x0 = x[0]
  x1 = x[1]
  return jnp.asarray([(1. - x1**2) * x0 - x1 + u, x0])

# Instantaneous cost
@jit
def c(x, u):
  return jnp.dot(x, x) + u**2

vector_c = jit(vmap(c))

# Integrate from the start state, using controls, to the final state
@jit
def integrate_fwd(us):
  def rk4_step(x, u):
    k1 = f(x, u)
    k2 = f(x + step_size * k1/2, u)
    k3 = f(x + step_size * k2/2, u)
    k4 = f(x + step_size * k3  , u)
    return x + (step_size/6)*(k1 + 2*k2 + 2*k3 + k4)

  def fn(carried_state, u):
    one_step_forward = rk4_step(carried_state, u)
    return one_step_forward, one_step_forward # (carry, y)

  last_state_and_all_xs = lax.scan(fn, x0, us)
  return last_state_and_all_xs

# Calculate cost over entire trajectory
@jit
def objective(us):
  _, xs = integrate_fwd(us)
  all_costs = vector_c(xs, us)
  return jnp.sum(all_costs) + jnp.dot(x0, x0) # add in cost of start state (will make no difference)

# Calculate defect of final state
@jit
def equality_constraints(us):
  final_state, _ = integrate_fwd(us)
  return final_state - xf

rng = jax.random.PRNGKey(42)
# rng, rng_input = jax.random.split(rng)
initial_controls_guess = jax.random.uniform(rng, shape=(num_control_intervals,), minval=-0.76, maxval=0.9)

constraints = ({'type': 'eq',
                'fun': equality_constraints,
                'jac': jax.jit(jax.jacrev(equality_constraints))
                })

options = {'maxiter': 500, 'ftol': 1e-6}




# Make Lagrangian out of the original OCP
key = jax.random.PRNGKey(1)

# Generate Lagrangian-related functions and augmented Bregman divergence
init_multipliers, lagrangian, breg_min_aug, breg_max_aug = make_lagrangian(objective, breg_min = make_bound_breg(lb=-0.75, ub=1.0), min_equality_constraints=equality_constraints)

# Initialize the augmented min player and max player
min_P, max_P = init_multipliers(initial_controls_guess,None,key)
dual_min = _tree_apply(breg_min_aug.DP,min_P)
dual_max = _tree_apply(breg_max_aug.DP,max_P)

# Construct a CMD state
init_state = CMDState(min_P, max_P, dual_min, dual_max )


# Testing tge Lagrangian funciton and the Bregman potentials
L = lagrangian(min_P, max_P) # L =  objective(initial_controls_guess) + max_P[1] @ equality_constraints(initial_controls_guess)

_tree_apply( breg_max_aug.DP_inv,max_P)
_tree_apply(_tree_apply( breg_max_aug.D2P,max_P),max_P)
_tree_apply(_tree_apply( breg_max_aug.inv_D2P,max_P),max_P)

_tree_apply( breg_min_aug.DP_inv,min_P)
_tree_apply(_tree_apply( breg_min_aug.D2P,min_P),min_P)
_tree_apply(_tree_apply( breg_min_aug.inv_D2P,min_P),min_P)

prev_state = init_state
for i in range(200):
    delta = updates(prev_state,1e-3, 1e-3, breg_min=breg_min_aug, breg_max = breg_max_aug, objective_func=lagrangian)
    new_state = cmd_step(prev_state, delta, breg_min_aug, breg_max_aug)
    prev_state = new_state
    if i%1 ==0:
        print("---------------",i,"------------")
        print(lagrangian(new_state.minPlayer, new_state.maxPlayer))
print(new_state.minPlayer)
"""
