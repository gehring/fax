import jax.scipy.linalg as LA
import jax.numpy as np
from jax import jacfwd, grad, random, ops
from jax.config import config

config.update("jax_enable_x64", True)

key = random.PRNGKey(0)

# Input L as a ROW VECTOR
def f(Lambda,L,Q,q,Rw,nx):
    return -np.trace(np.matmul(Lambda,(Rw*np.matmul(L.T,L)-Q-q*np.eye(nx))))

Df_lambda = grad(f,0)
Df_L = grad(f,1)
Df_lambda_L = jacfwd(Df_lambda,1)
Df_L_lambda = jacfwd(Df_L,0)


# def opt_LQR(A,B,Q,R,s):
#     R[1,1] = s
#     R[0,0] = 1.5-s
#     X = LA.solve_discrete_are(A,B,R,Q)
#     K = -np.matmul(np.matmul(np.matmul(LA.inv(Q+np.matmul(B.T,np.matmul(X,B))),B.T), X),A)
#     return K

#Done
def inf_cost(A,B,C,Q,Ru,Rv,K,L):
    d,p = B.shape
    a,b = C.shape
    K = K.reshape((p,d))
    L = L.reshape((b,a))
    R = np.linalg.diag((Ru,Rv))
    B_t = np.hstack((B,C))
    cl_map = A + np.matmul(B,K) + np.matmul(C,L)
    if np.amax(np.abs(LA.eigvals(cl_map))) < (1.0 - 1.0e-6):
        cost = np.trace(LA.solve_discrete_are(A,B_t,Q,R))
    else:
        #cost = float("inf")
        cost = -20
    return cost


def get_g(batch_size,A,B,C,Q,Ru,Rv,K,L,T,baseline = 0):
    # mini_batch is a single gradient(log sum derivative of pi), avg of this is ordinary gradient
    # but here it is equivalent to g.
    sigma_K = 5e-1
    sigma_L = 5e-1
    sigma_x = 1e-4
    nx, nu = B.shape
    _, nw = C.shape
    K = K.reshape((nu,nx))
    L = L.reshape((nw,nx))
    Q = np.kron(np.eye(T,dtype=int), Q)
    Rv = np.kron(np.eye(T,dtype=int), Rv)
    Ru = np.kron(np.eye(T, dtype=int), Ru)

    X = np.zeros((nx*(T+1),batch_size))
    # X[0:nx,:] = 0.2 * random.normal(key, shape=(nx,batch_size))
    X = ops.index_update(X, ops.index[0:nx,:], 0.2 * random.normal(key, shape=(nx,batch_size)) )


    U = np.zeros((nu*T,batch_size))
    W = np.zeros((nw*T,batch_size))
    Vu = sigma_K * random.normal(key, shape = (nu*T, batch_size)) # noise for U
    Vw = sigma_L * random.normal(key, shape = (nw*T, batch_size)) # noise for W

    for t in range(T):
        # U[t*nu:(t+1)*nu,:] = np.matmul(K,X[nx*t:nx*(t+1),:]) + Vu[t*nu:(t+1)*nu,:]
        U = ops.index_update(U, ops.index[t*nu:(t+1)*nu,:],
                             np.matmul(K,X[nx*t:nx*(t+1),:]) + Vu[t*nu:(t+1)*nu,:])
        # W[t*nw:(t + 1) * nw, :] = np.matmul(L, X[nx * t:nx * (t + 1), :]) + Vw[t * nw:(t + 1) * nw, :]
        W = ops.index_update(W, ops.index[t*nw:(t + 1) * nw, :],
                             np.matmul(L, X[nx * t:nx * (t + 1), :]) + Vw[t * nw:(t + 1) * nw, :])
        # X[nx*(t+1):nx*(t+2),:] = np.matmul(A,X[nx*t:nx*(t+1),:]) + np.matmul(B,U[t*nu:(t+1)*nu,:]).reshape((nx,batch_size)) +\
        #                        + np.matmul(C,W[t*nw:(t+1)*nw,:]).reshape((nx,batch_size)) + sigma_x * random.normal(key, shape=(nx, batch_size))
        X = ops.index_update(X, ops.index[nx*(t+1):nx*(t+2),:],
                             np.matmul(A,X[nx*t:nx*(t+1),:]) + np.matmul(B,U[t*nu:(t+1)*nu,:]).reshape((nx,batch_size)) + np.matmul(C,W[t*nw:(t+1)*nw,:]).reshape((nx,batch_size)) + sigma_x * random.normal(key, shape=(nx, batch_size)))

    X_cost = X[nx:,:]
    reward = np.diagonal(np.matmul(X_cost.T,Q.dot(X_cost))) + np.diagonal(np.matmul(U.T,Ru.dot(U))) - np.diagonal(np.matmul(W.T,Rv.dot(W)))
    new_baseline = np.mean(reward)
    reward = reward.reshape((len(reward),1))

    #DK portion
    X_hat = X[:-nx,:] #taking only T = 0:T-1 for X for log gradient computation
    outer_grad_log_K = np.einsum("ik, jk -> ijk",Vu,X_hat) # shape (a,b,c) means there are a of the (b,c) blocks. access (b,c) blocks via C[0,:,:]
    outer_grad_log_L = np.einsum("ik, jk -> ijk", Vw, X_hat)
    sum_grad_log_K =0
    sum_grad_log_L = 0
    for t in range(T):
        sum_grad_log_K += outer_grad_log_K[nu * t:nu * (t + 1), nx * t:nx * (t + 1),:] # Summing all diagonal blocks. gives p by d by batch_size
        sum_grad_log_L += outer_grad_log_L[nw * t:nw * (t + 1), nx * t:nx * (t + 1), :]


    mini_batch_K = (1/sigma_K)**2 * ((reward-new_baseline).T*sum_grad_log_K) #mini_batch is p by d, same size as K
    mini_batch_L = (1 /sigma_L) ** 2 * ((reward - new_baseline).T * sum_grad_log_L)  # mini_batch is b by a/d, same size as K
    # mini_batch_K = 2 * ((reward-new_baseline).T*sum_grad_log_K) #mini_batch is p by d, same size as K
    # mini_batch_L =  2 * ((reward - new_baseline).T * sum_grad_log_L)  # mini_batch is b by a/d, same size as K
    # print(mini_batch_K[0,0,:])

    temp = np.einsum('mnr,ndr->mdr', sum_grad_log_K.swapaxes(0,1),sum_grad_log_L)
    batch_mixed_KL = (1/(sigma_K*sigma_L))**2 * ((reward-new_baseline).T*temp)
    # print('---new---',sum_grad_log_K[:,:,10][0,0])

    return np.mean(mini_batch_K,axis = 2),np.mean(mini_batch_L,axis = 2),np.mean(batch_mixed_KL,axis = 2),new_baseline





def gradient( num_sample,batch_size,A,B,C,Q,Ru,Rv,K,L,T,baseline = 0):
    nu,nx = K.shape
    nw,nx = L.shape
    DK_samples = np.zeros(shape = (nx,num_sample))
    DL_samples = np.zeros(shape=(nx, num_sample))
    Dxy_all = np.zeros(shape = (num_sample,nx,nx))
    for i in range(num_sample):
        g,f,mixed,baseline= get_g(batch_size,A,B,C,Q,Ru,Rv,K,L,T,baseline)
        # DK_samples[:, i] = g.flatten()
        DK_samples = ops.index_update(DK_samples, ops.index[:, i], g.flatten())
        # DL_samples[:, i] = f.flatten()
        DL_samples = ops.index_update(DL_samples, ops.index[:, i], f.flatten())
        # Dxy_all[i,:,:] = mixed
        Dxy_all = ops.index_update(Dxy_all, ops.index[i,:,:], mixed)

    return  np.mean(DK_samples,axis = 1).reshape((nu,nx)),\
            np.mean(DL_samples,axis = 1).reshape((nw,nx)),\
            np.mean(Dxy_all, axis = 0)


def proj(L,temp,lower = 0.5):
    s,v = np.linalg.eig(L.T@L)
    s = np.minimum(np.maximum(s,lower),temp)
    return np.real((v@np.diag(s))@ v.T)


def proj_sgd(L,temp):
    temp = np.sqrt(temp)
    s = np.linalg.norm(L,2)
    s = np.minimum(np.maximum(s,-temp),temp)

    return L/np.linalg.norm(L,2) * s