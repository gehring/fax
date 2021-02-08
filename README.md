# fax: fixed-point jax 

Implicit and [competitive differentiation](https://optrl2019.github.io/assets/accepted_papers/70.pdf) in [JAX](https://github.com/google/jax).

Our "competitive differentiation" approach uses [Competitive Gradient Descent](https://arxiv.org/abs/1905.12103) to solve the equality-constrained nonlinear program associated with the fixed-point problem. A standalone implementation of CGD is provided under [fax/competitive/cga.py](fax/competitive/cga.py) and the equality-constrained solver derived from it can be accessed via `fax.constrained.cga_lagrange_min` or `fax.constrained.cga_ecp`. An implementation of implicit differentiation based on [Christianson's](https://doi.org/10.1080/10556789408805572) two-phases reverse accumulation algorithm can also be obtained with the function `fax.implicit.two_phase_solve`.

See [fax/constrained/constrained_test.py](fax/constrained/constrained_test.py) for examples. Please note that the API is subject to change.

## Installation


To get the latest version from Github: 
```sh
pip install git+https://github.com/gehring/fax.git
```

Otherwise on PyPI:
```sh
pip install jax-fixedpoint
```
## Basic Usage
The main entry point for Christianson's two-phases reverse accumulation is through `fax.implicit.two_phase_solver`. For example, imagine that you have a [fixed-point iteration method](https://en.wikipedia.org/wiki/Fixed-point_iteration) like [Power iteration](https://en.wikipedia.org/wiki/Power_iteration) and want to compute the gradient of a function of its output. You could write something like: 
```python
import jax
import jax.numpy as jnp
from fax import implicit


def power_iteration(A):
  def _power_iteration_step(b):
    b = A @ b
    return b/jnp.linalg.norm(b)
  return _power_iteration_step

def objective(A):
  b0 = jnp.ones((A.shape[0]))
  b = implicit.two_phase_solve(power_iteration, b0, A)
  return (b.T @ A @ b)/(b.T @ b)
  
A = jnp.array([[1, 2], [3, 4.]])
print(jax.grad(objective)(A))
# Output array should be close to:
# DeviceArray([[0.23888351, 0.52223295],
#             [0.34815535, 0.76111656]], dtype=float32)
```
Given a function and an initial guess, we can use `fax.implicit.two_phase_solve` to solve a fixed-point problem such that the result is differentiated using the [implicit](https://en.wikipedia.org/wiki/Implicit_function) form of the returned fixed-point. Behind the scene, `fax.implicit.two_phase_solve` tells `jax` to apply a custom [VJP rule](https://jax.readthedocs.io/en/latest/jax.html#jax.vjp) which `fax` derives from the fixed-point iteration function that it receives.

Not only does this provides numerical and computational benefits over backpropagating through the fixed-point iteration loop, it allows us to define gradients even when our fixed-point solver isn't differentiable. The `two_phase_solve` function allows us reproduce our power iteration example, but using a solver which `jax` is incapable of differentiating:

```python
import numpy as np
import jax
import jax.numpy as jnp
from fax import implicit


def numpy_max_eig(A):
  w, v = np.linalg.eig(A)
  return v[:, np.argmax(w)]

def power_iteration(A):
  def _power_iteration_step(b):
    b = A @ b
    return b/jnp.linalg.norm(b)
  return _power_iteration_step

def objective_non_diff_solver(A):
  b0 = jnp.ones((A.shape[0]))
  b = implicit.two_phase_solve(
      power_iteration,
      b0,
      A,
      solvers=(lambda f, init_b, matrix: numpy_max_eig(matrix),),
    )
  return (b.T @ A @ b)/(b.T @ b)
  
A = jnp.array([[1, 2], [3, 4.]])
print(jax.grad(objective_non_diff_solver)(A))
# Output array should be close to:
# DeviceArray([[0.23888351, 0.52223295],
#             [0.34815535, 0.76111656]], dtype=float32)
```

**NOTE:** this example will not work when jit'ed using [`jax.jit`](https://jax.readthedocs.io/en/latest/jax.html#jax.jit) since `jax` won't be able compile the "external" numpy call. This is only meant as a demonstration of how implicit differentiation doesn't care about whether the solver itself is differentiable; it only cares whether the fixed-point function is.

## References

Citing competitive differentiation:

```
@inproceedings{bacon2019optrl,
  author={Bacon, Pierre-Luc and Schafer, Florian and Gehring, Clement and Anandkumar, Animashree and Brunskill, Emma},
  title={A Lagrangian Method for Inverse Problems in Reinforcement Learning},
  booktitle={NeurIPS Optimization Foundations for Reinforcement Learning Workshop},
  year={2019},
  url={http://lis.csail.mit.edu/pubs/bacon-optrl-2019.pdf},
  keywords={Optimization, Reinforcement Learning, Lagrangian}
}
```

Citing this repo:

```
@misc{gehring2019fax,
  author = {Gehring, Clement and Bacon, Pierre-Luc and Schaefer, Florian},
  title = {{FAX: differentiating fixed point problems in JAX}},
  note = {Available at: https://github.com/gehring/fax},
  year = {2019}
}
```
