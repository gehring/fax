# fax: fixed-point jax 

Implicit and [competitive differentiation](https://optrl2019.github.io/assets/accepted_papers/70.pdf) in [JAX](https://github.com/google/jax).

Our "competitive differentiation" approach uses [Competitive Gradient Descent](https://arxiv.org/abs/1905.12103) to solve the equality-constrained nonlinear program associated with the fixed-point problem. A standalone implementation of CGD is provided under [fax/competitive/cga.py](fax/competitive/cga.py) and the equality-constrained solver derived from it can be accessed via `fax.constrained.cga_lagrange_min` or `fax.constrained.cga_ecp`. An implementation of implicit differentiation based on [Christianson's](https://doi.org/10.1080/10556789408805572) two-phases reverse accumulation algorithm can also be obtained with the function `fax.implicit.two_phase_solver`.

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
The main entry point for Christianson's two-phases reverse accumulation is through `fax.implicit.two_phase_solver`. For example, imagine that have a [fixed-point iteration method](https://en.wikipedia.org/wiki/Fixed-point_iteration) like [Power iteration](https://en.wikipedia.org/wiki/Power_iteration) and want to compute the gradient of the a function of its output, you would write something like: 
```python
import jax.numpy as jnp
from jax import grad
from fax.implicit import two_phase_solver

def make_power_iteration(A):
  def _power_iteration(_, b):
    b = A @ b
    return b/jnp.linalg.norm(b)
  return _power_iteration

def make_objective(A):
  b0 = jnp.ones((A.shape[0]))
  power_iteration = two_phase_solver(make_power_iteration)
  def _objective(A):
    b = power_iteration(b0, A).value
    return (b.T @ A @ b)/(b.T @ b)
  return _objective
  
A = jnp.array([[1, 2], [3, 4.]])
max_eigenvalue = make_objective(A)
grad(max_eigenvalue)(A)
```
Note how `fax.implicit.two_phase_solver` is able to construct a new function `power_iteration` which takes an initial guess for the fixed-point procedure and operator parameters (``A`` in this case) and returns the fixed-point as output. The magic of `fax.implicit.two_phase_solver`  is that we can now compose `power_iteration` inside another function and JAX will apply the custom [VJP rule](https://jax.readthedocs.io/en/latest/jax.html#jax.vjp) defined by FAX when requesting the desired gradient. 

## References

Citing competitive differentiation:

```
@inproceedings{bacon2019optrl,
  author={Pierre-Luc Bacon, Florian Schaefer, Clement Gehring, Animashree Anandkumar, Emma Brunskill},
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
  author = {Clement Gehring, Pierre-Luc Bacon, Florian Schaefer},
  title = {{FAX: differentiating fixed point problems in JAX}},
  note = {Available at: https://github.com/gehring/fax},
  year = {2019}
}
```
