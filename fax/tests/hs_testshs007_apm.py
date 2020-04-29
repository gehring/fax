from jax.numpy import *


def initialize():
    return (
        zeros(2),  # x
    )


objective_function = lambda x: -log(1+x[0]**2) - x[1]
optimal_solution = -array(-sqrt(3))
h0 = lambda x: (1+x[0]**2)**2 + x[1]**2 
