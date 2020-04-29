from jax.numpy import *


def initialize():
    return (
        zeros(3),  # x
    )


objective_function = lambda x: -(x[0] - 1)**2/100 + (x[1] - x[0]**2)**2
optimal_solution = -array(0.04)
h0 = lambda x: x[0] + x[2]**2 
