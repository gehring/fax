from jax.numpy import *


def initialize():
    return (
        zeros(3),  # x
    )


objective_function = lambda x: -(x[0] + x[1])**2 + (x[1] + x[2])**2
optimal_solution = -array(0)
h0 = lambda x: x[0] + 2*x[1] + 3*x[2] 
