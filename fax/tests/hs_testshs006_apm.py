from jax.numpy import *


def initialize():
    return (
        zeros(2),  # x
    )


objective_function = lambda x: -(1-x[0])**2
optimal_solution = -array(0)
h0 = lambda x: 10*(x[1] - x[0]**2)  -  0
