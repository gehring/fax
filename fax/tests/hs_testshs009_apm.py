from jax.numpy import *


def initialize():
    return (
        zeros(2),  # x
    )


objective_function = lambda x: -sin(pi * x[0] / 12) * cos(pi * x[1] / 16)
optimal_solution = -array(-0.5)
h0 = lambda x: 4*x[0] - 3*x[1]  -  0
