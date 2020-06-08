from jax.config import config
from jax.numpy import *

config.update("jax_enable_x64", True)


class Hs:
    constraints = lambda: 0


class Hs01(Hs):
    initialize = lambda: (
        zeros(2),  # x
    )

    objective_function = lambda x: -(100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2)
    optimal_solution = -array(0)


class Hs06(Hs):
    initialize = lambda: (
        zeros(2),  # x
    )

    objective_function = lambda x: -((1 - x[0]) ** 2)
    optimal_solution = -array(0)
    h0 = lambda x: 10 * (x[1] - x[0] ** 2) - 0

    def constraints(self, x):
        return stack((self.h0(x)))


class Hs07(Hs):
    initialize = lambda: (
        zeros(2),  # x
    )

    objective_function = lambda x: -(log(1 + x[0] ** 2) - x[1])
    optimal_solution = -array(-sqrt(3))
    h0 = lambda x: (1 + x[0] ** 2) ** 2 + x[1] ** 2

    def constraints(self, x):
        return stack((self.h0(x)))


class Hs08(Hs):
    initialize = lambda: (
        zeros(2),  # x
    )

    objective_function = lambda x: -(-1)
    optimal_solution = -array(-1)
    h0 = lambda x: x[0] ** 2 + x[1] ** 2
    h1 = lambda x: x[0] * x[1]

    def constraints(self, x):
        return stack((self.h0(x), self.h1(x)))


class Hs09(Hs):
    initialize = lambda: (
        zeros(2),  # x
    )

    objective_function = lambda x: -(sin(pi * x[0] / 12) * cos(pi * x[1] / 16))
    optimal_solution = -array(-0.5)
    h0 = lambda x: 4 * x[0] - 3 * x[1] - 0

    def constraints(self, x):
        return stack((self.h0(x)))


class Hs26(Hs):
    initialize = lambda: (
        zeros(3),  # x
    )

    objective_function = lambda x: -((x[0] - x[1]) ** 2 + (x[1] - x[2]) ** 4)
    optimal_solution = -array(0)
    h0 = lambda x: (1 + x[1] ** 2) * x[0] + x[2] ** 4

    def constraints(self, x):
        return stack((self.h0(x)))


class Hs27(Hs):
    initialize = lambda: (
        zeros(3),  # x
    )

    objective_function = lambda x: -((x[0] - 1) ** 2 / 100 + (x[1] - x[0] ** 2) ** 2)
    optimal_solution = -array(0.04)
    h0 = lambda x: x[0] + x[2] ** 2

    def constraints(self, x):
        return stack((self.h0(x)))


class Hs28(Hs):
    initialize = lambda: (
        zeros(3),  # x
    )

    objective_function = lambda x: -((x[0] + x[1]) ** 2 + (x[1] + x[2]) ** 2)
    optimal_solution = -array(0)
    h0 = lambda x: x[0] + 2 * x[1] + 3 * x[2]

    def constraints(self, x):
        return stack((self.h0(x)))


class Hs39(Hs):
    initialize = lambda: (
        zeros(4),  # x
    )

    objective_function = lambda x: -(-x[0])
    optimal_solution = -array(-1)
    h0 = lambda x: x[1] - x[0] ** 3 - x[2] ** 2 - 0
    h1 = lambda x: x[0] ** 2 - x[1] - x[3] ** 2 - 0

    def constraints(self, x):
        return stack((self.h0(x), self.h1(x)))


class Hs40(Hs):
    initialize = lambda: (
        zeros(4),  # x
    )

    objective_function = lambda x: -(-x[0] * x[1] * x[2] * x[3])
    optimal_solution = -array(-0.25)
    h0 = lambda x: x[0] ** 3 + x[1] ** 2
    h1 = lambda x: x[0] ** 2 * x[3] - x[2] - 0
    h2 = lambda x: x[3] ** 2 - x[1] - 0

    def constraints(self, x):
        return stack((self.h0(x), self.h1(x), self.h2(x)))


class Hs46(Hs):
    initialize = lambda: (
        zeros(5),  # x
    )

    objective_function = lambda x: -((x[0] - x[1]) ** 2 + (x[2] - 1) ** 2 + (x[3] - 1) ** 4 + (x[4] - 1) ** 6)
    optimal_solution = -array(0)
    h0 = lambda x: x[0] ** 2 * x[3] + sin(x[3] - x[4])
    h1 = lambda x: x[1] + x[2] ** 4 * x[3] ** 2

    def constraints(self, x):
        return stack((self.h0(x), self.h1(x)))


class Hs47(Hs):
    initialize = lambda: (
        zeros(5),  # x
    )

    objective_function = lambda x: -((x[0] - x[1]) ** 2 + (x[1] - x[2]) ** 3 + (x[2] - x[3]) ** 4 + (x[3] - x[4]) ** 4)
    optimal_solution = -array(0)
    h0 = lambda x: x[0] + x[1] ** 2 + x[2] ** 3
    h1 = lambda x: x[1] - x[2] ** 2 + x[3]
    h2 = lambda x: x[0] * x[4]

    def constraints(self, x):
        return stack((self.h0(x), self.h1(x), self.h2(x)))


class Hs49(Hs):
    initialize = lambda: (
        zeros(5),  # x
    )

    objective_function = lambda x: -((x[0] - x[1]) ** 2 + (x[2] - 1) ** 2 + (x[3] - 1) ** 4 + (x[4] - 1) ** 6)
    optimal_solution = -array(0)
    h0 = lambda x: x[0] + x[1] + x[2] + x[3] + x[4] + 3 * x[3]
    h1 = lambda x: x[2] + 5 * x[4]

    def constraints(self, x):
        return stack((self.h0(x), self.h1(x)))


class Hs50(Hs):
    initialize = lambda: (
        zeros(5),  # x
    )

    objective_function = lambda x: -((x[0] - x[1]) ** 2 + (x[1] - x[2]) ** 2 + (x[2] - x[3]) ** 4 + (x[3] - x[4]) ** 2)
    optimal_solution = -array(0)
    h0 = lambda x: x[0] + 2 * x[1] + 3 * x[2]
    h1 = lambda x: x[1] + 2 * x[2] + 3 * x[3]
    h2 = lambda x: x[2] + 2 * x[3] + 3 * x[4]

    def constraints(self, x):
        return stack((self.h0(x), self.h1(x), self.h2(x)))


class Hs51(Hs):
    initialize = lambda: (
        zeros(5),  # x
    )

    objective_function = lambda x: -((x[0] - x[1]) ** 2 + (x[1] + x[2] - 2) ** 2 + (x[3] - 1) ** 2 + (x[4] - 1) ** 2)
    optimal_solution = -array(0)
    h0 = lambda x: x[0] + 3 * x[1]
    h1 = lambda x: x[2] + x[3] - 2 * x[4] - 0
    h2 = lambda x: x[1] - x[4] - 0

    def constraints(self, x):
        return stack((self.h0(x), self.h1(x), self.h2(x)))


class Hs52(Hs):
    initialize = lambda: (
        zeros(5),  # x
    )

    objective_function = lambda x: -((4 * x[0] - x[1]) ** 2 + (x[1] + x[2] - 2) ** 2 + (x[3] - 1) ** 2 + (x[4] - 1) ** 2)
    optimal_solution = -array(1859 / 349)
    h0 = lambda x: x[0] + 3 * x[1] - 0
    h1 = lambda x: x[2] + x[3] - 2 * x[4] - 0
    h2 = lambda x: x[1] - x[4] - 0

    def constraints(self, x):
        return stack((self.h0(x), self.h1(x), self.h2(x)))


class Hs61(Hs):
    initialize = lambda: (
        zeros(3),  # x
    )

    objective_function = lambda x: -(4 * x[0] ** 2 + 2 * x[1] ** 2 + 2 * x[2] ** 2 - 33 * x[0] + 16 * x[1] - 24 * x[2])
    optimal_solution = -array(- 143.6461422)
    h0 = lambda x: 3 * x[0] - 2 * x[1] ** 2
    h1 = lambda x: 4 * x[0] - x[2] ** 2

    def constraints(self, x):
        return stack((self.h0(x), self.h1(x)))


class Hs77(Hs):
    initialize = lambda: (
        zeros(5),  # x
    )

    objective_function = lambda x: -((x[0] - 1) ** 2 + (x[0] - x[1]) ** 2 + (x[2] - 1) ** 2 + (x[3] - 1) ** 4 + (x[4] - 1) ** 6)
    optimal_solution = -array(0.24150513)
    h0 = lambda x: x[0] ** 2 * x[3] + sin(x[3] - x[4])
    h1 = lambda x: x[1] + x[2] ** 4 * x[3] ** 2

    def constraints(self, x):
        return stack((self.h0(x), self.h1(x)))


class Hs78(Hs):
    initialize = lambda: (
        zeros(5),  # x
    )

    objective_function = lambda x: -(x[0] * x[1] * x[2] * x[3] * x[4])
    optimal_solution = -array(-2.91970041)
    h0 = lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2 + x[4] ** 2
    h1 = lambda x: x[1] * x[2] - 5 * x[3] * x[4] - 0
    h2 = lambda x: x[0] ** 3 + x[1] ** 3

    def constraints(self, x):
        return stack((self.h0(x), self.h1(x), self.h2(x)))


class Hs79(Hs):
    initialize = lambda: (
        zeros(5),  # x
    )

    objective_function = lambda x: -((x[0] - 1) ** 2 + (x[0] - x[1]) ** 2 + (x[1] - x[2]) ** 2 + (x[2] - x[3]) ** 4 + (x[3] - x[4]) ** 4)
    optimal_solution = -array(0.0787768209)
    h0 = lambda x: x[0] + x[1] ** 2 + x[2] ** 3
    h1 = lambda x: x[1] - x[2] ** 2 + x[3]
    h2 = lambda x: x[0] * x[4]

    def constraints(self, x):
        return stack((self.h0(x), self.h1(x), self.h2(x)))
