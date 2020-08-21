import jax.numpy as np
from jax import random
# from jax.scipy import linalg
from cmd_helper import DP_pd, DP_inv_pd, D2P_pd, inv_D2P_pd
from cmd import make_lagrangian
import collections

BregmanPotential = collections.namedtuple("BregmanPotential", ["DP", "DP_inv", "D2P", "inv_D2P"])
breg_pd = BregmanPotential(DP_pd, DP_inv_pd, D2P_pd, inv_D2P_pd)

key1 = random.PRNGKey(0)
key = random.PRNGKey(1)
x1 = np.array([1., 2., 3., 4., 5.])
x2 = random.normal(key1, (5, ))
W1 = random.normal(key, (3, 3))
W2 = random.normal(key1, (3, 3))

x = [(x1, x2), (x1, x1, x2)]
W = [(W1, W2), (W1, W1, W2)]


# Test lagrangian making portion
def obj_func(x, y):
    return 2 * x * y - (1 - y) ** 2


breg_min = breg_pd
breg_max = breg_pd

lagrangian, breg_min_aug, breg_max_aug, init_multipliers = make_lagrangian(obj_func, breg_min, breg_max)
print(lagrangian(1., 2., obj_func))
