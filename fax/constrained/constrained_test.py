import absl.testing
import absl.testing.parameterized
import jax
import jax.experimental.optimizers
import jax.nn
import jax.numpy as np
import jax.scipy.special
import jax.test_util
import jax.tree_util
from absl.testing import absltest

import fax
import fax.config
import fax.test_util
from fax.competitive import extragradient
from fax.constrained import make_lagrangian

jax.config.update("jax_enable_x64", True)
test_params = dict(rtol=1e-3, atol=1e-3, check_dtypes=False)
convergence_params = dict(rtol=1e-5, atol=1e-5)
benchmark = list(fax.test_util.load_HockSchittkowski_models())
if fax.config.DEBUG:
    benchmark = [benchmark[1], ]

"""
class CGATest(jax.test_util.JaxTestCase):
    def test_cga_lagrange_min(self):
        n = 5
        opt_prob = fax.test_util.constrained_opt_problem(n)
        func, eq_constraints, _, opt_val = opt_prob

        init_mult, lagrangian, get_x = make_lagrangian(func, eq_constraints)

        rng = jax.random.PRNGKey(8413)
        init_params = jax.random.uniform(rng, (n,))
        lagr_params = init_mult(init_params)

        lr = 0.5
        rtol = atol = 1e-6
        opt_init, opt_update, get_params = cga_lagrange_min(lagrangian, lr)

        def convergence_test(x_new, x_old):
            return fax.converge.max_diff_test(x_new, x_old, rtol, atol)

        @jax.jit
        def step(i, opt_state):
            params = get_params(opt_state)
            grad_fn = jax.grad(lagrangian, (0, 1))
            grads = grad_fn(*params)
            return opt_update(i, grads, opt_state)

        opt_state = opt_init(lagr_params)

        for i in range(500):
            old_params = get_params(opt_state)
            opt_state = step(i, opt_state)

            if convergence_test(get_params(opt_state), old_params):
                break

        final_params = get_params(opt_state)
        self.assertAllClose(opt_val, func(get_x(final_params)), **test_params)

        h = eq_constraints(get_x(final_params))
        self.assertAllClose(h, jax.tree_util.tree_map(np.zeros_like, h), **test_params)

    @parameterized.parameters(
        {'method': cga_ecp, 'kwargs': {'max_iter': 1000, 'lr_func': 0.5}},
        {'method': slsqp_ecp, 'kwargs': {'max_iter': 1000}}, )
    @hypothesis.settings(max_examples=10, deadline=5000.)
    @hypothesis.given(
        hypothesis.extra.numpy.arrays(
            onp.float, (2,),
            elements=hypothesis.strategies.floats(0.1, 1)),
    )
    def test_ecp(self, method, kwargs, v):
        opt_solution = (1. / np.linalg.norm(v)) * v

        def objective(x, y):
            return np.dot(np.asarray([x, y]), v)

        def constraints(x, y):
            return 1 - np.linalg.norm(np.asarray([x, y]))

        rng = jax.random.PRNGKey(8413)
        initial_values = jax.random.uniform(rng, (len(v),))

        solution = method(objective, constraints, initial_values, **kwargs)
        self.assertAllClose(objective(*opt_solution), objective(*solution.value), **test_params)

    @parameterized.parameters(
        {'method': implicit_ecp,
         'kwargs': {'max_iter': 1000, 'lr_func': 0.01, 'optimizer': jax.experimental.optimizers.adam}},
        {'method': cga_ecp, 'kwargs': {'max_iter': 1000, 'lr_func': 0.15, 'lr_multipliers': 0.925}},
        {'method': slsqp_ecp, 'kwargs': {'max_iter': 1000}},
    )
    def test_omd(self, method, kwargs):
        true_transition = np.array([[[0.7, 0.3], [0.2, 0.8]],
                                    [[0.99, 0.01], [0.99, 0.01]]])
        true_reward = np.array(([[-0.45, -0.1],
                                 [0.5, 0.5]]))
        temperature = 1e-2
        true_discount = 0.9
        initial_distribution = np.ones(2) / 2

        optimal_value = 1.0272727  # pre-computed in other experiments, outside this code

        def smooth_bellman_optimality_operator(x, params):
            transition, reward, discount, temperature = params
            return reward + discount * np.einsum('ast,t->sa', transition, temperature * jax.scipy.special.logsumexp((1. / temperature) * x, axis=1))

        @jax.jit
        def objective(x, params):
            del params
            policy = jax.nn.softmax((1. / temperature) * x)
            ppi = np.einsum('ast,sa->st', true_transition, policy)
            rpi = np.einsum('sa,sa->s', true_reward, policy)
            vf = np.linalg.solve(np.eye(true_transition.shape[-1]) - true_discount * ppi, rpi)
            return initial_distribution @ vf

        @jax.jit
        def equality_constraints(x, params):
            transition_logits, reward_hat = params
            transition_hat = jax.nn.softmax((1. / temperature) * transition_logits)
            params = (transition_hat, reward_hat, true_discount, temperature)
            return smooth_bellman_optimality_operator(x, params) - x

        initial_values = (
            np.zeros_like(true_reward),
            (np.zeros_like(true_transition), np.zeros_like(true_reward))
        )
        solution = method(objective, equality_constraints, initial_values, **kwargs)
        self.assertAllClose(objective(*solution.value), optimal_value, **test_params)
"""


class EGTest(jax.test_util.JaxTestCase):
    def DISABLED_test_eg_lagrange_min(self):
        objective_function, equality_constraints, _, opt_val = fax.test_util.constrained_opt_problem(n=5)

        def convergence_test(x_new, x_old):
            return fax.converge.max_diff_test(x_new, x_old, **convergence_params)

        init_mult, lagrangian, get_x = make_lagrangian(objective_function, equality_constraints)

        rng = jax.random.PRNGKey(8413)
        initial_values = init_mult(jax.random.uniform(rng, (1,)))

        def maximize_lagrangian(*args):
            return -lagrangian(*args)

        final_val, h, x, _ = self.eg_solve(maximize_lagrangian, convergence_test, equality_constraints, objective_function, get_x, initial_values)

        print('val', opt_val, final_val)
        self.assertAllClose(opt_val, final_val, **test_params)
        print('h', h, 0)
        self.assertAllClose(h, jax.tree_util.tree_map(np.zeros_like, h), **test_params)

    @absl.testing.parameterized.parameters(
        list(dict(zip(['objective_function', 'equality_constraints', 'hs_optimal_value', 'state_space', 'model_name'], b)) for b in benchmark)
    )
    def test_eg_HockSchittkowski(self, objective_function, equality_constraints, hs_optimal_value: np.array, state_space, model_name) -> None:
        # TODO: plot real function + costraints
        # TODO: add x[0], initial xs

        def convergence_test(x_new, x_old):
            return fax.converge.max_diff_test(x_new, x_old, **convergence_params)

        init_mult, lagrangian, get_x = make_lagrangian(objective_function, equality_constraints)
        initial_values = init_mult(np.zeros(state_space.shape))
        final_val, h, x, multiplier = self.eg_solve(lagrangian, convergence_test, equality_constraints, objective_function, get_x, initial_values)

        import scipy.optimize
        cons = (
            {'type': 'eq', 'fun': equality_constraints, },
        )

        res = scipy.optimize.minimize(lambda *args: -objective_function(*args), initial_values[0], method='SLSQP', constraints=cons)
        scipy_optimal_value = res.fun
        scipy_constraint = equality_constraints(res.x)

        print(model_name)
        print(f"solution: {x} (ours) {res.x} (scipy)")
        print(f"final value: {final_val} (ours) {scipy_optimal_value} (scipy)")
        print(f"constraint: {h} (ours) {scipy_constraint} (scipy)")
        self.assertAllClose(final_val, scipy_optimal_value, **test_params)
        self.assertAllClose(h, scipy_constraint, **test_params)

    def eg_solve(self, lagrangian, convergence_test, equality_constraints, objective_function, get_x, initial_values):
        # optimizer_init, optimizer_update, optimizer_get_params = extragradient.rprop_extragradient_optimizer(
        #     step_size_x=1e-2,
        #     step_size_y=1e-3,
        # )

        optimizer_init, optimizer_update, optimizer_get_params = extragradient.adam_extragradient_optimizer(
            step_size_x=jax.experimental.optimizers.inverse_time_decay(1e-2, 50, 0.3, staircase=True),
            step_size_y=5e-3,
            # step_size_y=jax.experimental.optimizers.inverse_time_decay(1e-3, 50, 0.3, staircase=False),
        )

        @jax.jit
        def update(i, opt_state):
            grad_fn = jax.grad(lagrangian, (0, 1))
            return optimizer_update(i, grad_fn, opt_state)

        fixpoint_fn = fax.loop._debug_fixed_point_iteration if fax.config.DEBUG else fax.loop.fixed_point_iteration
        solution = fixpoint_fn(
            init_x=optimizer_init(initial_values),
            func=update,
            convergence_test=convergence_test,
            max_iter=100000000,
            get_params=optimizer_get_params,
            f=lagrangian,
        )
        x, multipliers = get_x(solution)
        final_val = objective_function(x)
        h = equality_constraints(x)
        return final_val, h, x, multipliers


if __name__ == "__main__":
    absltest.main()
