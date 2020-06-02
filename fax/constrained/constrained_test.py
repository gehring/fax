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
test_params = dict(rtol=1e-4, atol=1e-4, check_dtypes=False)
convergence_params = dict(rtol=1e-5, atol=1e-5)
benchmark = fax.test_util.load_HockSchittkowski_models()


class EGTest(jax.test_util.JaxTestCase):
    @absl.testing.parameterized.parameters(benchmark)
    def test_eg_HockSchittkowski(self, objective_function, equality_constraints, hs_optimal_value: np.array, initial_value):
        def convergence_test(x_new, x_old):
            return fax.converge.max_diff_test(x_new, x_old, **convergence_params)

        init_mult, lagrangian, get_x = make_lagrangian(objective_function, equality_constraints)

        x0 = initial_value()
        initial_values = init_mult(x0)

        final_val, h, x, multiplier = self.eg_solve(lagrangian, convergence_test, equality_constraints, objective_function, get_x, initial_values)

        import scipy.optimize
        constraints = ({'type': 'eq', 'fun': equality_constraints, },)

        res = scipy.optimize.minimize(lambda *args: -objective_function(*args), initial_values[0], method='SLSQP', constraints=constraints)
        scipy_optimal_value = -res.fun
        scipy_constraint = equality_constraints(res.x)

        print(objective_function)
        print(f"solution: {x} (ours) {res.x} (scipy)")
        print(f"final value: {final_val} (ours) {scipy_optimal_value} (scipy)")
        print(f"constraint: {h} (ours) {scipy_constraint} (scipy)")
        self.assertAllClose(final_val, scipy_optimal_value, **test_params)
        self.assertAllClose(h, scipy_constraint, **test_params)

    def eg_solve(self, lagrangian, convergence_test, equality_constraints, objective_function, get_x, initial_values):
        optimizer_init, optimizer_update, optimizer_get_params = extragradient.adam_extragradient_optimizer(
            step_size_x=jax.experimental.optimizers.inverse_time_decay(1e-1, 50, 0.3, staircase=True),
            step_size_y=5e-2,
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
