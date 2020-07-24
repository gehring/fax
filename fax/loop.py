import collections
import warnings

import jax
import jax.numpy as np

FixedPointSolution = collections.namedtuple(
    "FixedPointSolution",
    "value converged iterations previous_value"
)


def unrolled(i, init_x, func, num_iter, return_last_two=False):
    """Repeatedly apply a function using a regular python loop.

    Args:
        init_x: The initial values fed to `func`.
        func (callable): The function which is repeatedly called.
        num_iter (int): The number of times to apply the function `func`.
        return_last_two (bool, optional): When `True`, return the two last
            outputs of `func`.

    Returns:
        The last output of `func`, or, if `return_last_two` is `True`, a tuple
        with the last two outputs of `func`.
    """
    x = init_x
    x_old = None

    for _ in range(num_iter):
        x_old = x
        x = func(x_old)
        i = i + 1

    if return_last_two:
        return i, x, x_old
    else:
        return i, x


def fixed_point_iteration(init_x, func, convergence_test, max_iter,
                          batched_iter_size=1, unroll=False):
    """Find a fixed point of `func` by repeatedly applying `func`.

    Use this function to find a fixed point of `func` by repeatedly applying
    `func` to a candidate solution. This is done until the solution converges
    or until the maximum number of iterations, `max_iter` is reached.

    NOTE: if the maximum number of iterations is reached, the convergence
    will not be checked on the final application of `func` and the solution
    will always be marked as not converged when `unroll` is `False`.

    Args:
        init_x: The initial values to be used in `func`.
        func (callable): The function for which we want to find a fixed point.
            `func` should be of type `int, a -> a` where `a` is the type of
            `init_x` and the integer correspond to the current iteration count.
        convergence_test (callable): A two argument function of type
            `(a, a) -> bool` that takes in the newest solution and the previous
            solution and returns `True` if they have converged. The fixed point
            iteration will stop and return when `True` is returned.
        max_iter (int or None): The maximum number of iterations.
        batched_iter_size (int, optional): The number of iterations to be
            unrolled and executed per iterations of `while_loop` op. Convergence
            is only tested at the beginning of each batch. Set this to a number
            larger than 1 to reduce the number of times convergence is checked
            and to potentially allow for the graph of the unrolled batch to be
            more aggressively optimized.
        unroll (bool): If True, use `jax.lax.scan` instead of 
            `jax.lax.while`. This enables back-propagating through the iterations.
            
            NOTE: due to current limitations in `JAX`, when `unroll` is `True`,
            convergence is ignored and the loop always runs for the maximum
            number of iterations.

    Returns:
        FixedPointSolution: A named tuple containing the results of the
            fixed point iteration. The tuple contains the attributes `value`
            (the final solution), `converged` (a bool indicating whether
            convergence was achieved), `iterations` (the number of iterations
            used), and `previous_value` (the value of the solution on the
            previous iteration). The previous value satisfies
            `sol.value=func(sol.previous_value)` and allows us to log the size
            of the last step if desired.
    """

    if batched_iter_size < 1:
        raise ValueError(
            "Argument `batch_iter_size` must be greater than zero.")

    if max_iter is not None and batched_iter_size > max_iter:
        raise ValueError((
            "Argument `batched_iter_size` must be smaller or equal to "
            "`max_iter`."))

    if max_iter is not None and max_iter % batched_iter_size != 0:
        warnings.warn((
            "Argument `batched_iter_size` should be a multiple of `max_iter` "
            "to guarantee that no more than `max_iter` iterations are used."))

    max_batched_iter = None
    if max_iter is not None:
        max_batched_iter = max_iter // batched_iter_size

    def cond(args):
        i, x_new, x_old = args
        converged = convergence_test(x_new, x_old)

        if max_iter is not None:
            converged = converged | (max_iter <= i)
        return np.logical_not(converged)

    def body(args):
        i, x_new, _ = args
        i_new, x_new, x_old = unrolled(i, x_new, func, batched_iter_size,
                                       return_last_two=True)
        return i_new, x_new, x_old

    init_vals = unrolled(0, init_x, func, batched_iter_size,
                         return_last_two=True)

    if unroll:
        if max_batched_iter is None:
            raise ValueError("`max_iter` must be not None when using `unroll`.")

        def scan_step(args, idx):
            del idx
            return body(args), None

        if max_batched_iter < 2:
            iterations, sol, prev_sol = init_vals
        else:
            (iterations, sol, prev_sol), _ = jax.lax.scan(
                f=scan_step,
                init=init_vals,
                xs=np.arange(max_batched_iter - 1),
            )
        converged = convergence_test(sol, prev_sol)

    else:
        iterations, sol, prev_sol = jax.lax.while_loop(
            cond,
            body,
            init_vals,
        )
        converged = max_iter is None or iterations < max_iter

    return FixedPointSolution(
        value=sol,
        converged=converged,
        iterations=iterations,
        previous_value=prev_sol,
    )
