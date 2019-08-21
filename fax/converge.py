import numpy as onp

import jax.numpy as np


def adjust_tol_for_dtype(rtol, atol, dtype):
    """Adjust tolerances to the closest achievable values.

    Args:
        rtol (float): The relative tolerance (as used by `np.isclose`).
        atol (float): The absolute tolerance (as used by `np.isclose`).
        dtype (type): The data type used.

    Returns:
        A tuple `(rtol, atol)` of tolerances which are achievable when comparing
        floats of type `dtype`. If the given tolerances are large enough then
        the tolerances are returned unchanged.
    """
    finfo = onp.finfo(dtype)

    rtol = max(rtol, finfo.resolution)
    atol = max(atol, finfo.eps)

    return rtol, atol


def is_tolerance_achievable(rtol, atol, dtype):
    """Check if the tolerances are achievable for a given type.

        Args:
            rtol (float): The relative tolerance (as used by `np.isclose`).
            atol (float): The absolute tolerance (as used by `np.isclose`).
            dtype (type): The data type used.

        Returns:
            bool: True if the tolerance are achievable.
        """
    adj_rtol, adj_atol = adjust_tol_for_dtype(rtol, atol, dtype)
    return adj_rtol == rtol and adj_atol == atol


def close_or_nan(delta, scale, rtol, atol):
    is_close = delta < (rtol * scale + atol)
    is_nan = np.any(np.isnan(delta))
    return np.logical_or(is_close, is_nan)


def max_diff_test(x_new, x_old, rtol, atol):
    delta = np.max(np.abs(x_new - x_old))
    abs_old = np.max(np.abs(x_old))
    return close_or_nan(delta, abs_old, rtol, atol)
