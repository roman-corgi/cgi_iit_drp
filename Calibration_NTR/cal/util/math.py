"""Module to hold basic math functions."""
import numpy as np

from . import check


def rms(arrayIn):
    """Compute the root mean square of a real-valued array."""
    check.real_array(arrayIn, 'arrayIn', TypeError)

    return np.sqrt(np.mean(arrayIn**2))


def ceil_odd(x_in):
    """
    Compute the next highest odd integer above the input.

    Parameters
    ----------
    x_in : float
        Scalar value

    Returns
    -------
    x_out : integer
        Odd-valued integer
    """
    check.real_scalar(x_in, 'x_in', TypeError)

    x_out = int(np.ceil(x_in))
    if x_out % 2 == 0:
        x_out += 1
    return x_out


def ceil_even(x_in):
    """
    Compute the next highest even integer above the input.

    Parameters
    ----------
    x_in : float
        Scalar value

    Returns
    -------
    x_out : int
        Even-valued integer
    """
    check.real_scalar(x_in, 'x_in', TypeError)

    return int(2 * np.ceil(0.5 * x_in))
