"""Fill in outer values in an array by extending inner values radially."""
import numpy as np

from . import check


def extend_radially(array_in, x_offset=0, y_offset=0, direction=1, atol=1e-10):
    """
    Extend values radially outward in an array.

    Only values below the tolerance atol are overwritten.

    Parameters
    ----------
    array_in : array_like
        2-D array to modify
    x_offset, y_offset : float
        Lateral offsets in pixels from the array's center pixel to use
        when defining which point to move out radially from. Uses the FFT
        centering convention for the array center pixel. Default is 0.
    direction : int
        Sign of the direction to extend radially. Choices are -1 and 1.
        1 means extend outward, and -1 means extend inward. Default is 1.
    atol : float
        Tolerance below which values will be overwritten with their radially
        inner neighbor's value. Default is 1e-10.

    Returns
    -------
    array_out : numpy ndarray
        2-D array with values extended out radially
    """
    array_in = check.twoD_array(array_in, 'array_in', TypeError)
    check.real_scalar(x_offset, 'x_offset', TypeError)
    check.real_scalar(y_offset, 'y_offset', TypeError)
    check.scalar_integer(direction, 'direction', TypeError)
    if direction not in [-1, 1]:
        raise TypeError('direction must be either -1 or 1')
    check.real_positive_scalar(atol, 'atol', TypeError)

    # Compute coordinates
    n_total = array_in.size
    ny, nx = array_in.shape
    x0 = np.arange(nx) - nx//2  # FFT centering convention
    y0 = np.arange(ny) - ny//2
    x = x0 - x_offset
    y = y0 - y_offset
    [X, Y] = np.meshgrid(x, y)
    RHO = np.sqrt(X*X + Y*Y)
    if np.round(x_offset) > np.max(x0) or np.round(x_offset) < np.min(x0):
        raise ValueError('x_offset places center outside the input array. '
                         'The center must be within the given array.')
    if np.round(y_offset) > np.max(y0) or np.round(y_offset) < np.min(y0):
        raise ValueError('y_offset places center outside the input array. '
                         'The center must be within the given array.')

    # Sort the indices based on distance from center.
    ind_sorted_1d = list(np.argsort(RHO, axis=None))
    
    # Remove the center index for outward filling-in because it has no inner
    # neighbor, or if inward filling and it is at (0, 0) because the slope
    # is zero there and there is not a unique outer neighbor.
    center_pixel_has_zero_slope = (X.flatten()[ind_sorted_1d[0]] == 0 and
                                   Y.flatten()[ind_sorted_1d[0]] == 0)
    if (direction == 1) or center_pixel_has_zero_slope:
        ind_sorted_1d = ind_sorted_1d[1::]

    ind_sorted_2d = []
    for ii in ind_sorted_1d:
        ind_sorted_2d.append(np.unravel_index(ii, array_in.shape))
    
    # Reverse ordering for inward direction
    if direction == -1:
        ind_sorted_1d.reverse()
        ind_sorted_2d.reverse()

    # Compute the normalized slopes at each pixel to determine which direction
    # to move inward when checking for the nearest more inward pixel.
    slopes_norm = np.zeros((2, n_total))
    slopes_norm[0, :] = -2 * X.flatten() * direction
    slopes_norm[1, :] = -2 * Y.flatten() * direction
    for ii in ind_sorted_1d:
        slopes_norm[:, ii] /= np.sqrt(slopes_norm[0, ii]**2 +
                                      slopes_norm[1, ii]**2)

    # Don't do computations for pixels that won't have values updated
    ind_mat_orig = np.arange(n_total, dtype=int).reshape(array_in.shape)
    # Skip outermost rows and columns if going inward
    if direction == -1:
        use_map = np.ones_like(array_in)
        use_map[[0, -1], :] = 0
        use_map[:, [0, -1]] = 0
        use_map = use_map.astype(bool)
        ind_inner = ind_mat_orig[use_map]
    else:
        ind_inner = ind_mat_orig
    ind_values_to_replace_1d = np.intersect1d(
        ind_mat_orig[np.abs(array_in) < atol].flatten(), ind_inner.flatten())

    # Find the nearest inward (or outward) neighbor for each pixel
    ind_mat_neighbor = np.zeros_like(array_in, dtype=int)
    for ii in ind_sorted_1d:
        if ii in ind_values_to_replace_1d:
            ind_2d = np.unravel_index(ii, array_in.shape)
            x_neighbor = (X.flatten()[ii] + slopes_norm[0, ii])
            y_neighbor = (Y.flatten()[ii] + slopes_norm[1, ii])
            match_map = np.logical_and(
                np.logical_and(X > x_neighbor - 0.5, X <= x_neighbor + 0.5),
                np.logical_and(Y > y_neighbor - 0.5, Y <= y_neighbor + 0.5))

            if np.sum(match_map) == 1:
                ind_mat_neighbor[ind_2d] = ind_mat_orig[match_map][0]
            else:
                raise ValueError('Wrong number (%d) of matching values. '
                                 'Expected 1.' % np.sum(match_map))

    # Fill in the values below a given magnitude with the nearest inner/outer
    # neighbor's value.
    array_out = array_in.copy()
    for ind_2d in ind_sorted_2d:

        if np.abs(array_out[ind_2d]) < atol:

            array_out[ind_2d] = array_out[
                np.unravel_index(ind_mat_neighbor[ind_2d], array_in.shape)]

    return array_out
