"""Utility to create a bad pixel map from a dark frame and/or a flat frame."""

import numpy as np

import cal.util.check as check


def compute_fixedbp_excam(dark, flat, dthresh, ffrac, fwidth):
    """
    Compute a fixed bad pixel map for EXCAM from a master dark and flat.

    Parameters
    ----------
    dark : array_like
        2-D array with master dark frame.
    flat : array_like
        2-D array with master flat field.
    dthresh : float
        Number of standard deviations above the mean to threshold for
        hot pixels. Must be >= 0.
    ffrac : float
        Fraction of local mean value below which poorly-functioning pixels
        are flagged. Must be >=0.
    fwidth : int
        Number of pixels to include in local mean check with ffrac. Must be >0.

    Returns
    -------
    fixedbp : array_like
        2-D boolean array the same size as dark and flat, with bad pixels True.

    """
    # Check inputs
    check.twoD_array(dark, 'dark', TypeError)
    check.twoD_array(flat, 'flat', TypeError)
    if flat.shape != dark.shape:
        raise TypeError('dark and flat must be the same shape')
    check.real_nonnegative_scalar(dthresh, 'dthresh', TypeError)
    check.real_nonnegative_scalar(ffrac, 'ffrac', TypeError)
    check.positive_scalar_integer(fwidth, 'fwidth', TypeError)

    # Combine bad pixel maps
    fixedbp_dark = _compute_fixedbp_from_dark(dark, dthresh)
    fixedbp_flat = _compute_fixedbp_from_flat(flat, ffrac, fwidth)
    fixedbp = np.logical_or(fixedbp_dark, fixedbp_flat)

    return fixedbp


def _compute_fixedbp_from_dark(dark, dthresh):
    """
    Compute a fixed bad pixel map from a master dark frame.

    Detects warm/hot pixels from the dark frame, flagging anything above
    dthresh standard deviations above the mean dark level in the frame. For
    example, dthresh = 5 will flag any pixel > 5 standard deviations above the
    mean of dark.

    Parameters
    ----------
    dark : array_like
        2-D array with master dark frame.
    dthresh : float
        Number of standard deviations above the mean to threshold for
        hot pixels. Must be >= 0.

    Returns
    -------
    fixedbp_dark : array_like
        2D boolean array the same size as dark, with bad pixels set as True.

    """
    # Check inputs
    check.twoD_array(dark, 'dark', TypeError)
    check.real_nonnegative_scalar(dthresh, 'dthresh', TypeError)

    # Process dark frame
    fixedbp_dark = np.zeros(dark.shape).astype('bool')
    fixedbp_dark[dark > np.mean(dark) + dthresh*np.std(dark)] = True

    return fixedbp_dark


def _compute_fixedbp_from_flat(flat, ffrac, fwidth):
    """
    Compute a fixed bad pixel map from a flat field.

    Detects low- or non-functional pixels from the flat frame, flagging any
    pixels less than ffrac times the local mean flat level.  Flat uses
    local mean as flats may have low-spatial-frequency variations due to e.g.
    fringing or vignetting.  For example, ffrac = 0.8 and fwidth = 32 will
    flag any pixel which is < 80% of the mean value in a 32-pixel box
    centered on the pixel.  (Centration will use FFT rules, where odd-sized
    widths center on the pixel, and even-sized place the pixel to the right of
    center, e.g.:
     odd: [. x .]
     even: [. . x .]
    For boxes near the edge, only the subset of pixels within the box will be
    used for the calculation.


    Parameters
    ----------
    flat : array_like
        2-D array with master flat field.
    ffrac : float
        Fraction of local mean value below which poorly-functioning pixels
        are flagged. Must be >=0.
    fwidth : int
        Number of pixels to include in local mean check with ffrac. Must be >0.

    Returns
    -------
    fixedbp_flat : array_like
        2D boolean array the same size as flat, with bad pixels set as True.

    """
    # Check inputs
    check.twoD_array(flat, 'flat', TypeError)
    check.real_nonnegative_scalar(ffrac, 'ffrac', TypeError)
    check.positive_scalar_integer(fwidth, 'fwidth', TypeError)

    # Process flat frame
    fixedbp_flat = np.zeros(flat.shape).astype('bool')
    nrow, ncol = flat.shape
    for r in range(nrow):
        tmpr = np.arange(fwidth) - fwidth//2 + r
        rind = np.logical_and(tmpr >= 0, tmpr < nrow)
        for c in range(ncol):
            tmpc = np.arange(fwidth) - fwidth//2 + c
            cind = np.logical_and(tmpc >= 0, tmpc < ncol)

            # rind/cind removes indices that fall out of flat
            subflat = flat[tmpr[rind], :][:, tmpc[cind]]

            localm = np.mean(subflat)
            if flat[r, c] < ffrac*localm:
                fixedbp_flat[r, c] = True
                pass

            pass
        pass

    return fixedbp_flat
