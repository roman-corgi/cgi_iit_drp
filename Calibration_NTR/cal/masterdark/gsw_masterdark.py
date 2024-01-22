"""
Function to assemble a master dark frame from calibrated subcomponents
"""

import cal.util.check as check

CLEANROW = 1024
CLEANCOL = 1024

def build_dark(F, D, C, g, t):
    """
    Assemble a master dark from individual noise components

    This is done this way because the actual dark frame varies with gain and
    exposure time, and both of these values may vary over orders of magnitude
    in the course of acquisition, alignment, and HOWFSC.  Better to take data
    sets that don't vary

    Output is a bias-subtracted, gain-divided master dark in electrons,
    consistent with its use in the 'Process N Frames' activity diagram in
    D-105766 (AAC FDD).  (Bias is inherently subtracted as we don't use it as
    one of the building blocks to assemble the dark frame.)

    M = (F + g*t*D + g*C)/g = F/g + t*D + C

    Arguments:
     F: 1024x1024 array of floats.  This is a per-pixel map of fixed-pattern
      noise in electrons.  There are no constraints on value (may be positive
      or negative, similar to read noise)
     D: 1024x1024 array of floats.  This is a per-pixel map of dark current
      noise in electrons per second.  Each array element should be >= 0.
     C: 1024x1024 array of floats.  This is a per-pixel map of EXCAM clock-
      induced charge in electrons.  Each array element should be >= 0.
     g: current EXCAM gain, >= 1.  Unitless.
     t: current exposure time in seconds.  >= 0.

    Returns:
     1024x1024 array of floats.  This is the master dark frame in electrons.

    """
    # Check inputs
    check.twoD_array(F, 'F', TypeError)
    check.twoD_array(D, 'D', TypeError)
    check.twoD_array(C, 'C', TypeError)
    check.real_scalar(g, 'g', TypeError)
    check.real_nonnegative_scalar(t, 't', TypeError)

    if F.shape != (CLEANROW, CLEANCOL):
        raise TypeError('F must be ' + str(CLEANROW) + 'x' + str(CLEANCOL))
    if D.shape != (CLEANROW, CLEANCOL):
        raise TypeError('D must be ' + str(CLEANROW) + 'x' + str(CLEANCOL))
    if C.shape != (CLEANROW, CLEANCOL):
        raise TypeError('C must be ' + str(CLEANROW) + 'x' + str(CLEANCOL))

    if (D < 0).any():
        raise TypeError('All elements of D must be >= 0')
    if (C < 0).any():
        raise TypeError('All elements of C must be >= 0')
    if g < 1:
        raise TypeError('Gain must be a value >= 1.')

    # actual computation
    return F/g + t*D + C
