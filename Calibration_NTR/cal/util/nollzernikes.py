"""
Functions to manipulate Zernike polynomials
"""

import numpy as np
import scipy.special

from . import check

def xyzern(x, y, prad, orders):
    """ Evaluate points on specified Noll-ordered Zernike polynomials.

    Noll-ordering arranges the Zernike polynomials first by radial order, then
    by azimuthal order; when there are two polynomials of the same radial and
    azimuthal order (e.g. the x- and y-coma), the even-numbered one has the
    cosine angular dependence, and the odd-numbered one has sine angular
    dependence.

    This normalizes Noll Zernikes following convention in Noll 1976 rather
    than the original Zernike definition, which normalizes output coefficients
    for RMS amplitude for each polynomial.

    Commonly-used polynomials include:
    - Z1: piston
    - Z2/Z3: tip/tilt
    - Z4: focus
    - Z5/Z6: astigmatism
    - Z7/Z8: coma
    - Z9/Z10: trefoil
    - Z11: spherical

    See also: Noll, R.J., 'Zernike polynomials and atmospheric turbulence,'
    Journal of the Optical Society of America 66:3 (1976).

    Arguments:
    x : numpy ndarray
        2D array of grid points giving offsets in the x-direction. The
        pupil center is assumed to be at x = 0. Units may be anything
        (pixels, meters) as it will be normalized internally
    y : numpy ndarray
        2D array of grid points giving offsets in the y-direction. The
        pupil center is assumed to be at y = 0. Units may be anything
        (pixels, meters) as it will be normalized internally. Assumed
        to be the same size as x; this will be checked.
    prad :
        radius of pupil, in the same units as ``x`` and ``y``. Will be a
        real positive scalar.
    orders :
        a list of Noll orders, which are positive scalar integers.

    Returns:
     a 3D array, len(orders) x size(x), with each cube slice giving the Zernike
     polynomial of that order aligned to the grid specified by ``x`` and ``y``.

    """

    # Check inputs
    check.twoD_array(x, 'x', TypeError)
    check.twoD_array(y, 'y', TypeError)
    check.real_positive_scalar(prad, 'prad', TypeError)
    if x.shape != y.shape:
        raise TypeError('Input x and y must be same size')
    try:
        for index, order in enumerate(orders):
            check.positive_scalar_integer(order, 'order'+str(index), TypeError)
            pass
        pass
    except TypeError: # not iterable
        raise TypeError('orders must be an iterable')

    # Polar basis for Zernikes
    r = np.hypot(x, y)/float(prad)
    th = np.arctan2(y, x)

    # 1D view of original
    ra = np.array(r, ndmin=1).ravel()
    tha = np.array(th, ndmin=1).ravel()
    moa = np.array(orders, ndmin=1).ravel()

    nr = len(ra)
    nmo = len(moa)
    np1 = (np.ceil((np.sqrt(1.+8.*moa)-1.)/2.)+0.5).astype(int)
    #np1 is radial-mode n plus 1, calculated from imax=(np1+1)*np1/2, the
    # number of terms that have radial mode <= np1, imax=sum_i=1^np1 (i)
    ntermsn = (np1*(np1-1))//2 # num of polys with radial mode < np1
    m = ((moa - ntermsn - ((np1+1) % 2))//2)*2 + ((np1+1) % 2)
    n = np1-1
    # m,n are azimuthal and radial modes from Noll numbering scheme

    mmax = m.max()
    nmax = n.max()

    rpow = np.zeros((nmax+1, nr))+1.
    for i in range(1, nmax+1):
        rpow[i, :] = rpow[i-1, :]*ra
        pass
    sincos = np.zeros((mmax+1, 2, nr))+1.
    for i in range(1, mmax+1):
        sincos[i, 0, :] = np.cos(i*tha)
        sincos[i, 1, :] = np.sin(i*tha)
        pass

    # Implementation of the gamma-function sums in Eq. 2 in Noll 1976
    zpoly = np.zeros((nmo, nr))
    for i in range(nmo):
        rp = np.zeros(nr)
        kmax = (n[i]-m[i])//2
        rnum = np.zeros(kmax+1)
        rden = np.zeros((3, kmax+1))
        rnum[kmax] = scipy.special.gamma(n[i] - kmax + 1) * (-1)**kmax
        rden[0, 0] = 1.
        rden[1, kmax] = scipy.special.gamma((n[i]+m[i])//2 - kmax + 1)
        rden[2, kmax] = scipy.special.gamma((n[i]-m[i])//2 - kmax + 1)
        for k in range(1, kmax+1):
            rnum[kmax-k] = rnum[kmax-k+1] * -(n[i]-(kmax-k))
            rden[0, k] = rden[0, k-1]*k
            rden[1, kmax-k] = rden[1, kmax-k+1] * ((n[i]+m[i])//2 - (kmax-k))
            rden[2, kmax-k] = rden[2, kmax-k+1] * ((n[i]-m[i])//2 - (kmax-k))
            pass
        for k in range(kmax+1):
            rp += rnum[k]/rden[:, k].prod(axis=0) * rpow[n[i]-2*k, :]
            pass
        zpoly[i, :] = rp * sincos[m[i], moa[i] % 2, :]
        pass

    # Noll correction factor to make RMS amplitude match (Eq. 1)
    nc = np.array([np.sqrt(2*(n[i]+1)) if m[i] != 0 else np.sqrt(n[i]+1)
                   for i in range(nmo)])

    outarray = np.empty((len(orders), x.shape[0], x.shape[1]))
    for i in range(nmo):
        outarray[i, :, :] = nc[i]*np.reshape(zpoly[i, :], x.shape)
        pass
    return outarray
