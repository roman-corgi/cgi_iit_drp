"""Functions to generate 2-D geometric shapes."""
import numpy as np

from . import check


def circle(nx, ny, roiRadiusPix, xShear, yShear, nSubpixels=100):
    """
    Generate a circular aperture with an antialiased edge at specified offsets.

    Used as a software window for isolating a region of interest. Grayscale
    edges are used because the detector sampling may be low enough that
    fractional values along the edges are important.

    Parameters
    ----------
    nx, ny : array_like
        Dimensions of the 2-D array to create.
    roiRadiusPix : float
        Radius of the circle in pixels.
    xShear, yShear : float
        Lateral offsets in pixels of the circle's center from the array's
        center pixel.
    nSubpixels : int, optional
        Each edge pixel of the circle is subdivided into a square subarray
        nSubpixels across. The subarray is given binary values and then
        averaged to give the edge pixel a value between 0 and 1, inclusive.
        The default value is 100. Must be a positive scalar integer.

    Returns
    -------
    mask : numpy ndarray
        2-D array containing the circle
    """
    check.positive_scalar_integer(nx, 'nx', TypeError)
    check.positive_scalar_integer(ny, 'ny', TypeError)
    check.real_positive_scalar(roiRadiusPix, 'roiRadiusPix', TypeError)
    check.real_scalar(xShear, 'xShear', TypeError)
    check.real_scalar(yShear, 'yShear', TypeError)
    check.positive_scalar_integer(nSubpixels, 'nSubpixels', TypeError)

    if nx % 2 == 0:
        x = np.linspace(-nx/2., nx/2. - 1, nx) - xShear
    elif nx % 2 == 1:
        x = np.linspace(-(nx-1)/2., (nx-1)/2., nx) - xShear

    if ny % 2 == 0:
        y = np.linspace(-ny/2., ny/2. - 1, ny) - yShear
    elif ny % 2 == 1:
        y = np.linspace(-(ny-1)/2., (ny-1)/2., ny) - yShear

    dx = x[1] - x[0]
    [X, Y] = np.meshgrid(x, y)
    RHO = np.sqrt(X*X + Y*Y)

    halfWindowWidth = np.sqrt(2.)*dx
    mask = -1*np.ones(RHO.shape)
    mask[np.abs(RHO) < roiRadiusPix - halfWindowWidth] = 1
    mask[np.abs(RHO) > roiRadiusPix + halfWindowWidth] = 0
    grayInds = np.array(np.nonzero(mask == -1))
    # print('Number of grayscale points = %d' % grayInds.shape[1])

    dxHighRes = 1./float(nSubpixels)
    xUp = np.linspace(-(nSubpixels-1)/2., (nSubpixels-1)/2.,
                      nSubpixels)*dxHighRes
    [Xup, Yup] = np.meshgrid(xUp, xUp)

    subpixelArray = np.zeros((nSubpixels, nSubpixels))
    # plt.figure(); plt.imshow(RHO); plt.colorbar(); plt.pause(0.1)

    # Compute the value between 0 and 1 of each edge pixel along the circle by
    # taking the mean of the binary subpixels.
    for iInterior in range(grayInds.shape[1]):

        subpixelArray = 0*subpixelArray

        xCenter = X[grayInds[0, iInterior], grayInds[1, iInterior]]
        yCenter = Y[grayInds[0, iInterior], grayInds[1, iInterior]]
        RHOHighRes = np.sqrt((Xup+xCenter)**2 + (Yup+yCenter)**2)
        # plt.figure(); plt.imshow(RHOHighRes); plt.colorbar(); plt.pause(1/20)

        subpixelArray[RHOHighRes <= roiRadiusPix] = 1
        pixelValue = np.sum(subpixelArray)/float(nSubpixels*nSubpixels)
        mask[grayInds[0, iInterior], grayInds[1, iInterior]] = pixelValue

    return mask


def ellipse(nx, ny, rx, ry, rot, xOffset, yOffset, nSubpixels=100,
            isDark=False):
    """
    Generate a rotated, laterally shifted ellipse with antialiased edges.

    Parameters
    ----------
    nx, ny : array_like
        Dimensions of the 2-D array to create.
    rx, ry : float
        x- and y- radii of the ellipse in pixels.
    rot : float
        Counterclockwise rotation of the ellipse in degrees.
    xOffset, yOffset : float
        Lateral offsets in pixels of the circle's center from the array's
        center pixel.
    nSubpixels : int, optional
        Each edge pixel of the ellipse is subdivided into a square subarray
        nSubpixels across. The subarray is given binary values and then
        averaged to give the edge pixel a value between 0 and 1, inclusive.
        The default value is 100. Must be a positive scalar integer.
    isDark : bool
        Flag whether to change the rectangle from being an illuminated region
        to a dark region.

    Returns
    -------
    mask : numpy ndarray
        2-D array containing the ellipse
    """
    check.positive_scalar_integer(nx, 'nx', TypeError)
    check.positive_scalar_integer(ny, 'ny', TypeError)
    check.real_positive_scalar(rx, 'rx', TypeError)
    check.real_positive_scalar(ry, 'ry', TypeError)
    check.real_scalar(rot, 'rot', TypeError)
    check.real_scalar(xOffset, 'xOffset', TypeError)
    check.real_scalar(yOffset, 'yOffset', TypeError)
    check.positive_scalar_integer(nSubpixels, 'nSubpixels', TypeError)

    rotRad = (np.pi/180.) * rot

    if nx % 2 == 0:
        x = np.linspace(-nx/2., nx/2. - 1, nx) - xOffset
    elif nx % 2 == 1:
        x = np.linspace(-(nx-1)/2., (nx-1)/2., nx) - xOffset

    if ny % 2 == 0:
        y = np.linspace(-ny/2., ny/2. - 1, ny) - yOffset
    elif ny % 2 == 1:
        y = np.linspace(-(ny-1)/2., (ny-1)/2., ny) - yOffset

    [X, Y] = np.meshgrid(x, y)
    dx = x[1] - x[0]
    radius = 0.5

    RHO = 0.5*np.sqrt(
        1/(rx)**2*(np.cos(rotRad)*X + np.sin(rotRad)*Y)**2
        + 1/(ry)**2*(np.sin(rotRad)*X - np.cos(rotRad)*Y)**2
    )

    halfWindowWidth = np.max(np.abs((RHO[1, 0] - RHO[0, 0],
                                     RHO[0, 1] - RHO[0, 0])))
    mask = -1*np.ones(RHO.shape)
    mask[np.abs(RHO) < radius - halfWindowWidth] = 1
    mask[np.abs(RHO) > radius + halfWindowWidth] = 0
    grayInds = np.array(np.nonzero(mask == -1))
    # print('Number of grayscale points = %d' % grayInds.shape[1])

    dxUp = dx/float(nSubpixels)
    xUp = np.linspace(-(nSubpixels-1)/2., (nSubpixels-1)/2., nSubpixels)*dxUp
    [Xup, Yup] = np.meshgrid(xUp, xUp)

    subpixel = np.zeros((nSubpixels, nSubpixels))

    for iInterior in range(grayInds.shape[1]):

        subpixel = 0*subpixel

        xCenter = X[grayInds[0, iInterior], grayInds[1, iInterior]]
        yCenter = Y[grayInds[0, iInterior], grayInds[1, iInterior]]
        RHOup = 0.5*np.sqrt(
            1/(rx)**2*(np.cos(rotRad)*(Xup+xCenter) +
                       np.sin(rotRad)*(Yup+yCenter))**2
            + 1/(ry)**2*(np.sin(rotRad)*(Xup+xCenter) -
                         np.cos(rotRad)*(Yup+yCenter))**2)

        subpixel[RHOup <= radius] = 1
        pixelValue = np.sum(subpixel)/float(nSubpixels**2)
        mask[grayInds[0, iInterior], grayInds[1, iInterior]] = pixelValue

    if isDark:
        mask = 1.0 - mask

    return mask
