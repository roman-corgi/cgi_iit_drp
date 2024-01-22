"""Functions to generate 2-D geometric shapes."""
import numpy as np

import cv2
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import binary_dilation, binary_erosion

from cal.ampthresh.ampthresh import ampthresh
from . import check
from .insertinto import insertinto as inin


def fit_circle(image, min_radius, max_radius, edge_sigma=1.0, nBin=21):
    """
    Fit a circle to an image using a hough transform.

    Acts as a wrapper around skimage.feature.canny(), which detects edges, and
    skimage.transform.hough_circle(). The circle fitting algorithm only fits
    radii of integer values and returns integer center locations.

    Parameters
    ----------
    image : array_like
        2-D, real-valued array containing the image to fit.
    min_radius : int
        Minimum allowed radius with which to fit a circle. Must be an integer
        because that is what hough_circle() requires. Units of pixels.
    max_radius : int
        Maximum allowed radius with which to fit a circle. Must be an integer
        because that is what hough_circle() requires. Units of pixels.
    edge_sigma : float, optional
        Gaussian filter width used in the canny edge detection algorithm.
        Larger values accommodate more noise along the edges, but too large of
        a value allows pure noise to be fitted with a circle.
        Must be >0. The default is 1.0. Units of pixels.
        Refer to this website for more info:
        https://scikit-image.org/docs/stable/auto_examples/edges/plot_canny.html
    nBin : int
        Number of bins used when making a histogram of image values. Used
        by ampthresh(). Default is 21.

    Returns
    -------
    xOffsetEst, yOffsetEst : int
        Estimated offsets of the fitted circle's center from the array's center
        pixel. Units of pixels.
    radiusEst : int
        Estimated radius of the fitted circle. Units of pixels.

    """
    check.twoD_array(image, 'image', TypeError)
    check.real_array(image, 'image', TypeError)
    check.positive_scalar_integer(min_radius, 'min_radius', TypeError)
    check.positive_scalar_integer(max_radius, 'max_radius', TypeError)
    check.real_positive_scalar(edge_sigma, 'edge_sigma', TypeError)
    if not max_radius > min_radius:
        raise ValueError('max_radius must be larger than min_radius.')
    check.positive_scalar_integer(nBin, 'nBin', TypeError)

    # Detect edges in a binary image
    # The low_threshold and high_threshold values are hard-coded because
    # they only need to be between 0 and 1 when fitting a thresholded image
    # of all zeros and ones.
    image = ampthresh(image, nBin=nBin)
    edges = canny(image, sigma=edge_sigma, low_threshold=0.2,
                  high_threshold=0.8)

    # Detect one radius from the range of allowed values.
    # hough_circle() only allows integer radii values.
    hough_radii = np.arange(min_radius, max_radius+1, dtype=int)
    hough_res = hough_circle(edges, hough_radii)

    # Select the most prominent 1 circle
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                               total_num_peaks=1)

    radiusEst = radii[0]
    xCenterEst = cx[0]
    yCenterEst = cy[0]

    xOffsetEst = xCenterEst - image.shape[1]//2
    yOffsetEst = yCenterEst - image.shape[0]//2

    return xOffsetEst, yOffsetEst, radiusEst


def fit_ellipse(pupil_image, n_iter_dilate_erode=10, pad_factor=2, nBin=21):
    """
    Fit an ellipse to a pupil image using opencv's fitEllipse.

    Parameters
    ----------
    pupil_image : array_like
        2-D input array containing the ellipse to fit.
    n_iter_dilate_erode : int
        Number of iterations to perform for binary_dilation() and
        binary_erosion() from scipy.ndimage.morphology. Helps to fill
        in gaps such as from struts. Default is 10.
    pad_factor : float
        Factor by which to zero pad the pupil image before dilating
        and eroding. Needs to be large enough such that the dilation
        and erosion do not hit the edge of the array and fail.
        Default is 2.0.
    nBin : int
        Number of bins used when making a histogram of image values. Used
        by ampthresh(). Default is 21.

    Returns
    -------
    diamEst : float
        major diameter of the ellipse fitted to the pupil. Units of pixels.
    xOffsetEst, yOffsetEst : float
        x- and y-offsets of the fitted ellipse compared to the center pixel of
        the array. Units of pixels.

    """
    check.twoD_array(pupil_image, 'pupil_image', TypeError)
    check.real_array(pupil_image, 'pupil_image', TypeError)
    check.nonnegative_scalar_integer(n_iter_dilate_erode,
                                     'n_iter_dilate_erode',
                                     TypeError)
    check.real_positive_scalar(pad_factor, 'pad_factor', TypeError)
    check.positive_scalar_integer(nBin, 'nBin', TypeError)

    nMax = int(np.ceil(pad_factor*max(pupil_image.shape)))
    pupil_image = inin(pupil_image, (nMax, nMax))
    pupil_image = ampthresh(pupil_image, nBin=21).astype(float)

    # Dilate then erode the pupil to fill in the struts
    if n_iter_dilate_erode == 0:
        pupil_out = pupil_image
    else:
        struct = generate_binary_structure(2, 2)
        pupilDilated = binary_dilation(pupil_image, structure=struct,
                                       iterations=n_iter_dilate_erode)
        pupil_out = binary_erosion(pupilDilated, structure=struct,
                                   iterations=n_iter_dilate_erode)

    # Fit an ellipse to the pupil
    # Where (yc, xc) is the center, (a, b) the major and minor axes,
    # respectively.
    pupil_out = np.array(pupil_out, dtype=np.uint8)
    contours, hierarchy = cv2.findContours(pupil_out, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    ellipse = cv2.fitEllipse(cnt)

    xc = ellipse[0][0]
    yc = ellipse[0][1]
    a = ellipse[1][0]/2
    b = ellipse[1][1]/2
    # theta = ellipse[2]

    diamEst = 2*np.max([a, b])
    xCenterEst = xc
    yCenterEst = yc

    xOffsetEst = xCenterEst - pupil_out.shape[1]//2
    yOffsetEst = yCenterEst - pupil_out.shape[0]//2

    return diamEst, xOffsetEst, yOffsetEst
