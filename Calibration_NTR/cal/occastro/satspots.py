"""
TDD FDD v2.0:
Given 1) a clean occulted focal-plane image with a base DM setting,
2) a clean occulted focal-plane image with a relative satellite-spot DM setting added,
and 3) a clean occulted focal-plane image wiht the same relative DM setting
satellite-spot subtracted, taken with a given CFAM filter,
the CTC GSW shall compute the location of central star and of the spots,
in units of EXCAM pixels.
"""

import numpy as np

from cal.util import check
from cal.util.loadyaml import loadyaml

from cal.occastro.occastro import (
    calc_star_location_from_spots,
    calc_spot_separation,
)

def calc_sat_spots(img_ref, img_plus, img_minus,
                   xOffsetGuess, yOffsetGuess, thetaOffsetGuess,
                   fn_offset_YAML, fn_separation_YAML):
    """
    1. calc star location
    2. calc spot separation
    3. spot locations are r*cos(theta) + xoff, r*sin(theta)+yoff, where
       r = half spot separation
       xoff, yoff = star location
       theta as given for each spot pair + thetaOffsetGuess

    Parameters
    ----------
    img_ref : numpy ndarray
        2-D image = "a clean occulted focal-plane image with a base DM setting"
    img_plus : numpy ndarray
        2-D image = "a clean occulted focal-plane image with a relative
                     satellite-spot DM setting added"
    img_minus : numpy ndarray
        2-D image = "a clean occulted focal-plane image wiht the same relative
                     DM setting satellite-spot subtracted"

    xOffsetGuess, yOffsetGuess : float
        Starting guess for the number of pixels in x and y that the star is
        offset from the center pixel of the spots image. The convention
        for the center pixel follows that of FFTs.

    thetaOffsetGuess : float (degrees)
        theta rotation of spot locations on camera might be different from
        expected because of clocking error between the DM and the camera. Such
        clocking angle is measured with DM registration.

    fn_offset_YAML : str
        Name of the YAML file containing the tuning parameters for offset
        estimation.

    fn_separation_YAML : str
        Name of the YAML file containing the tuning parameters for separation
        estimation.

    Returns
    -------
    star_xy = [xOffsetEst, yOffsetEst] : list of floats
        Estimated lateral offsets of the stellar center from the center pixel
        of the spots image. The convention for the center pixel follows
        that of FFTs.

    list_spots_xy = [[x_0, y_0], [x_1, y_1], ...] : list of x,y duples floats


    Notes
    -----
    Offset Tuning parameters in the offset YAML file are explained below:

    spotSepPix : float
        Expected (i.e., model-based) separation of the satellite spots from the
        star. Used as the starting point for the separation for the center of
        the region of interest. Units of pixels. Compute beforehand as
        separation in lambda/D multiplied by pixels per lambda/D.
        6.5*(51.46*0.575/13)
    roiRadiusPix : float
        Radius of each region of interest used when summing the intensity of a
        satellite spot. Units of pixels.
    probeRotVecDeg : array_like
        1-D array of how many degrees counterclockwise from the x-axis to
        rotate the regions of interest used when summing the satellite spots.
        Note that a pair of satellite spots is given by just one value. For
        example, for a single pair of satellite spots along the x-axis use
        [0, ] and not [0, 180]. And for a plus-shaped layout of spots,
        use [0, 90].
    nSubpixels : int
        Number of subpixels across used to make edge values of the region-of-
        interest mask. The value of the edge pixels in the ROI is the mean of
        all the subpixel values.
    nSteps : int
        Number of points used along each direction for the grid search.
        Odd numbers are better to provide symmetry of values when the array is
        truly centered.
    stepSize : float
        The step size used in the grid search. Units of pixels.
    nIter : int
        Number of iterations in the loop that hones in on the radial separation
        of the satellite spots.

    Separation Tuning parameters in the separation YAML file are explained
    below:

    spotSepGuessPix : float
        Expected (i.e., model-based) separation of the satellite spots from the
        star. Used as the starting point for the separation for the center of
        the region of interest. Units of pixels. Compute beforehand as
        separation in lambda/D multiplied by pixels per lambda/D.
        6.5*(51.46*0.575/13)
    roiRadiusPix : float
        Radius of each region of interest used when summing the intensity of a
        satellite spot. Units of pixels.
    probeRotVecDeg : array_like
        1-D array of how many degrees counterclockwise from the x-axis to
        rotate the regions of interest used when summing the satellite spots.
        Note that a pair of satellite spots is given by just one value. For
        example, for a single pair of satellite spots along the x-axis use
        [0, ] and not [0, 180]. And for a plus-shaped layout of spots,
        use [0, 90].
    nSubpixels : int
        Number of subpixels across used to make edge values of the region-of-
        interest mask. The value of the edge pixels in the ROI is the mean of
        all the subpixel values.
    nSteps : int
        Number of points used along each direction for the grid search.
        Odd numbers are better to provide symmetry of values when the array is
        truly centered.
    stepSize : float
        The step size used in the grid search. Units of pixels.
    nIter : int
        Number of iterations in the loop that hones in on the radial separation
        of the satellite spots.

    """

    # check inputs
    # img_ref, img_plus, img_minus, xOffsetGuess, yOffsetGuess,
    #                fn_offset_YAML, fn_separation_YAML):
    # check img_ref is 2-d and real
    check.real_array(img_ref, 'img_ref', TypeError)
    check.twoD_array(img_ref, 'img_ref', TypeError)
    img_shp = img_ref.shape
    # check img_plus and img_minus are just like img_ref
    check.real_array(img_plus, 'img_plus', TypeError)
    check.twoD_array(img_plus, 'img_plus', TypeError)
    if img_plus.shape != img_shp:
        raise TypeError('img_plus not same shape as img_ref')
    check.real_array(img_minus, 'img_minus', TypeError)
    check.twoD_array(img_minus, 'img_minus', TypeError)
    if img_minus.shape != img_shp:
        raise TypeError('img_minus not same shape as img_ref')
    # check offset guess
    check.real_scalar(xOffsetGuess, 'xOffsetGuess', TypeError)
    check.real_scalar(yOffsetGuess, 'yOffsetGuess', TypeError)
    check.real_scalar(thetaOffsetGuess, 'thetaOffsetGuess', TypeError)

    # for fn_offset_YAML and fn_separation_YAML, rely on file checking in
    # the occastro routines, which use util.loadyaml()

    # combine input images to create image with satellite spots
    img_spots = 0.5*(img_plus + img_minus) - img_ref

    # estimate star location
    # xOffsetEst, yOffsetEst are relative to image center, fft style
    xOffsetEst, yOffsetEst = calc_star_location_from_spots(
        img_spots, xOffsetGuess, yOffsetGuess, fn_offset_YAML)

    # estimate spot separation (from the star, i.e. radius)
    spotRadiusEst = calc_spot_separation(img_spots,
                                         xOffsetEst,
                                         yOffsetEst,
                                         fn_separation_YAML)

    # get spot theta values from input YAML
    tuningParamDict = loadyaml(fn_separation_YAML)
    probeRotVecDeg = tuningParamDict['probeRotVecDeg']
    check.oneD_array(probeRotVecDeg, 'probeRotVecDeg',
                     TypeError) # check is redundant

    # calculate locations of spot pairs
    list_spots_xy = []
    for theta_deg in probeRotVecDeg:
        theta = (theta_deg + thetaOffsetGuess) * np.pi / 180.0
        list_spots_xy.append(
            [spotRadiusEst*np.cos(theta)+xOffsetEst,
             spotRadiusEst*np.sin(theta)+yOffsetEst],
        )
        list_spots_xy.append(
            [spotRadiusEst*np.cos(theta+np.pi)+xOffsetEst,
             spotRadiusEst*np.sin(theta+np.pi)+yOffsetEst],
        )

    star_xy = [xOffsetEst, yOffsetEst]

    return star_xy, list_spots_xy
