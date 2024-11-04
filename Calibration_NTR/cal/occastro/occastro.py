"""Functions to locate the occulted stellar center based on satellite spots."""
import numpy as np
from scipy.optimize import minimize

from cal.util import check, findopt, shapes
from cal.util.loadyaml import loadyaml


def _roi_mask_for_spots(optim_params, fixed_params):
    """
    Compute the stellar offset and spot separation simultaneously.

    Parameters
    ----------
    optim_params : list
        List of the parameters to be optimized by scipy.optimize.minimize().
        They must be these 3 variables in this exact order:
            [xOffsetGuess, yOffsetGuess, spotSepGuessPix]
            xOffsetGuess, yOffsetGuess are the starting guesses for the star
            offsets from the array center pixel. Units of pixels
            spotSepGuessPix is the starting guess for the spot separation from
            the star center in units of pixels.
    fixed_params : list
        List of the fixed parameters used when computing the cost function.
        They must be these 4 variables in this exact order:
            [spotArray, probeRotVecDeg, roiRadiusPix, nSubpixels]
            spotArray is the 2-D array of the DM-generated satellite spots.
            Calculated outside this function from probed images as
            (Iplus + Iminus)/2 - Iunprobed.
            probeRotVecDeg is the 1-D array of how many degrees
            counterclockwise from the x-axis to rotate the regions of interest
            used when summing the satellite spots. Note that a pair of
            satellite spots is given by just one value. For example, for a
            single pair of satellite spots along the x-axis use [0, ] and not 
            [0, 180]. And for a plus-shaped layout of spots, use [0, 90].
            roiRadiusPix is the radius of each region of interest used when
            summing the intensity of a satellite spot. Units of pixels.
            nSubpixels is the number of subpixels across used to make edge
            values of the region-of-interest mask. The value of the edge
            pixels in the ROI is the mean of all the subpixel values.

    Returns
    -------
    cost : float
        The summed total intensity getting through the region-of-interest mask
        multiplied by -1. This is used as the cost to be minimized by
        scipy.optimize.minimize()..
    roi_mask : numpy ndarray
        2-D array with values between 0 and 1, inclusive, telling how much
        to weight the intensity in each pixel in the summed cost.
    """
    # Unpack and check the input list of optimization variables
    try:
        xOffsetGuess, yOffsetGuess, spotSepGuessPix = optim_params
    except:
        raise ValueError('optim_params must be an iterable of length 3')
    check.real_scalar(xOffsetGuess, 'xOffsetGuess', TypeError)
    check.real_scalar(yOffsetGuess, 'yOffsetGuess', TypeError)
    check.real_positive_scalar(spotSepGuessPix, 'spotSepGuessPix', TypeError)

    # Unpack and check the values in the list of fixed inputs.
    try:
        spotArray, probeRotVecDeg, roiRadiusPix, nSubpixels = fixed_params
    except:
        raise ValueError('fixed_params must be an iterable of length 4')
    check.twoD_array(spotArray, 'spotArray', TypeError)
    check.oneD_array(probeRotVecDeg, 'probeRotVecDeg', TypeError)
    check.real_positive_scalar(roiRadiusPix, 'roiRadiusPix', TypeError)
    check.positive_scalar_integer(nSubpixels, 'nSubpixels', TypeError)

    # Generate mask to isolate the satellite spots in spotArray
    ny, nx = spotArray.shape
    roi_mask = np.zeros((ny, nx))
    for iRot, rotDeg in enumerate(probeRotVecDeg):
        rotRad = np.radians(rotDeg)
        xProbePlusCoord = np.array([
            np.sin(rotRad)*spotSepGuessPix + yOffsetGuess,
            np.cos(rotRad)*spotSepGuessPix + xOffsetGuess])
        rotRad += np.pi
        xProbeMinusCoord = np.array([
            np.sin(rotRad)*spotSepGuessPix + yOffsetGuess,
            np.cos(rotRad)*spotSepGuessPix + xOffsetGuess])
        
        roi_mask += shapes.circle(nx, ny, roiRadiusPix,
                                  xProbePlusCoord[1],
                                  xProbePlusCoord[0],
                                  nSubpixels=nSubpixels)
        roi_mask += shapes.circle(nx, ny, roiRadiusPix,
                                  xProbeMinusCoord[1],
                                  xProbeMinusCoord[0],
                                  nSubpixels=nSubpixels)

    # Cost is negative because scipy minimizes the cost function
    # but we want to maximize the sum.
    cost = -np.sum(roi_mask * spotArray)

    return cost, roi_mask

def _cost_func_spots(optim_params, fixed_params):
    """
    Return only the cost value for scipy.optimize.minimize().
    
    This is a thin wrapper for occastro.roi_mask_for_spots() because
    scipy.optimize.minimize() requires the given function to return only
    the cost function value but we want some other diagnostic outputs from
    occastro.roi_mask_for_spots() as well.
    """
    cost, _ = _roi_mask_for_spots(optim_params, fixed_params)

    return cost


def calc_star_location_and_spot_separation(
        spotArray, xOffsetGuess, yOffsetGuess, fnTuning):
    """
    Calculate the center of the occulted star using satellite spots.

    Just one processed image of satellite spots is used. A multitude of
    software masks are generated and applied to the measured spots in order to
    determine the stellar location. The best estimate of the star location is
    the one that maximizes the total summed energy, as found by a 2-D quadratic
    fit.

    All filenames may be absolute or relative paths.  If relative, they will be
    relative to the current working directory, not to any particular location
    in Calibration.

    Parameters
    ----------
    spotArray : numpy ndarray
        2-D array of the DM-generated satellite spots. Calculated outside
        this function from probed images as (Iplus + Iminus)/2 - Iunprobed.
    xOffsetGuess, yOffsetGuess : float
        Starting guess for the number of pixels in x and y that the star is
        offset from the center pixel of the array spotArray. The convention
        for the center pixel follows that of FFTs.
    fnTuning : str
        Name of the YAML file containing the tuning parameter values.

    Returns
    -------
    xOffsetEst, yOffsetEst : float
        Estimated lateral offsets of the stellar center from the center pixel
        of the array spotArray. The convention for the center pixel follows
        that of FFTs.
    roi_mask : numpy ndarray
        2-D float array of the best-fit region-of-interest mask used to fit
        the translation and scaling of the ROI regions to spotArray.

    Notes
    -----
    Tuning parameters in the YAML file are explained below:

    spotSepPix : float
        Expected separation of the satellite spots from the star. Used as the
        separation for the center of the region of interest. Units of pixels.
        Compute beforehand as sep in lambda/D and multiply by pix per lambda/D.
    roiRadiusPix : float
        Radius of each region of interest used when summing the intensity of a
        satellite spot. Units of pixels.
    probeRotVecDeg : array_like
        1-D array of how many degrees counterclockwise from the x-axis to
        rotate the regions of interest used when summing the satellite spots.
        Note that a pair of satellite spots is given by just one value. For
        example, for a single pair of satellite spots along the x-axis use [0,]
        and [0, 180]. And for a plus-shaped layout of spots, use [0, 90].
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
        Number of iterations in the loop that hones in on the stellar center
        location.
    """
    check.twoD_array(spotArray, 'spotArray', TypeError)
    check.real_scalar(xOffsetGuess, 'xOffsetGuess', TypeError)
    check.real_scalar(yOffsetGuess, 'yOffsetGuess', TypeError)
    check.string(fnTuning, 'fnTuning', TypeError)

    tuningParamDict = loadyaml(fnTuning)
    spotSepGuessPix = tuningParamDict['spotSepGuessPix']
    roiRadiusPix = tuningParamDict['roiRadiusPix']
    probeRotVecDeg = tuningParamDict['probeRotVecDeg']
    nSubpixels = tuningParamDict['nSubpixels']
    offset_tol_pix = tuningParamDict['offset_tol_pix']
    sep_tol_pix = tuningParamDict['sep_tol_pix']
    opt_method = tuningParamDict['opt_method']

    check.real_positive_scalar(spotSepGuessPix, 'spotSepGuessPix', TypeError)
    check.real_positive_scalar(roiRadiusPix, 'roiRadiusPix', TypeError)
    check.oneD_array(probeRotVecDeg, 'probeRotVecDeg', TypeError)
    check.positive_scalar_integer(nSubpixels, 'nSubpixels', TypeError)

    # Define optimization input lists
    bounds = [
        [xOffsetGuess-offset_tol_pix, xOffsetGuess+offset_tol_pix],
        [yOffsetGuess-offset_tol_pix, yOffsetGuess+offset_tol_pix],
        [spotSepGuessPix-sep_tol_pix, spotSepGuessPix+sep_tol_pix],
    ]
    initial_guess = [xOffsetGuess, yOffsetGuess, spotSepGuessPix]
    fixed_params = [spotArray, probeRotVecDeg, roiRadiusPix, nSubpixels]

    # Run the optimization
    result = minimize(_cost_func_spots,
                      initial_guess,
                      args=fixed_params,
                      bounds=bounds,
                      method=opt_method,
                      )
    if result.success:
        fitted_params = result.x
    else:
        raise ValueError(result.message)

    param_dict = {
        'xOffset': fitted_params[0],
        'yOffset': fitted_params[1],
        'spotSepPix': fitted_params[2],
    }

    _, roi_mask = _roi_mask_for_spots(fitted_params, fixed_params)

    return param_dict, roi_mask


def calc_star_location_from_spots(spotArray, xOffsetGuess, yOffsetGuess,
                                  fnYAML):
    """
    Calculate the center of the occulted star using satellite spots.

    Just one processed image of satellite spots is used. A multitude of
    software masks are generated and applied to the measured spots in order to
    determine the stellar location. The best estimate of the star location is
    the one that maximizes the total summed energy, as found by a 2-D quadratic
    fit.

    All filenames may be absolute or relative paths.  If relative, they will be
    relative to the current working directory, not to any particular location
    in Calibration.

    Parameters
    ----------
    spotArray : numpy ndarray
        2-D array of the DM-generated satellite spots. Calculated outside
        this function from probed images as (Iplus + Iminus)/2 - Iunprobed.
    xOffsetGuess, yOffsetGuess : float
        Starting guess for the number of pixels in x and y that the star is
        offset from the center pixel of the array spotArray. The convention
        for the center pixel follows that of FFTs.
    fnYAML : str
        Name of the YAML file containing the tuning parameter values.

    Returns
    -------
    xOffsetEst, yOffsetEst : float
        Estimated lateral offsets of the stellar center from the center pixel
        of the array spotArray. The convention for the center pixel follows
        that of FFTs.

    Notes
    -----
    Tuning parameters in the YAML file are explained below:

    spotSepPix : float
        Expected separation of the satellite spots from the star. Used as the
        separation for the center of the region of interest. Units of pixels.
        Compute beforehand as sep in lambda/D and multiply by pix per lambda/D.
    roiRadiusPix : float
        Radius of each region of interest used when summing the intensity of a
        satellite spot. Units of pixels.
    probeRotVecDeg : array_like
        1-D array of how many degrees counterclockwise from the x-axis to
        rotate the regions of interest used when summing the satellite spots.
        Note that a pair of satellite spots is given by just one value. For
        example, for a single pair of satellite spots along the x-axis use [0,]
        and [0, 180]. And for a plus-shaped layout of spots, use [0, 90].
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
        Number of iterations in the loop that hones in on the stellar center
        location.
    """
    check.twoD_array(spotArray, 'spotArray', TypeError)
    check.real_scalar(xOffsetGuess, 'xOffsetGuess', TypeError)
    check.real_scalar(yOffsetGuess, 'yOffsetGuess', TypeError)

    tuningParamDict = loadyaml(fnYAML)
    spotSepPix = tuningParamDict['spotSepPix']
    roiRadiusPix = tuningParamDict['roiRadiusPix']
    probeRotVecDeg = tuningParamDict['probeRotVecDeg']
    nSubpixels = tuningParamDict['nSubpixels']
    nSteps = tuningParamDict['nSteps']
    stepSize = tuningParamDict['stepSize']
    nIter = tuningParamDict['nIter']

    check.real_positive_scalar(spotSepPix, 'spotSepPix', TypeError)
    check.real_positive_scalar(roiRadiusPix, 'roiRadiusPix', TypeError)
    check.oneD_array(probeRotVecDeg, 'probeRotVecDeg', TypeError)
    check.positive_scalar_integer(nSubpixels, 'nSubpixels', TypeError)
    check.positive_scalar_integer(nSteps, 'nSteps', TypeError)
    check.real_positive_scalar(stepSize, 'stepSize', TypeError)
    check.positive_scalar_integer(nIter, 'nIter', TypeError)

    ny, nx = spotArray.shape
    costFuncMat = np.zeros((nSteps, nSteps))
    xOffsetEst = 0
    yOffsetEst = 0

    for iter_ in range(nIter):

        xOffsetVec = (np.arange(nSteps)*stepSize -
                      (nSteps-1)/2*stepSize +
                      xOffsetEst
                      )
        yOffsetVec = (np.arange(nSteps)*stepSize -
                      (nSteps-1)/2*stepSize +
                      yOffsetEst
                      )

        for iy, yOffset in enumerate(yOffsetVec):

            for ix, xOffset in enumerate(xOffsetVec):

                # Generate mask of all ROI regions
                roi_mask = np.zeros((ny, nx))
                for iRot, rotDeg in enumerate(probeRotVecDeg):

                    rotRad = np.radians(rotDeg)
                    xProbePlusCoord = np.array([
                        np.sin(rotRad)*spotSepPix + yOffsetGuess + yOffset,
                        np.cos(rotRad)*spotSepPix + xOffsetGuess + xOffset])
                    rotRad += np.pi
                    xProbeMinusCoord = np.array([
                        np.sin(rotRad)*spotSepPix + yOffsetGuess + yOffset,
                        np.cos(rotRad)*spotSepPix + xOffsetGuess + xOffset])

                    roi_mask += shapes.circle(nx, ny, roiRadiusPix,
                                             xProbePlusCoord[1],
                                             xProbePlusCoord[0],
                                             nSubpixels=nSubpixels)
                    roi_mask += shapes.circle(nx, ny, roiRadiusPix,
                                             xProbeMinusCoord[1],
                                             xProbeMinusCoord[0],
                                             nSubpixels=nSubpixels)

                costFuncMat[iy, ix] = np.sum(roi_mask * spotArray)

        xOffsetEst, yOffsetEst = findopt.find_optimum_2d(xOffsetVec,
                                                         yOffsetVec,
                                                         costFuncMat,
                                                         np.ones_like(
                                                             costFuncMat)
                                                         )

    # estimates are w.r.t. initial guess. Return, "from the center pixel"
    return xOffsetEst+xOffsetGuess, yOffsetEst+yOffsetGuess


def calc_spot_separation(spotArray, xOffset, yOffset, fnYAML):
    """
    Calculate the radial separation in pixels of the satellite spots.

    Just one processed image of satellite spots is used. Several software
    masks are generated and applied to the measured spots in order to
    determine the radial separation of the spots from the given star center.
    The best estimate of the radial spot separation is the one that maximizes
    the total summed energy in the software mask, as found by a 1-D quadratic
    fit.

    All filenames may be absolute or relative paths.  If relative, they will be
    relative to the current working directory, not to any particular location
    in Calibration.

    Parameters
    ----------
    spotArray : numpy ndarray
        2-D array of the DM-generated satellite spots. Calculated outside
        this function from probed images as (Iplus + Iminus)/2 - Iunprobed.
    xOffset, yOffset : float
        Previously estimated stellar center offset from the array's center
        pixel. Units of pixels. The convention for the center pixel follows
        that of FFTs.
    fnYAML : str
        Name of the YAML file containing the tuning parameter values.

    Returns
    -------
    spotSepEst : float
        Estimated radial separation of the satellite spots from the stellar
        center. Units of pixels.

    Notes
    -----
    Tuning parameters in the YAML file are explained below:

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
    check.twoD_array(spotArray, 'spotArray', TypeError)
    check.real_scalar(xOffset, 'xOffset', TypeError)
    check.real_scalar(yOffset, 'yOffset', TypeError)

    tuningParamDict = loadyaml(fnYAML)
    spotSepGuessPix = tuningParamDict['spotSepGuessPix']
    roiRadiusPix = tuningParamDict['roiRadiusPix']
    probeRotVecDeg = tuningParamDict['probeRotVecDeg']
    nSubpixels = tuningParamDict['nSubpixels']
    nSteps = tuningParamDict['nSteps']
    stepSize = tuningParamDict['stepSize']
    nIter = tuningParamDict['nIter']

    check.real_positive_scalar(spotSepGuessPix, 'spotSepGuessPix', TypeError)
    check.real_positive_scalar(roiRadiusPix, 'roiRadiusPix', TypeError)
    check.oneD_array(probeRotVecDeg, 'probeRotVecDeg', TypeError)
    check.positive_scalar_integer(nSubpixels, 'nSubpixels', TypeError)
    check.positive_scalar_integer(nSteps, 'nSteps', TypeError)
    check.real_positive_scalar(stepSize, 'stepSize', TypeError)
    check.positive_scalar_integer(nIter, 'nIter', TypeError)

    ny, nx = spotArray.shape
    costFuncVec = np.zeros((nSteps, ))
    spotSepEst = spotSepGuessPix  # initialize

    for iter_ in range(nIter):

        spotSepVec = (np.arange(nSteps)*stepSize -
                      (nSteps-1)/2*stepSize + spotSepEst
                      )

        for iSep, spotSep in enumerate(spotSepVec):

            # Generate mask of all ROI regions
            roi_mask = np.zeros((ny, nx))
            for iRot, rotDeg in enumerate(probeRotVecDeg):

                rotRad = np.radians(rotDeg)
                xProbePlusCoord = np.array([
                    np.sin(rotRad)*spotSep + yOffset,
                    np.cos(rotRad)*spotSep + xOffset])
                rotRad += np.pi
                xProbeMinusCoord = np.array([
                    np.sin(rotRad)*spotSep + yOffset,
                    np.cos(rotRad)*spotSep + xOffset])

                roi_mask += shapes.circle(nx, ny, roiRadiusPix,
                                         xProbePlusCoord[1],
                                         xProbePlusCoord[0],
                                         nSubpixels=nSubpixels)
                roi_mask += shapes.circle(nx, ny, roiRadiusPix,
                                         xProbeMinusCoord[1],
                                         xProbeMinusCoord[0],
                                         nSubpixels=nSubpixels)

            costFuncVec[iSep] = np.sum(roi_mask * spotArray)

        # At the first iteration, uses the maximum instead of a
        # quadratic fit in order to get a larger capture range.
        if iter_ == 0 and nIter > 1:
            bestInd = np.argmax(costFuncVec)
            spotSepEst = spotSepVec[bestInd]
        else:
            spotSepEst = findopt.find_optimum_1d(spotSepVec, costFuncVec)

    return spotSepEst
