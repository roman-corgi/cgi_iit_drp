"""Test suite for OCCASTRO module."""
from math import isclose
import unittest
import os
import numpy as np
from astropy.io import fits

from cal.occastro.occastro import (
    calc_star_location_from_spots,
    calc_spot_separation,
    calc_star_location_and_spot_separation,
    _roi_mask_for_spots,
    _cost_func_spots
)
from cal.util.loadyaml import loadyaml

not_an_iterable_of_length_3 = (1, 1.1, 1j, np.arange(5), 'string', {'a': 2})
not_an_iterable_of_length_4 = (1, 1.1, 1j, np.arange(5), 'string', {'a': 2})
not_a_1d_array = (True, -1, 1, 1.1, 1j, np.ones((5, 5)), 'string', {'a': 2})
not_a_2d_array = (True, -1, 1, 1.1, 1j, np.ones((5, )), 'string', {'a': 2})
not_a_real_scalar = (1j, 1j, np.ones(5), 'string', {'a': 2})
not_a_real_positive_scalar = (-1, 0, 1j, np.ones(5), 'string', {'a': 2})
not_a_string = (True, -1, 1, 1.1, 1j, np.ones((5, )), {'a': 2})
not_a_positive_scalar_integer = (-1, 0, 1.1, 1j, np.ones(5), np.ones((5, 5)),
                                 'string')

class TestOccastroInputFailure(unittest.TestCase):
    """Test suite for valid function inputs."""

    def test_calc_offset_from_spots_inputs(self):
        """Test the inputs of calc_star_location_from_spots."""
        localpath = os.path.dirname(os.path.abspath(__file__))
        fnTuning = os.path.join(localpath, 'testdata',
                                'occastro_offset_tuning_nfov.yaml')
        fnSpots = os.path.join(localpath, 'testdata',
                               'occastro_nfov_4spots_R6.50_no_errors.fits')
        spotArray = fits.getdata(fnSpots)

        xOffsetGuess = 0
        yOffsetGuess = 0

        # Check that standard inputs do not raise anything first
        _, _ = calc_star_location_from_spots(
            spotArray, xOffsetGuess, yOffsetGuess, fnTuning)

        # Tests of bad inputs
        # except for fnTuning which uses the tests of util.loadyaml()
        with self.assertRaises(TypeError):
            calc_star_location_from_spots(
                np.ones((10, )), xOffsetGuess, yOffsetGuess, fnTuning)
        with self.assertRaises(TypeError):
            calc_star_location_from_spots(
                spotArray, [-1.3], yOffsetGuess, fnTuning)
        with self.assertRaises(TypeError):
            calc_star_location_from_spots(
                spotArray, xOffsetGuess, [-1.3], fnTuning)

    def test_calc_spot_separation_inputs(self):
        """Test the inputs of calc_spot_separation."""
        localpath = os.path.dirname(os.path.abspath(__file__))
        fnTuning = os.path.join(localpath, 'testdata',
                                'occastro_separation_tuning_nfov.yaml')
        fnSpots = os.path.join(localpath, 'testdata',
                               'occastro_nfov_4spots_R6.50_no_errors.fits')
        spotArray = fits.getdata(fnSpots)

        xOffset = 0
        yOffset = 0

        # Check that standard inputs do not raise anything first
        _ = calc_spot_separation(spotArray, xOffset, yOffset, fnTuning)

        # Tests of bad inputs
        # except for fnTuning which uses the tests of util.loadyaml()
        with self.assertRaises(TypeError):
            calc_star_location_from_spots(
                np.ones((10, )), xOffset, yOffset, fnTuning)
        with self.assertRaises(TypeError):
            calc_star_location_from_spots(
                spotArray, [-1.3], yOffset, fnTuning)
        with self.assertRaises(TypeError):
            calc_star_location_from_spots(
                spotArray, xOffset, [-1.3], fnTuning)


class TestOccastroOffset(unittest.TestCase):
    """Integration tests of occastro's stellar offset estimation."""

    def test_occastro_offset_nfov(self):
        """Make sure NFOV astrometry is estimated to within +/- 0.1 pixels."""
        localpath = os.path.dirname(os.path.abspath(__file__))
        fnTuning = os.path.join(localpath, 'testdata',
                                'occastro_offset_tuning_nfov.yaml')
        # this example has star at center of image:
        fnSpots = os.path.join(localpath, 'testdata',
                               'occastro_nfov_4spots_R6.50_no_errors.fits')
        spotArray = fits.getdata(fnSpots)
        xOffset_truth = 0.0
        yOffset_truth = 0.0

        xOffsetGuessVec = [2, 0]
        yOffsetGuessVec = [-2, 0]
        nOffsets = len(xOffsetGuessVec)

        for iOffset in range(nOffsets):
            xOffsetGuess = xOffsetGuessVec[iOffset]
            yOffsetGuess = yOffsetGuessVec[iOffset]
            xOffsetEst, yOffsetEst = calc_star_location_from_spots(
                spotArray, xOffsetGuess, yOffsetGuess, fnTuning)

            # estimates are w.r.t. center of image array
            xErrorPix = np.abs(xOffsetEst - xOffset_truth) # + xOffsetGuess)
            yErrorPix = np.abs(yOffsetEst - yOffset_truth) # + yOffsetGuess)
            self.assertTrue(xErrorPix < 0.1)
            self.assertTrue(yErrorPix < 0.1)

    def test_occastro_offset_spec(self):
        """Make sure Spec astrometry is estimated to within +/- 0.1 pixels."""
        localpath = os.path.dirname(os.path.abspath(__file__))
        fnTuning = os.path.join(localpath, 'testdata',
                                'occastro_offset_tuning_spec.yaml')
        # this example has star at center of image
        fnSpots = os.path.join(localpath, 'testdata',
                               'occastro_spec_2spots_R6.00_no_errors.fits')
        spotArray = fits.getdata(fnSpots)
        xOffset_truth = 0.0
        yOffset_truth = 0.0

        xOffsetGuessVec = [2, 0]
        yOffsetGuessVec = [-2, 0]
        nOffsets = len(xOffsetGuessVec)

        for iOffset in range(nOffsets):
            xOffsetGuess = xOffsetGuessVec[iOffset]
            yOffsetGuess = yOffsetGuessVec[iOffset]
            xOffsetEst, yOffsetEst = calc_star_location_from_spots(
                spotArray, xOffsetGuess, yOffsetGuess, fnTuning)

            xErrorPix = np.abs(xOffsetEst - xOffset_truth) # + xOffsetGuess)
            yErrorPix = np.abs(yOffsetEst - yOffset_truth) # + yOffsetGuess)
            self.assertTrue(xErrorPix < 0.1)
            self.assertTrue(yErrorPix < 0.1)

    def test_occastro_offset_wfov(self):
        """Make sure WFOV astrometry is estimated to within +/- 0.1 pixels."""
        localpath = os.path.dirname(os.path.abspath(__file__))
        fnTuning = os.path.join(localpath, 'testdata',
                                'occastro_offset_tuning_wfov.yaml')
        fnSpots = os.path.join(localpath, 'testdata',
                               'occastro_wfov_4spots_R13.00_no_errors.fits')
        spotArray = fits.getdata(fnSpots)
        xOffset_truth = 0.0
        yOffset_truth = 0.0

        xOffsetGuessVec = [2, 0]
        yOffsetGuessVec = [-2, 0]
        nOffsets = len(xOffsetGuessVec)

        for iOffset in range(nOffsets):
            xOffsetGuess = xOffsetGuessVec[iOffset]
            yOffsetGuess = yOffsetGuessVec[iOffset]
            xOffsetEst, yOffsetEst = calc_star_location_from_spots(
                spotArray, xOffsetGuess, yOffsetGuess, fnTuning)

            xErrorPix = np.abs(xOffsetEst - xOffset_truth) # + xOffsetGuess)
            yErrorPix = np.abs(yOffsetEst - yOffset_truth) #  + yOffsetGuess)
            self.assertTrue(xErrorPix < 0.1)
            self.assertTrue(yErrorPix < 0.1)


class TestOccastroSeparation(unittest.TestCase):
    """Integration tests of occastro's spot separation estimation."""

    def test_occastro_separation_nfov(self):
        """Test that NFOV spot separation is accuracte to +/- 0.1 pixels."""
        localpath = os.path.dirname(os.path.abspath(__file__))
        fnTuning = os.path.join(localpath, 'testdata',
                                'occastro_separation_tuning_nfov.yaml')
        fnSpots = os.path.join(localpath, 'testdata',
                               'occastro_nfov_4spots_R6.50_no_errors.fits')
        spotArray = fits.getdata(fnSpots)

        xOffset = 0
        yOffset = 0
        spotSepEst = calc_spot_separation(spotArray,
                                          xOffset,
                                          yOffset,
                                          fnTuning)
        spotSepTrue = 14.79
        errorPix = np.abs(spotSepEst - spotSepTrue)
        self.assertTrue(errorPix < 0.1)

    def test_occastro_separation_spec(self):
        """Test that Spec spot separation is accuracte to +/- 0.1 pixels."""
        localpath = os.path.dirname(os.path.abspath(__file__))
        fnTuning = os.path.join(localpath, 'testdata',
                                'occastro_separation_tuning_spec.yaml')
        fnSpots = os.path.join(localpath, 'testdata',
                               'occastro_spec_2spots_R6.00_no_errors.fits')
        spotArray = fits.getdata(fnSpots)

        xOffset = 0
        yOffset = 0
        spotSepEst = calc_spot_separation(spotArray,
                                          xOffset,
                                          yOffset,
                                          fnTuning)
        spotSepTrue = 17.34
        errorPix = np.abs(spotSepEst - spotSepTrue)
        self.assertTrue(errorPix < 0.1)

    def test_occastro_separation_wfov(self):
        """Test that WFOV spot separation is accuracte to +/- 0.1 pixels."""
        localpath = os.path.dirname(os.path.abspath(__file__))
        fnTuning = os.path.join(localpath, 'testdata',
                                'occastro_separation_tuning_wfov.yaml')
        fnSpots = os.path.join(localpath, 'testdata',
                               'occastro_wfov_4spots_R13.00_no_errors.fits')
        spotArray = fits.getdata(fnSpots)

        xOffset = 0
        yOffset = 0
        spotSepEst = calc_spot_separation(spotArray,
                                          xOffset,
                                          yOffset,
                                          fnTuning)
        spotSepTrue = 42.45
        errorPix = np.abs(spotSepEst - spotSepTrue)
        self.assertTrue(errorPix < 0.1)


class TestOccastroOffsetAndSeparation(unittest.TestCase):
    """Integration tests of occastro's all-in-one estimation."""

    def setUp(self):
        localpath = os.path.dirname(os.path.abspath(__file__))
        self.fnTuning = os.path.join(
            localpath,
            'cgidata',
            'occastro_offset_and_separation_tuning_band1.yaml')
        fnSpots = os.path.join(localpath, 'testdata',
                               'occastro_nfov_4spots_R6.50_no_errors.fits')
        self.spotArray = fits.getdata(fnSpots)
        self.xOffsetGuess = -2.0
        self.yOffsetGuess = 1.9

    def test_input_0(self):
        """Test bad input type."""
        for bad_val in not_a_2d_array:
            with self.assertRaises(TypeError):
                calc_star_location_and_spot_separation(
                    bad_val,
                    self.xOffsetGuess,
                    self.yOffsetGuess,
                    self.fnTuning,
                )

    def test_input_1(self):
        """Test bad input type."""
        for bad_val in not_a_real_scalar:
            with self.assertRaises(TypeError):
                calc_star_location_and_spot_separation(
                    self.spotArray,
                    bad_val,
                    self.yOffsetGuess,
                    self.fnTuning,
                )

    def test_input_2(self):
        """Test bad input type."""
        for bad_val in not_a_real_scalar:
            with self.assertRaises(TypeError):
                calc_star_location_and_spot_separation(
                    self.spotArray,
                    self.xOffsetGuess,
                    bad_val,
                    self.fnTuning,
                )

    def test_input_3(self):
        """Test bad input type."""
        for bad_val in not_a_string:
            with self.assertRaises(TypeError):
                calc_star_location_and_spot_separation(
                    self.spotArray,
                    self.xOffsetGuess,
                    self.yOffsetGuess,
                    bad_val,
                )


    def test_occastro_nfov(self):
        """Test NFOV spot separation and star offset location."""
        xOffsetTrue = 0
        yOffsetTrue = 0
        spotSepTrue = 14.79

        param_dict, roi_mask = calc_star_location_and_spot_separation(
            self.spotArray,
            self.xOffsetGuess,
            self.yOffsetGuess,
            self.fnTuning,
        )

        xOffsetEst = param_dict['xOffset']
        yOffsetEst = param_dict['yOffset']
        spotSepEst = param_dict['spotSepPix']

        self.assertTrue(np.abs(xOffsetTrue - xOffsetEst) < 0.1)
        self.assertTrue(np.abs(yOffsetTrue - yOffsetEst) < 0.1)
        self.assertTrue(np.abs(spotSepTrue - spotSepEst) < 0.1)

    def test_occastro_separation_spec(self):
        """Test SPEC spot separation and star offset location."""
        localpath = os.path.dirname(os.path.abspath(__file__))
        fnTuning = os.path.join(
            localpath, 'cgidata',
            'occastro_offset_and_separation_tuning_band3.yaml')
        fnSpots = os.path.join(localpath, 'testdata',
                               'occastro_spec_2spots_R6.00_no_errors.fits')
        spotArray = fits.getdata(fnSpots)

        xOffsetTrue = 0
        spotSepTrue = 17.34

        xOffsetGuess = 2.0
        yOffsetGuess = -1.9

        param_dict, roi_mask = calc_star_location_and_spot_separation(
            spotArray, xOffsetGuess, yOffsetGuess, fnTuning)

        xOffsetEst = param_dict['xOffset']
        spotSepEst = param_dict['spotSepPix']

        # No y-offset test here because it doesn't do well with just the
        # x-axis pair of spots. Also a larger separation tolerance is needed,
        # of 0.2 pixels.
        self.assertTrue(np.abs(xOffsetTrue - xOffsetEst) < 0.1)
        self.assertTrue(np.abs(spotSepTrue - spotSepEst) < 0.2)

    def test_occastro_separation_wfov(self):
        """Test WFOV spot separation and star offset location."""
        localpath = os.path.dirname(os.path.abspath(__file__))
        fnTuning = os.path.join(
            localpath, 'cgidata',
            'occastro_offset_and_separation_tuning_band4.yaml')
        fnSpots = os.path.join(localpath, 'testdata',
                               'occastro_wfov_4spots_R13.00_no_errors.fits')
        spotArray = fits.getdata(fnSpots)

        xOffsetTrue = 0
        yOffsetTrue = 0
        spotSepTrue = 42.45

        xOffsetGuess = 2.0
        yOffsetGuess = -1.9

        param_dict, roi_mask = calc_star_location_and_spot_separation(
            spotArray, xOffsetGuess, yOffsetGuess, fnTuning)

        xOffsetEst = param_dict['xOffset']
        yOffsetEst = param_dict['yOffset']
        spotSepEst = param_dict['spotSepPix']

        self.assertTrue(np.abs(xOffsetTrue - xOffsetEst) < 0.1)
        self.assertTrue(np.abs(yOffsetTrue - yOffsetEst) < 0.1)
        self.assertTrue(np.abs(spotSepTrue - spotSepEst) < 0.1)


class TestOccastroSupportFunctions(unittest.TestCase):
    """Test support functions for occastro."""

    def setUp(self):
        localpath = os.path.dirname(os.path.abspath(__file__))
        self.fnTuning = os.path.join(
            localpath,
            'cgidata',
            'occastro_offset_and_separation_tuning_band1.yaml')
        fnSpots = os.path.join(localpath, 'testdata',
                               'occastro_nfov_4spots_R6.50_no_errors.fits')
        self.spotArray = fits.getdata(fnSpots)
        self.xOffsetGuess = -2.0
        self.yOffsetGuess = 1.9
        self.spotSepGuessPix = 16.1234
        self.optim_params = [self.xOffsetGuess,
                             self.yOffsetGuess,
                             self.spotSepGuessPix]

        tuning_dict = loadyaml(self.fnTuning)
        self.probeRotVecDeg = tuning_dict['probeRotVecDeg']
        self.roiRadiusPix = tuning_dict['roiRadiusPix']
        self.nSubpixels = tuning_dict['nSubpixels']
        self.fixed_params = [self.spotArray,
                             self.probeRotVecDeg,
                             self.roiRadiusPix,
                             self.nSubpixels,
                             ]

    def test_output_shape(self):
        """Check that the output has the right shape."""
    
        cost, roi_mask =  _roi_mask_for_spots(self.optim_params,
                                              self.fixed_params)
            
        self.assertTrue(np.all(roi_mask.shape == self.spotArray.shape))

    def test_cost_value(self):
        """Check that the computed cost is correct."""
    
        cost, roi_mask = _roi_mask_for_spots(self.optim_params,
                                             self.fixed_params)
        
        cost_expected = -np.sum(roi_mask * self.spotArray)

        self.assertTrue(isclose(cost, cost_expected, rel_tol=1e-8))
        
    def test_cost_value_passed_through(self):
        """Check that the nested computed cost is still correct."""
    
        cost0, roi_mask = _roi_mask_for_spots(self.optim_params,
                                              self.fixed_params)
    
        cost1 = _cost_func_spots(self.optim_params, self.fixed_params)

        self.assertTrue(isclose(cost0, cost1, rel_tol=1e-8))

    def test_roi_mask_for_spots_inputs(self):
        """
        Perform checks on bad inputs to _roi_mask_for_spots().
        
        Also counts for the thin wrapper function _cost_func_spots(), which
        takes the exact same inputs.
        """
        for optim_params_bad in not_an_iterable_of_length_3:
            with self.assertRaises(ValueError):
                _roi_mask_for_spots(optim_params_bad,
                                    self.fixed_params)

        for fixed_params_bad in not_an_iterable_of_length_4:
            with self.assertRaises(ValueError):
                _roi_mask_for_spots(self.optim_params,
                                    fixed_params_bad)

        for bad_val in not_a_real_scalar:
            opt_params_bad = [bad_val, self.yOffsetGuess, self.spotSepGuessPix]
            with self.assertRaises(TypeError):
                _roi_mask_for_spots(opt_params_bad,
                                    self.fixed_params)

        for bad_val in not_a_real_scalar:
            opt_params_bad = [self.xOffsetGuess, bad_val, self.spotSepGuessPix]
            with self.assertRaises(TypeError):
                _roi_mask_for_spots(opt_params_bad,
                                    self.fixed_params)

        for bad_val in not_a_real_positive_scalar:
            opt_params_bad = [self.xOffsetGuess, self.yOffsetGuess, bad_val]
            with self.assertRaises(TypeError):
                _roi_mask_for_spots(opt_params_bad,
                                    self.fixed_params)

        for bad_val in not_a_2d_array:
            fixed_params_bad = [bad_val,
                                self.probeRotVecDeg,
                                self.roiRadiusPix,
                                self.nSubpixels,
                                ]
            with self.assertRaises(TypeError):
                _roi_mask_for_spots(self.optim_params,
                                    fixed_params_bad)

        for bad_val in not_a_1d_array:
            fixed_params_bad = [self.spotArray,
                                bad_val,
                                self.roiRadiusPix,
                                self.nSubpixels,
                                ]
            with self.assertRaises(TypeError):
                _roi_mask_for_spots(self.optim_params,
                                    fixed_params_bad)

        for bad_val in not_a_real_positive_scalar:
            fixed_params_bad = [self.spotArray,
                                self.probeRotVecDeg,
                                bad_val,
                                self.nSubpixels,
                                ]
            with self.assertRaises(TypeError):
                _roi_mask_for_spots(self.optim_params,
                                    fixed_params_bad)

        for bad_val in not_a_positive_scalar_integer:
            fixed_params_bad = [self.spotArray,
                                self.probeRotVecDeg,
                                self.roiRadiusPix,
                                bad_val,
                                ]
            with self.assertRaises(TypeError):
                _roi_mask_for_spots(self.optim_params,
                                    fixed_params_bad)

if __name__ == '__main__':
    unittest.main()
