"""
Test class for satspots.calc_sat_spots
"""

import unittest
import os
import numpy as np
from astropy.io import fits

from cal.occastro.satspots import calc_sat_spots
from cal.util import loadyaml

# utility for comparing angles near +/- pi
def mod2pi(a):
    return np.mod(a + np.pi, 2*np.pi) - np.pi

# utility for shifting images to test star offset
def roll_img(img, yroll, xroll):
    return np.roll(np.roll(img, yroll, axis=0), xroll, axis=1)


class Test_cal_sat_spots(unittest.TestCase):
    """
    Unit test for function to estimate radius and offset (star location)
    of images generated by applying sat spots ddmv to the DM
    """

    def setUp(self):
        """
        Predefine OK parameters
        """

        localpath = os.path.dirname(os.path.abspath(__file__))

        # nfov images:
        fnRef = os.path.join(localpath, 'testdata', 'Im_satspots_nfov_ref.fits')
        self.img_nfov_ref = fits.getdata(fnRef)

        fnPlus = os.path.join(localpath, 'testdata', 'Im_satspots_nfov_plus.fits')
        self.img_nfov_plus = fits.getdata(fnPlus)

        fnMinus = os.path.join(localpath, 'testdata', 'Im_satspots_nfov_minus.fits')
        self.img_nfov_minus = fits.getdata(fnMinus)

        self.fn_nfov_offset_YAML = os.path.join(
            localpath, 'testdata', 'occastro_offset_tuning_nfov.yaml')
        self.fn_nfov_separation_YAML = os.path.join(
            localpath, 'testdata', 'occastro_separation_tuning_nfov.yaml')

        # spec images:
        fnRef = os.path.join(localpath, 'testdata', 'Im_satspots_spec_ref.fits')
        self.img_spec_ref = fits.getdata(fnRef)

        fnPlus = os.path.join(localpath, 'testdata', 'Im_satspots_spec_plus.fits')
        self.img_spec_plus = fits.getdata(fnPlus)

        fnMinus = os.path.join(localpath, 'testdata', 'Im_satspots_spec_minus.fits')
        self.img_spec_minus = fits.getdata(fnMinus)

        self.fn_spec_offset_YAML = os.path.join(
            localpath, 'testdata', 'occastro_offset_tuning_spec.yaml')
        self.fn_spec_separation_YAML = os.path.join(
            localpath, 'testdata', 'occastro_separation_tuning_spec.yaml')


        # wfov images:
        fnRef = os.path.join(localpath, 'testdata', 'Im_satspots_wfov_ref.fits')
        self.img_wfov_ref = fits.getdata(fnRef)

        fnPlus = os.path.join(localpath, 'testdata', 'Im_satspots_wfov_plus.fits')
        self.img_wfov_plus = fits.getdata(fnPlus)

        fnMinus = os.path.join(localpath, 'testdata', 'Im_satspots_wfov_minus.fits')
        self.img_wfov_minus = fits.getdata(fnMinus)

        self.fn_wfov_offset_YAML = os.path.join(
            localpath, 'testdata', 'occastro_offset_tuning_wfov.yaml')
        self.fn_wfov_separation_YAML = os.path.join(
            localpath, 'testdata', 'occastro_separation_tuning_wfov.yaml')

        # all cases start with:
        self.xOffsetGuess = 0.0
        self.yOffsetGuess = 0.0
        self.thetaOffsetGuess = 0.0

    # test inputs
    def test_success(self):
        """
        Run with valid inputs should not throw anything
        """
        calc_sat_spots(self.img_nfov_ref, self.img_nfov_plus, self.img_nfov_minus,
                       self.xOffsetGuess, self.yOffsetGuess, self.thetaOffsetGuess,
                       self.fn_nfov_offset_YAML, self.fn_nfov_separation_YAML)

    def test_img_ref_invalid(self):
        """
        Test bad input image ref
        """
        img_shp = self.img_nfov_ref.shape

        for perr in [None, 'txt', 1.5, [1.0, 2.0], np.ones((img_shp[0], img_shp[1]+1))]:
            with self.assertRaises(TypeError):
                calc_sat_spots(perr, self.img_nfov_plus, self.img_nfov_minus,
                               self.xOffsetGuess, self.yOffsetGuess, self.thetaOffsetGuess,
                               self.fn_nfov_offset_YAML, self.fn_nfov_separation_YAML)

    def test_img_plus_invalid(self):
        """
        Test bad input image plus
        """
        img_shp = self.img_nfov_ref.shape

        for perr in [None, 'txt', 1.5, [1.0, 2.0], np.ones((img_shp[0], img_shp[1]+1))]:
            with self.assertRaises(TypeError):
                calc_sat_spots(self.img_nfov_ref, perr, self.img_nfov_minus,
                               self.xOffsetGuess, self.yOffsetGuess, self.thetaOffsetGuess,
                               self.fn_nfov_offset_YAML, self.fn_nfov_separation_YAML)

    def test_img_minus_invalid(self):
        """
        Test bad input image minus
        """
        img_shp = self.img_nfov_ref.shape

        for perr in [None, 'txt', 1.5, [1.0, 2.0], np.ones((img_shp[0], img_shp[1]+1))]:
            with self.assertRaises(TypeError):
                calc_sat_spots(self.img_nfov_ref, self.img_nfov_plus, perr,
                               self.xOffsetGuess, self.yOffsetGuess, self.thetaOffsetGuess,
                               self.fn_nfov_offset_YAML, self.fn_nfov_separation_YAML)

    def test_xOffsetGuess_invalid(self):
        """
        Test bad input xOffsetGuess (real scalar)
        """
        for perr in [None, 'txt', 1j, [1.0, 2.0], np.ones((3, 3))]:
            with self.assertRaises(TypeError):
                calc_sat_spots(self.img_nfov_ref, self.img_nfov_plus, self.img_nfov_minus,
                               perr, self.yOffsetGuess, self.thetaOffsetGuess,
                               self.fn_nfov_offset_YAML, self.fn_nfov_separation_YAML)

    def test_yOffsetGuess_invalid(self):
        """
        Test bad input yOffsetGuess (real scalar)
        """
        for perr in [None, 'txt', 1j, [1.0, 2.0], np.ones((3, 3))]:
            with self.assertRaises(TypeError):
                calc_sat_spots(self.img_nfov_ref, self.img_nfov_plus, self.img_nfov_minus,
                               self.xOffsetGuess, perr, self.thetaOffsetGuess,
                               self.fn_nfov_offset_YAML, self.fn_nfov_separation_YAML)

    # for fn_offset_YAML and fn_separation_YAML rely on file checks in
    # util.loadyaml(), which is used by the  occastro.py routines

    # test operation
    # Requirement:
    # Given 3 images (ref, ref + sat spots, ref - sat spots), compute the location
    # of the central star and of the spots

    # test baseline cases.
    # Is there a tolerance requirement for star and spot estimatition?

    def test_baseline_nfov(self):
        """
        Test estimation error for baseline nfov case
        The input images were created with these parameters:
        radius = 6.5 lam/D = 6.5*(51.46*0.575/13) = 14.79 pixels
        theta = [0, pi/2]
        star center = [0, 0]
        """
        star_xy, list_spots_xy = calc_sat_spots(
            self.img_nfov_ref, self.img_nfov_plus, self.img_nfov_minus,
            self.xOffsetGuess, self.yOffsetGuess, self.thetaOffsetGuess,
            self.fn_nfov_offset_YAML, self.fn_nfov_separation_YAML
        )

        # check star has [x, y]
        self.assertTrue(len(star_xy) == 2)
        # check star is close to 0, 0
        self.assertTrue(np.max(np.abs(star_xy[0] - 0.0)) < 0.5)
        self.assertTrue(np.max(np.abs(star_xy[1] - 0.0)) < 0.5)

        # expected locations of sat spots:
        radius_pix = 6.5*(51.46*0.575/13) # = 14.79 pixels
        theta = (np.array([0, 0.5, 1.0, 1.5])*np.pi).tolist()
        for xy_test in list_spots_xy:
            radius_test = np.hypot(xy_test[0]-star_xy[0], xy_test[1]-star_xy[1])
            theta_test = np.arctan2(xy_test[1]-star_xy[1], xy_test[0]-star_xy[0])
            self.assertTrue(np.max(np.abs(radius_test - radius_pix)) < 0.5)
            ii = np.argmin(np.abs(mod2pi(np.array(theta) - theta_test)))
            self.assertTrue(np.abs(mod2pi(theta[ii] - theta_test)) < 0.1)
            theta.pop(ii)

    def test_baseline_spec(self):
        """
        Test estimation error for baseline spec case
        The input images were created with these parameters:
        radius = 6.0 lam/D = 6.0*(51.46*0.730/13) = 17.34 pixels
        theta = [0,]
        star center = [0, 0]
        """
        star_xy, list_spots_xy = calc_sat_spots(
            self.img_spec_ref, self.img_spec_plus, self.img_spec_minus,
            self.xOffsetGuess, self.yOffsetGuess, self.thetaOffsetGuess,
            self.fn_spec_offset_YAML, self.fn_spec_separation_YAML
        )

        # check star has [x, y]
        self.assertTrue(len(star_xy) == 2)
        # check star is close to 0, 0
        self.assertTrue(np.max(np.abs(star_xy[0] - 0.0)) < 0.5)
        self.assertTrue(np.max(np.abs(star_xy[1] - 0.0)) < 0.5)

        # expected locations of sat spots:
        radius_pix = 6.0*(51.46*0.730/13) # = 17.34 pixels
        theta = (np.array([0, 1.0])*np.pi).tolist()
        for xy_test in list_spots_xy:
            radius_test = np.hypot(xy_test[0]-star_xy[0], xy_test[1]-star_xy[1])
            theta_test = np.arctan2(xy_test[1]-star_xy[1], xy_test[0]-star_xy[0])
            self.assertTrue(np.max(np.abs(radius_test - radius_pix)) < 1.0)
            ii = np.argmin(np.abs(mod2pi(np.array(theta) - theta_test)))
            self.assertTrue(np.abs(mod2pi(theta[ii] - theta_test)) < 0.1)
            theta.pop(ii)

    def test_baseline_wfov(self):
        """
        Test estimation error for baseline wfov case
        The input images were created with these parameters:
        radius = 13.0 lam/D = 13.0*(51.46*0.825/13) = 42.45
        theta = [0, pi/2]
        star center = [0, 0]
        """
        star_xy, list_spots_xy = calc_sat_spots(
            self.img_wfov_ref, self.img_wfov_plus, self.img_wfov_minus,
            self.xOffsetGuess, self.yOffsetGuess, self.thetaOffsetGuess,
            self.fn_wfov_offset_YAML, self.fn_wfov_separation_YAML
        )

        # check star has [x, y]
        self.assertTrue(len(star_xy) == 2)
        # check star is close to 0, 0
        self.assertTrue(np.max(np.abs(star_xy[0] - 0.0)) < 0.5)
        self.assertTrue(np.max(np.abs(star_xy[1] - 0.0)) < 0.5)

        # expected locations of sat spots:
        radius_pix = 13.0*(51.46*0.825/13) # = 42.45 pixels
        theta = (np.array([0, 0.5, 1.0, 1.5])*np.pi).tolist()

        for xy_test in list_spots_xy:
            radius_test = np.hypot(xy_test[0]-star_xy[0], xy_test[1]-star_xy[1])
            theta_test = np.arctan2(xy_test[1]-star_xy[1], xy_test[0]-star_xy[0])
            self.assertTrue(np.max(np.abs(radius_test - radius_pix)) < 1.0)
            ii = np.argmin(np.abs(mod2pi(np.array(theta) - theta_test)))
            self.assertTrue(np.abs(mod2pi(theta[ii] - theta_test)) < 0.1)
            theta.pop(ii)

    # test move the star away from 0,0
    # Is there a tolerance requirement for star and spot estimatition?

    def test_offset_y_nfov(self):
        """
        Test estimation error for baseline nfov case
        The input images were created with these parameters:
        radius = 6.5 lam/D = 6.5*(51.46*0.575/13) = 14.79 pixels
        theta = [0, pi/2]
        star center = [0, 0]
        """

        xroll = 0
        yroll = 2

        img_nfov_ref = roll_img(self.img_nfov_ref, yroll, xroll)
        img_nfov_plus = roll_img(self.img_nfov_plus, yroll, xroll)
        img_nfov_minus = roll_img(self.img_nfov_minus, yroll, xroll)

        star_xy, list_spots_xy = calc_sat_spots(
            img_nfov_ref, img_nfov_plus, img_nfov_minus,
            self.xOffsetGuess, self.yOffsetGuess, self.thetaOffsetGuess,
            self.fn_nfov_offset_YAML, self.fn_nfov_separation_YAML
        )

        # check star has [x, y]
        self.assertTrue(len(star_xy) == 2)
        # check star is close to 0, 2
        self.assertTrue(np.abs(star_xy[0] - xroll) < 0.5)
        self.assertTrue(np.abs(star_xy[1] - yroll) < 0.5)

        # expected locations of sat spots:
        radius_pix = 6.5*(51.46*0.575/13) # = 14.79 pixels
        theta = (np.array([0, 0.5, 1.0, 1.5])*np.pi).tolist()
        for xy_test in list_spots_xy:
            radius_test = np.hypot(xy_test[0]-star_xy[0], xy_test[1]-star_xy[1])
            theta_test = np.arctan2(xy_test[1]-star_xy[1], xy_test[0]-star_xy[0])
            self.assertTrue(np.abs(radius_test - radius_pix) < 1.0)
            ii = np.argmin(np.abs(mod2pi(np.array(theta) - theta_test)))
            self.assertTrue(np.abs(mod2pi(theta[ii] - theta_test)) < 0.1)
            theta.pop(ii)

    def test_offset_y_spec(self):
        """
        Test estimation error for spec with offset y case
        The input images were created with these parameters:
        radius = 6.0 lam/D = 6.0*(51.46*0.730/13) = 17.34 pixels
        theta = [0,]
        star center = [0, 2]
        """

        xroll = 0
        yroll = 2

        img_spec_ref = roll_img(self.img_spec_ref, yroll, xroll)
        img_spec_plus = roll_img(self.img_spec_plus, yroll, xroll)
        img_spec_minus = roll_img(self.img_spec_minus, yroll, xroll)

        star_xy, list_spots_xy = calc_sat_spots(
            img_spec_ref, img_spec_plus, img_spec_minus,
            self.xOffsetGuess, self.yOffsetGuess, self.thetaOffsetGuess,
            self.fn_spec_offset_YAML, self.fn_spec_separation_YAML
        )

        # check star has [x, y]
        self.assertTrue(len(star_xy) == 2)
        # check star is close to 0, 2
        self.assertTrue(np.abs(star_xy[0] - xroll) < 0.5)
        self.assertTrue(np.abs(star_xy[1] - yroll) < 0.5)

        # expected locations of sat spots:
        radius_pix = 6.0*(51.46*0.730/13) # = 17.34 pixels
        theta = (np.array([0, 1.0])*np.pi).tolist()
        for xy_test in list_spots_xy:
            radius_test = np.hypot(xy_test[0]-star_xy[0], xy_test[1]-star_xy[1])
            theta_test = np.arctan2(xy_test[1]-star_xy[1], xy_test[0]-star_xy[0])
            self.assertTrue(np.abs(radius_test - radius_pix) < 1.0)
            ii = np.argmin(np.abs(mod2pi(np.array(theta) - theta_test)))
            self.assertTrue(np.abs(mod2pi(theta[ii] - theta_test)) < 0.1)
            theta.pop(ii)

    def test_offset_y_wfov(self):
        """
        Test estimation error for wfov with y offset case
        The input images were created with these parameters:
        radius = 13.0 lam/D = 13.0*(51.46*0.825/13) = 42.45
        theta = [0, pi/2]
        star center = [0, 0]
        """

        xroll = 0
        yroll = 2

        img_wfov_ref = roll_img(self.img_wfov_ref, yroll, xroll)
        img_wfov_plus = roll_img(self.img_wfov_plus, yroll, xroll)
        img_wfov_minus = roll_img(self.img_wfov_minus, yroll, xroll)

        star_xy, list_spots_xy = calc_sat_spots(
            img_wfov_ref, img_wfov_plus, img_wfov_minus,
            self.xOffsetGuess, self.yOffsetGuess, self.thetaOffsetGuess,
            self.fn_wfov_offset_YAML, self.fn_wfov_separation_YAML
        )

        # check star has [x, y]
        self.assertTrue(len(star_xy) == 2)
        # check star is close to 0, 2
        self.assertTrue(np.abs(star_xy[0] - xroll) < 0.5)
        self.assertTrue(np.abs(star_xy[1] - yroll) < 0.5)

        # expected locations of sat spots:
        radius_pix = 13.0*(51.46*0.825/13) # = 42.45 pixels
        theta = (np.array([0, 0.5, 1.0, 1.5])*np.pi).tolist()
        for xy_test in list_spots_xy:
            radius_test = np.hypot(xy_test[0]-star_xy[0], xy_test[1]-star_xy[1])
            theta_test = np.arctan2(xy_test[1]-star_xy[1], xy_test[0]-star_xy[0])
            self.assertTrue(np.abs(radius_test - radius_pix) < 1.0)
            ii = np.argmin(np.abs(mod2pi(np.array(theta) - theta_test)))
            self.assertTrue(np.abs(mod2pi(theta[ii] - theta_test)) < 0.1)
            theta.pop(ii)

    def test_offset_x_nfov(self):
        """
        Test estimation error for baseline nfov case
        The input images were created with these parameters:
        radius = 6.5 lam/D = 6.5*(51.46*0.575/13) = 14.79 pixels
        theta = [0, pi/2]
        star center = [0, 0]
        """

        xroll = 2
        yroll = 0

        img_nfov_ref = roll_img(self.img_nfov_ref, yroll, xroll)
        img_nfov_plus = roll_img(self.img_nfov_plus, yroll, xroll)
        img_nfov_minus = roll_img(self.img_nfov_minus, yroll, xroll)

        star_xy, list_spots_xy = calc_sat_spots(
            img_nfov_ref, img_nfov_plus, img_nfov_minus,
            self.xOffsetGuess, self.yOffsetGuess, self.thetaOffsetGuess,
            self.fn_nfov_offset_YAML, self.fn_nfov_separation_YAML
        )

        # check star has [x, y]
        self.assertTrue(len(star_xy) == 2)
        # check star is close to 0, 2
        self.assertTrue(np.abs(star_xy[0] - xroll) < 0.5)
        self.assertTrue(np.abs(star_xy[1] - yroll) < 0.5)

        # expected locations of sat spots:
        radius_pix = 6.5*(51.46*0.575/13) # = 14.79 pixels
        theta = (np.array([0, 0.5, 1.0, 1.5])*np.pi).tolist()
        for xy_test in list_spots_xy:
            radius_test = np.hypot(xy_test[0]-star_xy[0], xy_test[1]-star_xy[1])
            theta_test = np.arctan2(xy_test[1]-star_xy[1], xy_test[0]-star_xy[0])
            self.assertTrue(np.abs(radius_test - radius_pix) < 1.0)
            ii = np.argmin(np.abs(mod2pi(np.array(theta) - theta_test)))
            self.assertTrue(np.abs(mod2pi(theta[ii] - theta_test)) < 0.1)
            theta.pop(ii)

    def test_offset_x_spec(self):
        """
        Test estimation error for spec with offset y case
        The input images were created with these parameters:
        radius = 6.0 lam/D = 6.0*(51.46*0.730/13) = 17.34 pixels
        theta = [0,]
        star center = [0, 2]
        """

        xroll = 2
        yroll = 0

        img_spec_ref = roll_img(self.img_spec_ref, yroll, xroll)
        img_spec_plus = roll_img(self.img_spec_plus, yroll, xroll)
        img_spec_minus = roll_img(self.img_spec_minus, yroll, xroll)

        star_xy, list_spots_xy = calc_sat_spots(
            img_spec_ref, img_spec_plus, img_spec_minus,
            self.xOffsetGuess, self.yOffsetGuess, self.thetaOffsetGuess,
            self.fn_spec_offset_YAML, self.fn_spec_separation_YAML
        )

        # check star has [x, y]
        self.assertTrue(len(star_xy) == 2)
        # check star is close to 0, 2
        self.assertTrue(np.abs(star_xy[0] - xroll) < 0.5)
        self.assertTrue(np.abs(star_xy[1] - yroll) < 0.5)

        # expected locations of sat spots:
        radius_pix = 6.0*(51.46*0.730/13) # = 17.34 pixels
        theta = (np.array([0, 1.0])*np.pi).tolist()
        for xy_test in list_spots_xy:
            radius_test = np.hypot(xy_test[0]-star_xy[0], xy_test[1]-star_xy[1])
            theta_test = np.arctan2(xy_test[1]-star_xy[1], xy_test[0]-star_xy[0])
            self.assertTrue(np.abs(radius_test - radius_pix) < 1.0)
            ii = np.argmin(np.abs(mod2pi(np.array(theta) - theta_test)))
            self.assertTrue(np.abs(mod2pi(theta[ii] - theta_test)) < 0.1)
            theta.pop(ii)

    def test_offset_x_wfov(self):
        """
        Test estimation error for wfov with y offset case
        The input images were created with these parameters:
        radius = 13.0 lam/D = 13.0*(51.46*0.825/13) = 42.45
        theta = [0, pi/2]
        star center = [0, 0]
        """

        xroll = 2
        yroll = 0

        img_wfov_ref = roll_img(self.img_wfov_ref, yroll, xroll)
        img_wfov_plus = roll_img(self.img_wfov_plus, yroll, xroll)
        img_wfov_minus = roll_img(self.img_wfov_minus, yroll, xroll)

        star_xy, list_spots_xy = calc_sat_spots(
            img_wfov_ref, img_wfov_plus, img_wfov_minus,
            self.xOffsetGuess, self.yOffsetGuess, self.thetaOffsetGuess,
            self.fn_wfov_offset_YAML, self.fn_wfov_separation_YAML
        )

        # check star has [x, y]
        self.assertTrue(len(star_xy) == 2)
        # check star is close to 0, 2
        self.assertTrue(np.abs(star_xy[0] - xroll) < 0.5)
        self.assertTrue(np.abs(star_xy[1] - yroll) < 0.5)

        # expected locations of sat spots:
        radius_pix = 13.0*(51.46*0.825/13) # = 42.45 pixels
        theta = (np.array([0, 0.5, 1.0, 1.5])*np.pi).tolist()
        for xy_test in list_spots_xy:
            radius_test = np.hypot(xy_test[0]-star_xy[0], xy_test[1]-star_xy[1])
            theta_test = np.arctan2(xy_test[1]-star_xy[1], xy_test[0]-star_xy[0])
            self.assertTrue(np.abs(radius_test - radius_pix) < 1.0)
            ii = np.argmin(np.abs(mod2pi(np.array(theta) - theta_test)))
            self.assertTrue(np.abs(mod2pi(theta[ii] - theta_test)) < 0.1)
            theta.pop(ii)

if __name__ == '__main__':
    unittest.main()