"""Test suite for fit_shapes.py."""
import unittest
from math import isclose
import numpy as np

from . import fit_shapes, shapes

list_not_pos_scalar_int = [-1, 0, 1.5, 1j, 'string',  np.ones((5,))]


class TestFitCircle(unittest.TestCase):
    """Unit tests for fit_circle."""

    def setUp(self):
        """Define re-used variables."""
        # Make the image containing a circle
        nx = 51
        ny = 60
        self.radiusTrue = 10.3
        self.xOffsetTrue = 10
        self.yOffsetTrue = -8.4
        nSubpixels = 101
        self.image = shapes.circle(nx, ny, self.radiusTrue, self.xOffsetTrue,
                                   self.yOffsetTrue, nSubpixels=nSubpixels)

        # Define tuning parameters
        self.min_radius = 8
        self.max_radius = 12
        self.sigma = 1.0

    def test_fit_bright_circle(self):
        """Test that a bright circle is correctly fitted."""
        # Perform the fit
        xOffsetEst, yOffsetEst, radiusEst = fit_shapes.fit_circle(
            self.image, self.min_radius, self.max_radius,
            edge_sigma=self.sigma)

        abs_tol = 1.0  # pixels
        self.assertTrue(isclose(self.xOffsetTrue, xOffsetEst, abs_tol=abs_tol))
        self.assertTrue(isclose(self.yOffsetTrue, yOffsetEst, abs_tol=abs_tol))
        self.assertTrue(isclose(self.radiusTrue, radiusEst, abs_tol=abs_tol))

    def test_fit_dark_circle(self):
        """Test that a dark circle is correctly fitted."""
        # Perform the fit
        xOffsetEst, yOffsetEst, radiusEst = fit_shapes.fit_circle(
            1.0-self.image, self.min_radius, self.max_radius,
            edge_sigma=self.sigma)

        abs_tol = 1.0  # pixels
        self.assertTrue(isclose(self.xOffsetTrue, xOffsetEst, abs_tol=abs_tol))
        self.assertTrue(isclose(self.yOffsetTrue, yOffsetEst, abs_tol=abs_tol))
        self.assertTrue(isclose(self.radiusTrue, radiusEst, abs_tol=abs_tol))

    def test_fit_circle_input_0(self):
        """Test bad inputs to fit_circle()."""
        for badVal in (-1, 0, 1, 1.5, (1, 2), np.ones((3, 3, 3)), 1j*np.eye(3),
                       'asdf'):
            with self.assertRaises(TypeError):
                _, _, _ = fit_shapes.fit_circle(
                    badVal, self.min_radius, self.max_radius,
                    edge_sigma=self.sigma)

    def test_fit_circle_input_1(self):
        """Test bad inputs to fit_circle()."""
        for badVal in (-1, 0, 1.5, (1, 2), np.ones((3, 3, 3)), 'asdf'):
            with self.assertRaises(TypeError):
                _, _, _ = fit_shapes.fit_circle(
                    self.image, badVal, self.max_radius,
                    edge_sigma=self.sigma)

    def test_fit_circle_input_2(self):
        """Test bad inputs to fit_circle()."""
        for badVal in (-1, 0, 1.5, (1, 2), np.ones((3, 3, 3)), 'asdf'):
            with self.assertRaises(TypeError):
                _, _, _ = fit_shapes.fit_circle(
                    self.image, self.min_radius, badVal,
                    edge_sigma=self.sigma)

    def test_fit_circle_input_2_b(self):
        """Test bad inputs to fit_circle()."""
        for badVal in (self.min_radius-1, self.min_radius):
            with self.assertRaises(ValueError):
                _, _, _ = fit_shapes.fit_circle(
                    self.image, self.min_radius, badVal,
                    edge_sigma=self.sigma)

    def test_fit_circle_input_3(self):
        """Test bad inputs to fit_circle()."""
        for badVal in (-1, 0, (1, 2), np.ones((3, 3, 3)), 'asdf'):
            with self.assertRaises(TypeError):
                _, _, _ = fit_shapes.fit_circle(
                    self.image, self.min_radius, self.max_radius,
                    edge_sigma=badVal)

    def test_fit_circle_input_4(self):
        """Test bad inputs to fit_circle()."""
        for badVal in list_not_pos_scalar_int:
            with self.assertRaises(TypeError):
                _, _, _ = fit_shapes.fit_circle(
                    self.image, self.min_radius, self.max_radius, nBin=badVal)


class TestFitEllipse(unittest.TestCase):
    """Test suite for fit_ellipse()."""

    def setUp(self):
        """Define re-used variables."""
        self.diamTrue = 41.5  # pixels
        self.xOffsetTrue = 5.672  # pixels
        self.yOffsetTrue = -12.250  # pixels
        self.clockDegTrue = 10.0  # degrees
        nArray = 100

        self.pupilMeas = shapes.ellipse(
            nArray, nArray, 0.45*self.diamTrue, 0.5*self.diamTrue,
            self.clockDegTrue, self.xOffsetTrue, self.yOffsetTrue)

    def test_fit_ellipse(self):
        """
        Test that the unmasked pupil parameters are fitted correctly.

        The hough_ellipse() function used within fit_ellipse() seems to be
        accurate only to within about roughly a quarter pixel for the
        lateral alignment.
        """
        diamEst, xOffsetEst, yOffsetEst = \
            fit_shapes.fit_ellipse(self.pupilMeas)

        self.assertTrue(np.abs(xOffsetEst-self.xOffsetTrue) < 0.5)  # pixels
        self.assertTrue(np.abs(yOffsetEst-self.yOffsetTrue) < 0.5)  # pixels
        self.assertTrue(np.abs(diamEst-self.diamTrue) < 0.5)  # pixels

    def test_bad_inputs_0(self):
        """Test incorrect inputs."""
        for badPupil in (-1, 0, 1, 1.5, 1j, 1j*np.eye(6), np.ones((3, 3, 3))):
            with self.assertRaises(TypeError):
                fit_shapes.fit_ellipse(badPupil)

    def test_bad_inputs_1(self):
        """Test incorrect inputs."""
        for bad_n in (-1, 1j, 1j*np.eye(6), np.ones((3, 3, 3))):
            with self.assertRaises(TypeError):
                fit_shapes.fit_ellipse(self.pupilMeas,
                                       n_iter_dilate_erode=bad_n)

    def test_bad_inputs_2(self):
        """Test incorrect inputs."""
        for bad_pad_factor in (-1, 0, 1j, 1j*np.eye(6), np.ones((3, 3, 3))):
            with self.assertRaises(TypeError):
                fit_shapes.fit_ellipse(self.pupilMeas,
                                       pad_factor=bad_pad_factor)

    def test_bad_inputs_3(self):
        """Test incorrect inputs."""
        for badVal in list_not_pos_scalar_int:
            with self.assertRaises(TypeError):
                fit_shapes.fit_ellipse(self.pupilMeas, nBin=badVal)

if __name__ == '__main__':
    unittest.main()
