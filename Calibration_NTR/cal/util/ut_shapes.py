"""Test suite for shapes.py."""
import unittest
from math import isclose
import numpy as np

from . import shapes
from .insertinto import insertinto as inin


class TestShapesInputFailure(unittest.TestCase):
    """Test suite for valid function inputs."""

    def test_circle_inputs(self):
        """Test bad inputs of shapes.circle()."""
        # Nominal inputs
        nx = 51
        ny = 60
        roiRadiusPix = 5.0
        xShear = 10
        yShear = -8.2
        nSubpixels = 101

        # Check standard inputs do not raise anything first
        shapes.circle(nx, ny, roiRadiusPix, xShear, yShear)
        shapes.circle(nx, ny, roiRadiusPix, xShear, yShear,
                       nSubpixels=nSubpixels)

        with self.assertRaises(TypeError):
            shapes.circle(2.5, ny, roiRadiusPix, xShear, yShear,
                            nSubpixels=nSubpixels)
        with self.assertRaises(TypeError):
            shapes.circle(nx, -8, roiRadiusPix, xShear, yShear,
                            nSubpixels=nSubpixels)
        with self.assertRaises(TypeError):
            shapes.circle(nx, ny, -10, xShear, yShear,
                            nSubpixels=nSubpixels)
        with self.assertRaises(TypeError):
            shapes.circle(nx, ny, roiRadiusPix, 1j, yShear,
                            nSubpixels=nSubpixels)
        with self.assertRaises(TypeError):
            shapes.circle(nx, ny, roiRadiusPix, xShear, [8],
                            nSubpixels=nSubpixels)
        with self.assertRaises(TypeError):
            shapes.circle(nx, ny, roiRadiusPix, xShear, yShear,
                            nSubpixels=10.5)


class TestShapes(unittest.TestCase):
    """Unit tests."""

    def test_circle(self):
        """Test that circle makes the expected array."""
        # Nominal inputs
        nx = 51
        ny = 60
        roiRadiusPix = 5.0
        xShear = 10
        yShear = -8.2
        nSubpixels = 101

        # Check standard inputs do not raise anything first
        roiArray = shapes.circle(nx, ny, roiRadiusPix, xShear, yShear,
                             nSubpixels=nSubpixels)

        increaseFac = 100
        roiSum = increaseFac*np.sum(roiArray)
        expectedSum = increaseFac*np.pi*(roiRadiusPix*roiRadiusPix)
        self.assertTrue(isclose(roiSum, expectedSum, rel_tol=1e-3),
                               msg='Area of circle incorrect')

        xShear = -10
        yShear = 5
        roiArrayCentered = shapes.circle(nx, ny, roiRadiusPix, 0, 0)
        roiArrayShift = shapes.circle(nx, ny, roiRadiusPix, xShear, yShear)
        roiArrayRecentered = np.roll(roiArrayShift, (-yShear, -xShear),
                                     axis=(0, 1))
        diffSum = np.sum(np.abs(roiArrayCentered - roiArrayRecentered))
        self.assertTrue(isclose(diffSum, 0.),
                        msg='Shear incorrectly applied.')

        xShear = 2*nx
        yShear = -2*ny
        roiArray = shapes.circle(nx, ny, roiRadiusPix, xShear, yShear)
        self.assertTrue(isclose(np.sum(roiArray), 0.),
                        msg='ROI should be outside array.')
        pass


class TestEllipse(unittest.TestCase):
    """Unit tests for shapes.ellipse()."""

    def setUp(self):
        self.nx = 51
        self.ny = 60
        self.rx = 5.0
        self.ry = 6.5
        self.rot = 0.0
        self.xOffset = 10
        self.yOffset = -8.2
        self.nSubpixels = 101

    def test_input_failures(self):
        with self.assertRaises(TypeError):
            shapes.circle(2.5, self.ny, self.rx, self.ry, self.rot,
                          self.xOffset, self.yOffset,
                          nSubpixels=self.nSubpixels)
        with self.assertRaises(TypeError):
            shapes.circle(self.nx, 2.5, self.rx, self.ry, self.rot,
                          self.xOffset, self.yOffset,
                          nSubpixels=self.nSubpixels)
        with self.assertRaises(TypeError):
            shapes.circle(self.nx, self.ny, -1, self.ry, self.rot,
                          self.xOffset, self.yOffset,
                          nSubpixels=self.nSubpixels)
        with self.assertRaises(TypeError):
            shapes.circle(self.nx, self.ny, self.rx, 1j, self.rot,
                          self.xOffset, self.yOffset,
                          nSubpixels=self.nSubpixels)
        with self.assertRaises(TypeError):
            shapes.circle(self.nx, self.ny, self.rx, self.ry, [1, ],
                          self.xOffset, self.yOffset,
                          nSubpixels=self.nSubpixels)
        with self.assertRaises(TypeError):
            shapes.circle(self.nx, self.ny, self.rx, self.ry, self.rot,
                          1j, self.yOffset, nSubpixels=self.nSubpixels)
        with self.assertRaises(TypeError):
            shapes.circle(self.nx, self.ny, self.rx, self.ry, self.rot,
                          self.xOffset, [2, ], nSubpixels=self.nSubpixels)
        with self.assertRaises(TypeError):
            shapes.circle(self.nx, self.ny, self.rx, self.ry, self.rot,
                          self.xOffset, self.yOffset, nSubpixels=10.2)

    def test_area(self):
        """Test that ellipse area matches analytical value."""
        out = shapes.ellipse(self.nx, self.ny, self.rx, self.ry, self.rot,
                             self.xOffset, self.yOffset,
                             nSubpixels=self.nSubpixels)

        shapeSum = np.sum(out)
        expectedSum = np.pi*self.rx*self.ry
        self.assertTrue(isclose(shapeSum, expectedSum, rel_tol=1e-3),
                        msg='Area of ellipse incorrect')

    def test_shift(self):
        """Test that ellipse shifts as expected."""
        xOffset = -10
        yOffset = 5
        maskCentered = shapes.ellipse(self.nx, self.ny, self.rx, self.ry,
                                      self.rot, 0, 0)
        maskShifted = shapes.ellipse(self.nx, self.ny, self.rx, self.ry,
                                     self.rot, xOffset, yOffset)
        maskRecentered = np.roll(maskShifted, (-yOffset, -xOffset),
                                 axis=(0, 1))
        diffSum = np.sum(np.abs(maskCentered - maskRecentered))
        self.assertTrue(isclose(diffSum, 0.),
                        msg='Shear incorrectly applied.')

    def test_rotation(self):
        """Test that ellipse rotates as expected."""
        mask0 = shapes.ellipse(51, 51, self.rx, self.ry, 0, 0, 0)
        maskRot = shapes.ellipse(51, 51, self.rx, self.ry, 90, 0, 0)
        maskDerot = np.rot90(maskRot, 1)
        diffSum = np.sum(np.abs(mask0 - maskDerot))

        self.assertTrue(isclose(diffSum, 0.),
                        msg='Rotation incorrect.')

    def test_outside(self):
        """Test that the ellipse isn't in the array."""
        xOffset = 2*self.nx
        yOffset = -2*self.ny
        out = shapes.ellipse(self.nx, self.ny, self.rx, self.ry, self.rot,
                             xOffset, yOffset)
        self.assertTrue(isclose(np.sum(out), 0.),
                        msg='Ellipse should be outside array.')

    def test_centering(self):
        """Test that center doesn't shift for different array sizes."""
        out0 = shapes.ellipse(self.nx, self.ny, self.rx, self.ry, self.rot,
                              self.xOffset, self.yOffset,
                              nSubpixels=self.nSubpixels)
        out1 = shapes.ellipse(self.nx+1, self.ny+1, self.rx, self.ry, self.rot,
                              self.xOffset, self.yOffset,
                              nSubpixels=self.nSubpixels)
        out1 = inin(out1, out0.shape)

        self.assertTrue(np.allclose(out0, out1, rtol=1e-3),
                        msg='Centering changed with array size.')


if __name__ == '__main__':
    unittest.main()
