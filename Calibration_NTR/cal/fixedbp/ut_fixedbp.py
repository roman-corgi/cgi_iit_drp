"""Unit tests for fixed bad pixel map."""
import unittest
import numpy as np

from cal.fixedbp.fixedbp import (
    compute_fixedbp_excam,
    _compute_fixedbp_from_dark, _compute_fixedbp_from_flat
)


class TestBPMapForEXCAM(unittest.TestCase):
    """Tests for building the fixed bad-pixel map for EXCAM."""

    def setUp(self):
        """Set up."""
        self.dark = np.zeros((10, 10))
        self.flat = np.ones((10, 10))
        self.dthresh = 5
        self.ffrac = 0.8
        self.fwidth = 4

    def test_success(self):
        """Valid inputs complete successfully."""
        compute_fixedbp_excam(self.dark, self.flat, self.dthresh, self.ffrac,
                    self.fwidth)

    def test_success_clean(self):
        """Valid inputs complete successfully for clean-sized frame."""
        dark = np.zeros((1024, 1024))
        flat = np.ones((1024, 1024))
        compute_fixedbp_excam(dark, flat, self.dthresh, self.ffrac,
                          self.fwidth)

    def test_success_nonsquare(self):
        """Valid inputs complete successfully for nonsquare frame."""
        dark = np.zeros((10, 11))
        flat = np.ones((10, 11))
        compute_fixedbp_excam(dark, flat, self.dthresh, self.ffrac,
                          self.fwidth)

    def test_exact_uniform(self):
        """Uniform dark + flat gives no bad pixels."""
        # Use tight constraints, shouldn't matter for uniform
        dark = np.zeros((10, 10))
        flat = np.ones((10, 10))
        dthresh = 0
        ffrac = 1.0
        fwidth = 5

        target = np.zeros_like(dark).astype('bool')
        out = compute_fixedbp_excam(dark, flat, dthresh, ffrac, fwidth)

        self.assertTrue((out == target).all())

    def test_exact_nonuniform_darkonly(self):
        """Bad pixel in dark caught."""
        dark = np.array([[10, 10, 10],
                         [10, 10, 100],
                         [10, 10, 10]])
        flat = np.ones((3, 3))
        dthresh = 1.0
        ffrac = 1.0
        fwidth = 5

        target = np.array([[0, 0, 0],
                           [0, 0, 1],
                           [0, 0, 0]]).astype('bool')
        out = compute_fixedbp_excam(dark, flat, dthresh, ffrac, fwidth)

        self.assertTrue((out == target).all())

    def test_exact_nonuniform_flatonly(self):
        """Bad pixel in flat caught."""
        dark = np.zeros((3, 3))
        flat = np.array([[1, 1, 1],
                         [0.2, 1, 1],
                         [1, 1, 1]])
        dthresh = 1.0
        ffrac = 0.8
        fwidth = 5

        target = np.array([[0, 0, 0],
                           [1, 0, 0],
                           [0, 0, 0]]).astype('bool')
        out = compute_fixedbp_excam(dark, flat, dthresh, ffrac, fwidth)

        self.assertTrue((out == target).all())

    def test_exact_nonuniform_darkflat(self):
        """Bad pixels in dark and flat caught."""
        dark = np.array([[10, 10, 10],
                         [10, 10, 100],
                         [10, 10, 10]])
        flat = np.array([[1, 1, 1],
                         [0.2, 1, 1],
                         [1, 1, 1]])
        dthresh = 1.0
        ffrac = 0.8
        fwidth = 5

        target = np.array([[0, 0, 0],
                           [1, 0, 1],
                           [0, 0, 0]]).astype('bool')
        out = compute_fixedbp_excam(dark, flat, dthresh, ffrac, fwidth)

        self.assertTrue((out == target).all())

    def test_exact_nonuniform_darkflat_same(self):
        """Same bad pixel in dark *and* flat handled correctly."""
        dark = np.array([[10, 10, 10],
                         [10, 10, 100],
                         [10, 10, 10]])
        flat = np.array([[1, 1, 1],
                         [1, 1, 0.2],
                         [1, 1, 1]])
        dthresh = 1.0
        ffrac = 0.8
        fwidth = 5

        target = np.array([[0, 0, 0],
                           [0, 0, 1],
                           [0, 0, 0]]).astype('bool')
        out = compute_fixedbp_excam(dark, flat, dthresh, ffrac, fwidth)

        self.assertTrue((out == target).all())

    def test_invalid_dark(self):
        """Invalid inputs caught."""
        xlist = [1, -1, 0, 1.5, 1j, 'txt', None, (5,),
                 np.ones((5,)), np.ones((5, 5, 2))]

        for x in xlist:
            with self.assertRaises(TypeError):
                compute_fixedbp_excam(x, self.flat, self.dthresh, self.ffrac,
                            self.fwidth)

    def test_invalid_flat(self):
        """Invalid inputs caught."""
        xlist = [1, -1, 0, 1.5, 1j, 'txt', None, (5,),
                 np.ones((5,)), np.ones((5, 5, 2))]

        for x in xlist:
            with self.assertRaises(TypeError):
                compute_fixedbp_excam(self.dark, x, self.dthresh, self.ffrac,
                            self.fwidth)

    def test_invalid_dthresh(self):
        """Invalid inputs caught."""
        xlist = [-1, 1j, 'txt', None, (5,),
                 np.ones((5,)), np.ones((5, 5, 2))]

        for x in xlist:
            with self.assertRaises(TypeError):
                compute_fixedbp_excam(self.dark, self.flat, x, self.ffrac,
                            self.fwidth)

    def test_invalid_ffrac(self):
        """Invalid inputs caught."""
        xlist = [-1, 1j, 'txt', None, (5,),
                 np.ones((5,)), np.ones((5, 5, 2))]

        for x in xlist:
            with self.assertRaises(TypeError):
                compute_fixedbp_excam(self.dark, self.flat, self.dthresh, x,
                            self.fwidth)

    def test_invalid_fwidth(self):
        """Invalid inputs caught."""
        xlist = [-1, 0, 1.5, 1j, 'txt', None, (5,),
                 np.ones((5,)), np.ones((5, 5, 2))]

        for x in xlist:
            with self.assertRaises(TypeError):
                compute_fixedbp_excam(
                    self.dark, self.flat, self.dthresh, self.ffrac, x)



class TestBPMapForDarks(unittest.TestCase):
    """Tests for building the fixed bad-pixel map from a dark."""

    def setUp(self):
        """Set up."""
        self.dark = np.zeros((10, 10))
        self.dthresh = 5

    def test_success(self):
        """Valid inputs complete successfully."""
        _compute_fixedbp_from_dark(self.dark, self.dthresh)

    def test_success_clean(self):
        """Valid inputs complete successfully for clean-sized frame."""
        dark = np.zeros((1024, 1024))
        _compute_fixedbp_from_dark(dark, self.dthresh)

    def test_success_nonsquare(self):
        """Valid inputs complete successfully for nonsquare frame."""
        dark = np.zeros((10, 11))
        _compute_fixedbp_from_dark(dark, self.dthresh)

    def test_exact_uniform(self):
        """Uniform dark gives no bad pixels."""
        # Use tight constraints, shouldn't matter for uniform
        dark = np.zeros((10, 10))
        dthresh = 0
        target = np.zeros_like(dark).astype('bool')
        out = _compute_fixedbp_from_dark(dark, dthresh)

        self.assertTrue((out == target).all())

    def test_exact_nonuniform_darkonly(self):
        """Bad pixel in dark caught."""
        dark = np.array([[10, 10, 10],
                         [10, 10, 100],
                         [10, 10, 10]])
        dthresh = 1.0
        target = np.array([[0, 0, 0],
                           [0, 0, 1],
                           [0, 0, 0]]).astype('bool')
        out = _compute_fixedbp_from_dark(dark, dthresh)

        self.assertTrue((out == target).all())

    def test_invalid_dark(self):
        """Invalid inputs caught."""
        xlist = [1, -1, 0, 1.5, 1j, 'txt', None, (5,),
                 np.ones((5,)), np.ones((5, 5, 2))]

        for x in xlist:
            with self.assertRaises(TypeError):
                _compute_fixedbp_from_dark(x, self.dthresh)

    def test_invalid_dthresh(self):
        """Invalid inputs caught."""
        xlist = [-1, 1j, 'txt', None, (5,),
                 np.ones((5,)), np.ones((5, 5, 2))]

        for x in xlist:
            with self.assertRaises(TypeError):
                _compute_fixedbp_from_dark(self.dark, x)


class TestBPMapForFlats(unittest.TestCase):
    """Tests for building the fixed bad-pixel map from a flat."""

    def setUp(self):
        """Set up."""
        self.flat = np.ones((10, 10))
        self.ffrac = 0.8
        self.fwidth = 4

    def test_success(self):
        """Valid inputs complete successfully."""
        _compute_fixedbp_from_flat(self.flat, self.ffrac, self.fwidth)

    def test_success_clean(self):
        """Valid inputs complete successfully for clean-sized frame."""
        flat = np.ones((1024, 1024))
        _compute_fixedbp_from_flat(flat, self.ffrac, self.fwidth)

    def test_success_nonsquare(self):
        """Valid inputs complete successfully for nonsquare frame."""
        flat = np.ones((10, 11))
        _compute_fixedbp_from_flat(flat, self.ffrac, self.fwidth)

    def test_exact_uniform(self):
        """Uniform flat gives no bad pixels."""
        # Use tight constraints, shouldn't matter for uniform
        flat = np.ones((10, 10))
        ffrac = 1.0
        fwidth = 5
        target = np.zeros_like(flat).astype('bool')
        out = _compute_fixedbp_from_flat(flat, ffrac, fwidth)

        self.assertTrue((out == target).all())

    def test_exact_nonuniform_flatonly(self):
        """Bad pixel in flat caught."""
        flat = np.array([[1, 1, 1],
                         [0.2, 1, 1],
                         [1, 1, 1]])
        ffrac = 0.8
        fwidth = 5
        target = np.array([[0, 0, 0],
                           [1, 0, 0],
                           [0, 0, 0]]).astype('bool')
        out = _compute_fixedbp_from_flat(flat, ffrac, fwidth)

        self.assertTrue((out == target).all())

    def test_invalid_flat(self):
        """Invalid inputs caught."""
        xlist = [1, -1, 0, 1.5, 1j, 'txt', None, (5,),
                 np.ones((5,)), np.ones((5, 5, 2))]

        for x in xlist:
            with self.assertRaises(TypeError):
                _compute_fixedbp_from_flat(x, self.ffrac, self.fwidth)

    def test_invalid_ffrac(self):
        """Invalid inputs caught."""
        xlist = [-1, 1j, 'txt', None, (5,),
                 np.ones((5,)), np.ones((5, 5, 2))]

        for x in xlist:
            with self.assertRaises(TypeError):
                _compute_fixedbp_from_flat(self.flat, x, self.fwidth)

    def test_invalid_fwidth(self):
        """Invalid inputs caught."""
        xlist = [-1, 0, 1.5, 1j, 'txt', None, (5,),
                 np.ones((5,)), np.ones((5, 5, 2))]

        for x in xlist:
            with self.assertRaises(TypeError):
                _compute_fixedbp_from_flat(self.flat, self.ffrac, x)


if __name__ == '__main__':
    unittest.main()
