"""Unit tests for the findopt.py utility functions."""
import unittest
from math import isclose

import numpy as np

from .math import rms, ceil_odd, ceil_even


class TestFindoptFunctions(unittest.TestCase):
    """Test all normal use cases."""

    def test_rms(self):
        """Test that rms returns the root-mean-square of an array."""
        self.assertTrue(isclose(np.sqrt(2)/2., rms(np.sin(2*np.pi
                       * np.linspace(0., 1., 10001))), rel_tol=0.01))

        array = np.ones((10, 10))
        array[:, ::2] = -1
        self.assertTrue(isclose(1., rms(array)))

    def test_ceil_odd(self):
        """Test that ceil_even returns the next highest even integer."""
        self.assertTrue(ceil_odd(1) == 1)
        self.assertTrue(ceil_odd(1.5) == 3)
        self.assertTrue(ceil_odd(4) == 5)
        self.assertTrue(ceil_odd(3.14159) == 5)
        self.assertTrue(ceil_odd(2001) == 2001)

    def test_ceil_even(self):
        """Test that ceil_even returns the next highest even integer."""
        self.assertTrue(ceil_even(1) == 2)
        self.assertTrue(ceil_even(1.5) == 2)
        self.assertTrue(ceil_even(4) == 4)
        self.assertTrue(ceil_even(3.14159) == 4)
        self.assertTrue(ceil_even(2001) == 2002)


class TestInputFailure(unittest.TestCase):
    """Test suite for valid function inputs."""

    def test_ceil_odd_input(self):
        """Test incorrect inputs of ceil_odd."""
        with self.assertRaises(TypeError):
            ceil_odd('this is a string')
        with self.assertRaises(TypeError):
            ceil_odd(np.array([2.0, 3.1]))

    def test_ceil_even_input(self):
        """Test incorrect inputs of ceil_even."""
        with self.assertRaises(TypeError):
            ceil_even('this is a string')
        with self.assertRaises(TypeError):
            ceil_even(np.array([2.0, 3.1]))

    def test_rms_input(self):
        """Test incorrect inputs of rms."""
        values = ('a')
        for val in values:
            with self.assertRaises(TypeError):
                rms(val)


if __name__ == '__main__':
    unittest.main()
