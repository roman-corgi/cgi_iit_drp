"""Test suite for extend_radially.py."""
import unittest
import numpy as np

from .extend_radially import extend_radially
from .insertinto import insertinto as inin

not_a_2d_array = (True, -1, 1, 1.1, 1j, np.ones((5, )), 'string', {'a': 2})
not_a_real_scalar = (1j, 1j, np.ones(5), 'string', {'a': 2})
not_a_real_positive_scalar = (-1, 0, 1j, np.ones(5), 'string', {'a': 2})
not_valid_direction_values = (-1.1, 0, 2, 1j, np.ones(5), 'string', {'a': 2})

class TestInputFailures(unittest.TestCase):
    """Test suite for valid function inputs."""

    def setUp(self):
        
        self.x_offset = 0.
        self.y_offset = 0.
        array_in = np.zeros((5, 5))
        array_in[2, 2] = 3.0
        array_in[3, 3] = 2.0
        array_in[1, 2] = 5.0
        self.array_in = array_in
        self.direction = 1
        self.atol = 1e-10

        self.array_expected = 3*np.ones((5, 5))
        self.array_expected[0:2, 2] = 5.0
        self.array_expected[3::, 3::] = 2.0

    def test_check_standard_inputs(self):
        """Verify standard inputs produce a good result."""
        array_out = extend_radially(
            self.array_in,
            x_offset=self.x_offset,
            y_offset=self.y_offset,
            direction=self.direction,
            atol=self.atol,
        ) 
        self.assertTrue(np.array_equal(array_out, self.array_expected))

    def test_x_offset_out_of_range(self):
        """Test out-of-range offset."""
        with self.assertRaises(ValueError):
            extend_radially(
                self.array_in,
                x_offset=2.6,
                y_offset=self.y_offset,
                atol=self.atol,
            )

    def test_y_offset_out_of_range(self):
        """Test out-of-range offset."""
        with self.assertRaises(ValueError):
            extend_radially(
                self.array_in,
                x_offset=self.x_offset,
                y_offset=-2.6,
                atol=self.atol,
            )

    def test_input_0(self):
        """Test bad input type."""
        for bad_val in not_a_2d_array:
            with self.assertRaises(TypeError):
                extend_radially(
                    bad_val,
                    x_offset=self.x_offset,
                    y_offset=self.y_offset,
                    atol=self.atol,
                )

    def test_input_1(self):
        """Test bad input type."""
        for bad_val in not_a_real_scalar:
            with self.assertRaises(TypeError):
                extend_radially(
                    self.array_in,
                    x_offset=bad_val,
                    y_offset=self.y_offset,
                    atol=self.atol,
                )

    def test_input_2(self):
        """Test bad input type."""
        for bad_val in not_a_real_scalar:
            with self.assertRaises(TypeError):
                extend_radially(
                    self.array_in,
                    x_offset=self.x_offset,
                    y_offset=bad_val,
                    atol=self.atol,
                )

    def test_input_3(self):
        """Test bad input type."""
        for bad_val in not_valid_direction_values:
            with self.assertRaises(TypeError):
                extend_radially(
                    self.array_in,
                    x_offset=self.x_offset,
                    y_offset=self.y_offset,
                    direction=bad_val,
                )

    def test_input_4(self):
        """Test bad input type."""
        for bad_val in not_a_real_positive_scalar:
            with self.assertRaises(TypeError):
                extend_radially(
                    self.array_in,
                    x_offset=self.x_offset,
                    y_offset=self.y_offset,
                    atol=bad_val,
                )


class PerformanceTests(unittest.TestCase):
    """More performance tests."""

    def test_centered_on_corner(self):
        """Test that a corner center fills in the rest of the array."""
        array_in = np.zeros((3, 3))
        array_in[0, 0] = 1
        array_out = extend_radially(array_in, x_offset=-1, y_offset=-1)
        self.assertTrue(np.array_equal(array_out, np.ones((3, 3))))

    def test_centered_on_opposite_corner(self):
        """Test that an opposite corner center changes nothing."""
        array_in = np.zeros((3, 3))
        array_in[0, 0] = 1
        array_out = extend_radially(array_in, x_offset=1, y_offset=1)
        self.assertTrue(np.array_equal(array_out, array_in))
        
    def test_even_sized_array(self):
        """Test that an even-sized array is filled out from the center."""
        array_in = np.zeros((4, 4))
        array_in[2, 2] = 1
        array_out = extend_radially(array_in)
        self.assertTrue(np.array_equal(array_out, np.ones((4, 4))))

    def test_mixed_sized_array(self):
        """Test that a mixed-sized array is filled out from the center."""
        array_in = np.zeros((4, 5))
        array_in[2, 2] = 1
        array_out = extend_radially(array_in)
        self.assertTrue(np.array_equal(array_out, np.ones((4, 5))))

    def test_float_offsets(self):
        """Test floating point offsets work."""
        array_in = np.zeros((3, 3))
        array_in[0, 0] = 1
        array_out = extend_radially(array_in, x_offset=-1.4, y_offset=-0.9)
        self.assertTrue(np.array_equal(array_out, np.ones((3, 3))))

    def test_inward_centered_outer_ring_odd(self):
        """Test that an array is filled inward in an odd-sized array."""
        array_in = np.zeros((5, 5))
        array_in[[0, -1], :] = 1
        array_in[:, [0, -1]] = 1
        array_out = extend_radially(array_in, direction=-1)
        array_expected = np.ones((5, 5))
        array_expected[2, 2] = 0
        self.assertTrue(np.array_equal(array_out, array_expected))

    def test_inward_centered_outer_ring_even(self):
        """Test that an array is filled inward in an even-sized array."""
        array_in = np.zeros((4, 4))
        array_in[[0, -1], :] = 1
        array_in[:, [0, -1]] = 1
        array_out = extend_radially(array_in, direction=-1)
        array_expected = np.ones((4, 4))
        array_expected[2, 2] = 0
        self.assertTrue(np.array_equal(array_out, array_expected))

    def test_inward_centered_outer_ring_even_offset(self):
        """Test that an array is filled inward completely with an offset."""
        array_in = np.zeros((4, 4))
        array_in[[0, -1], :] = 1
        array_in[:, [0, -1]] = 1
        array_out = extend_radially(array_in,
                                    x_offset=-0.5,
                                    y_offset=-0.5,
                                    direction=-1)
        array_expected = np.ones((4, 4))
        self.assertTrue(np.array_equal(array_out, array_expected))
        
    def test_inward_centered_middle_ring(self):
        """Test that an array is filled inward inside a middle ring."""
        array_in = np.zeros((4, 4))
        array_in[[0, -1], :] = 1
        array_in[:, [0, -1]] = 1
        array_in = inin(array_in, (6, 6))
        array_out = extend_radially(array_in, direction=-1)

        array_expected = np.ones((4, 4))
        array_expected[2, 2] = 0
        array_expected = inin(array_expected, (6, 6))
        self.assertTrue(np.array_equal(array_out, array_expected))

if __name__ == '__main__':
    unittest.main()
