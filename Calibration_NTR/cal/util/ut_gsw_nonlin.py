"""Unit tests for nonlin."""
# NOTE TO FSW: THIS MODULE IS ONLY USED BY THE GROUND PIPELINE, DO NOT PORT

import unittest
import os

import numpy as np

from .gsw_nonlin import get_relgains, _parse_file
from .gsw_nonlin import NonlinException


# Create nonlin array to be written to csv
# # Row headers are counts
check_count_ax = np.array([1, 1000, 2000, 3000])
# Column headers are em gains
check_gain_ax = np.array([1, 10, 100, 1000, 2000])
# Relative gain data, valid curves
curve1 = np.array([0.900, 0.910, 0.950, 1.000])  # Last value is exactly 1
curve2 = np.array([0.950, 0.950, 1.000, 1.001])  # Mid value is exactly 1
curve3 = np.array([0.989, 0.990, 1.010, 1.011])  # Values straddle 1
curve4 = np.array([1.000, 1.010, 1.050, 1.060])  # First value is exactly 1
curve5 = np.array([1.000, 1.010, 1.060, 1.050])  # Peaks in middle
# Fill array with relative gain curves
check_relgains = np.zeros((len(check_count_ax), len(check_gain_ax)))
check_relgains[:, 0] = curve1
check_relgains[:, 1] = curve2
check_relgains[:, 2] = curve3
check_relgains[:, 3] = curve4
check_relgains[:, 4] = curve5

# Build nonlin array
nonlin_array = np.zeros((len(check_count_ax)+1, len(check_gain_ax)+1))
nonlin_array[0, 0] = np.nan  # Top left corner is set to nan
nonlin_array[1:, 0] = check_count_ax
nonlin_array[0, 1:] = check_gain_ax
nonlin_array[1:, 1:] = check_relgains

localpath = os.path.dirname(os.path.abspath(__file__))

class TestParseFile(unittest.TestCase):
    """Unit tests for _parse_file function."""

    def setUp(self):
        self.nonlin_path = os.path.join(localpath,
                                        'testdata', 'ut_nonlin_array.txt')

        self.check_count_ax = check_count_ax
        self.check_gain_ax = check_gain_ax
        self.check_relgains = check_relgains

    def test_gain_ax(self):
        """Verify function returns correct values for gain axis."""
        gain_ax, _, _ = _parse_file(self.nonlin_path)
        self.assertEqual(gain_ax.tolist(), self.check_gain_ax.tolist())

    def test_count_ax(self):
        """Verify function returns correct values for count axis."""
        _, count_ax, _ = _parse_file(self.nonlin_path)
        self.assertEqual(count_ax.tolist(), self.check_count_ax.tolist())

    def test_relgains(self):
        """Verify function returns correct values for relative gains."""
        _, _, relgains = _parse_file(self.nonlin_path)
        self.assertEqual(relgains.tolist(), self.check_relgains.tolist())

    def test_exception_2_rows(self):
        """Verify function throws exception for csv less than 2 rows."""
        nonlin_path_bad = os.path.join(localpath,
                                       'testdata',
                                       'ut_nonlin_array_1_row.txt'
        )
        with self.assertRaises(NonlinException):
            _parse_file(nonlin_path_bad)

    def test_exception_2_cols(self):
        """Verify function throws exception for csv less than 2 columns."""
        nonlin_path_bad = os.path.join(localpath,
                                       'testdata',
                                       'ut_nonlin_array_1_col.txt'
        )
        with self.assertRaises(NonlinException):
            _parse_file(nonlin_path_bad)

    def test_exception_first_val_nan(self):
        """Verify function throws exception if first value is not nan."""
        nonlin_path_bad = os.path.join(localpath,
                                       'testdata',
                                       'ut_nonlin_array_no_nan.txt'
        )
        with self.assertRaises(NonlinException):
            _parse_file(nonlin_path_bad)

    def test_exception_gain_axis_decreasing(self):
        """Verify function throws exception if gain axis is decreasing."""
        nonlin_path_bad = os.path.join(localpath,
                                       'testdata',
                                       'ut_nonlin_array_gain_decr.txt'
        )
        with self.assertRaises(NonlinException):
            _parse_file(nonlin_path_bad)

    def test_exception_gain_axis_same(self):
        """Verify function throws exception if gain axis is not increasing."""
        nonlin_path_bad = os.path.join(localpath,
                                       'testdata',
                                       'ut_nonlin_array_gain_same.txt'
        )
        with self.assertRaises(NonlinException):
            _parse_file(nonlin_path_bad)

    def test_exception_count_axis_decreasing(self):
        """Verify function throws exception if count axis is decreasing."""
        nonlin_path_bad = os.path.join(localpath,
                                       'testdata',
                                       'ut_nonlin_array_count_decr.txt'
        )
        with self.assertRaises(NonlinException):
            _parse_file(nonlin_path_bad)

    def test_exception_count_axis_same(self):
        """Verify function throws exception if count axis is not increasing."""
        nonlin_path_bad = os.path.join(localpath,
                                       'testdata',
                                       'ut_nonlin_array_count_same.txt'
        )
        with self.assertRaises(NonlinException):
            _parse_file(nonlin_path_bad)

    def test_exception_relgain_low(self):
        """
        Verify function throws exception if a relative gain curve does not
        straddle 1 (always low).
        """
        nonlin_path_bad = os.path.join(localpath,
                                       'testdata',
                                       'ut_nonlin_array_relgain_low.txt'
        )
        with self.assertRaises(NonlinException):
            _parse_file(nonlin_path_bad)

    def test_exception_relgain_high(self):
        """
        Verify function throws exception if a relative gain curve does not
        straddle 1 (always high).
        """
        nonlin_path_bad = os.path.join(localpath,
                                       'testdata',
                                       'ut_nonlin_array_relgain_high.txt'
        )
        with self.assertRaises(NonlinException):
            _parse_file(nonlin_path_bad)


class TestGetRelgains(unittest.TestCase):
    """Unit tests for get_relgains function."""

    def setUp(self):
        # Create a temporary file to act as the nonlinearity csv
        self.nonlin_path = os.path.join(localpath,
                                        'testdata', 'ut_nonlin_array.txt')

        self.check_count_ax = check_count_ax
        self.check_gain_ax = check_gain_ax
        self.check_relgains = check_relgains

        self.frame_ones = np.ones((10, 10))


    def test_both_ax_all(self):
        """
        Verify function returns the expected array for all gain/count combos.

        Includes the peak-in-middle case to satisfy #42.

        """
        for row in range(len(self.check_count_ax)):
            for col in range(len(self.check_gain_ax)):
                frame_counts = self.check_count_ax[row] * self.frame_ones
                em_gain = self.check_gain_ax[col]

                frame_relgains = get_relgains(frame_counts,
                                              em_gain,
                                              self.nonlin_path)

                check_frame_relgains = (self.check_relgains[row, col]
                                        *self.frame_ones)
                self.assertEqual(frame_relgains.tolist(),
                                 check_frame_relgains.tolist())
                pass
            pass
        pass


    def test_count_ax_interp(self):
        """Verify function performs linear interoplation over count axis."""
        row = 1
        col = 0
        # Choose the midpoint between two points on the count axis
        mid_count_ax = (self.check_count_ax[row]
                        + self.check_count_ax[row+1]) / 2
        frame_counts = mid_count_ax * self.frame_ones
        em_gain = self.check_gain_ax[col]

        frame_relgains = get_relgains(frame_counts, em_gain, self.nonlin_path)

        mid_relgains = (self.check_relgains[row, col]
                        + self.check_relgains[row+1, col]) / 2
        check_frame_relgains = mid_relgains * self.frame_ones
        self.assertEqual(frame_relgains.tolist(),
            check_frame_relgains.tolist())

    def test_gain_ax_interp(self):
        """Verify function performs linear interoplation over em gain axis."""
        row = 0
        col = 1
        frame_counts = self.check_count_ax[row] * self.frame_ones
        # Choose the midpoint between two points on the gain axis
        em_gain = (self.check_gain_ax[col] + self.check_gain_ax[col+1]) / 2

        frame_relgains = get_relgains(frame_counts, em_gain, self.nonlin_path)

        mid_relgains = (self.check_relgains[row, col]
                        + self.check_relgains[row, col+1]) / 2
        check_frame_relgains = mid_relgains * self.frame_ones
        self.assertEqual(frame_relgains.tolist(),
            check_frame_relgains.tolist())

    def test_both_ax_below_min(self):
        """Verify function returns the expected array when the input gain and
        counts are both set below the minimum supplied axis values.

        """
        row = 0
        col = 0
        frame_counts = self.frame_ones * self.check_count_ax[row]-1
        em_gain = self.check_gain_ax[col]-1

        frame_relgains = get_relgains(frame_counts, em_gain, self.nonlin_path)

        check_frame_relgains = self.check_relgains[row, col] * self.frame_ones
        self.assertEqual(frame_relgains.tolist(),
            check_frame_relgains.tolist())

    def test_both_ax_above_max(self):
        """Verify function returns the expected array when the input gain and
        counts are both set above the maximum supplied axis values.

        """
        row = -1
        col = -1
        frame_counts = self.frame_ones * self.check_count_ax[row]+1
        em_gain = self.check_gain_ax[col]+1

        frame_relgains = get_relgains(frame_counts, em_gain, self.nonlin_path)

        check_frame_relgains = self.check_relgains[row, col] * self.frame_ones
        self.assertEqual(frame_relgains.tolist(),
            check_frame_relgains.tolist())


if __name__ == '__main__':
    unittest.main()
