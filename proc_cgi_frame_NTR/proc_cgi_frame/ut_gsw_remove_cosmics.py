# -*- coding: utf-8 -*-
"""Unit tests for remove_cosmics."""
from __future__ import absolute_import, division, print_function

import copy
import unittest
from unittest.mock import patch

import numpy as np

from .gsw_remove_cosmics import find_plateaus, remove_cosmics
from . import gsw_remove_cosmics

# Input parameters
fwc = 10000.
sat_thresh = 0.99
plat_thresh = 0.85
cosm_filter = 2

# Create a bias subtracted image with cosmics that cover all corner cases
# Make a variety of plateaus
p_basic = np.array([fwc]*cosm_filter)  # Smallest allowed through filter
p_small = np.array([fwc]*(cosm_filter-1))  # Smaller than filter
p_large = np.array([fwc]*cosm_filter*10)  # Larger than filter
p_dip = np.append(p_basic, [plat_thresh*fwc, fwc])  # Shallow dip mid cosmic
p_dip_deep = np.hstack((p_basic, [0.], p_basic))  # Deep dip mid cosmic
p_uneven = np.array([fwc*sat_thresh, fwc, fwc*plat_thresh, fwc,
                     fwc*plat_thresh])  # Uneven cosmic
p_below_min = np.array([fwc*sat_thresh - 1]*cosm_filter)  # Below min value

# Create tail
# An exponential tail with no noise should be able to be perfectly removed
tail = np.exp(np.linspace(0, -10, 50)) * 0.1*fwc

# Create streak row
streak_row = np.ones(1000)

# Create mask rows
c_mask_row = np.zeros(len(streak_row), dtype=int)
t_mask_row = np.zeros(len(streak_row), dtype=int)

# Create bias subtracted image
i_streak_rows_t = np.array([0, 499, 500, 999])
cosm_bs = np.append(p_basic, tail)
not_cosm_bs = np.append(p_below_min, tail)

bs_image = np.ones((len(streak_row), 1000))
bs_image_below_thresh = np.ones(bs_image.shape)
bs_image_single_pix = np.ones(bs_image.shape)
bs_image_two_cosm = np.ones(bs_image.shape)
bs_image[i_streak_rows_t[0], 0:len(cosm_bs)] = cosm_bs
bs_image[i_streak_rows_t[1], 50:50+len(cosm_bs)] = cosm_bs
bs_image[i_streak_rows_t[1], 50+len(cosm_bs):50+len(cosm_bs)*2] = cosm_bs
bs_image[i_streak_rows_t[2], 51:51+len(cosm_bs)] = cosm_bs
bs_image[i_streak_rows_t[3], bs_image.shape[1]-len(p_basic):] = p_basic
bs_image_below_thresh[500, 50:50+len(not_cosm_bs)] = not_cosm_bs
bs_image_single_pix[500, 500] = fwc


class TestRemoveCosmics(unittest.TestCase):
    """Unit tests for remove_cosmics function."""

    def setUp(self):
        self.bs_image = bs_image
        self.fwc = fwc
        self.sat_thresh = sat_thresh
        self.plat_thresh = plat_thresh
        self.cosm_filter = cosm_filter

        self.bs_image_below_thresh = bs_image_below_thresh
        self.bs_image_single_pix = bs_image_single_pix
        self.bs_image_two_cosm = bs_image_two_cosm
        self.i_streak_rows_t = i_streak_rows_t
        self.clean_row = np.zeros(len(self.bs_image))
        self.c_mask_row = copy.copy(c_mask_row)
        self.t_mask_row = copy.copy(t_mask_row)
        self.p_basic = p_basic
        self.tail = tail
        self.cosm = np.append(self.p_basic, self.tail)


    @patch.object(gsw_remove_cosmics,'find_plateaus')
    def test_called_find_plateaus(self, mock_find_plateaus):
        """Assert that find_plateaus is called for each streak row."""
        mock_find_plateaus.return_value = 0 # i.e. index 0 starts each plateau
        remove_cosmics(self.bs_image, self.fwc, self.sat_thresh,
                       self.plat_thresh, self.cosm_filter)
        self.assertEqual(mock_find_plateaus.call_count,
                         len(self.i_streak_rows_t))

    @patch.object(gsw_remove_cosmics, 'find_plateaus')
    def test_not_called_find_plateaus(self, mock_find_plateaus):
        """Assert that find_plateaus not called if no cosmic rows."""
        mock_find_plateaus.return_value = (np.array([], dtype=int),
                                           np.array([], dtype=int))
        remove_cosmics(self.bs_image_below_thresh, self.fwc, self.sat_thresh,
                       self.plat_thresh, self.cosm_filter)
        self.assertFalse(mock_find_plateaus.called)

    def test_mask(self):
        """Assert correct elements are masked."""
        bs_image = np.ones_like(self.bs_image)
        bs_image[1, 2:2+len(self.cosm)] = self.cosm
        check_mask = np.zeros_like(self.bs_image, dtype=int)
        check_mask[1, 1:] = 1  # Mask starts 1 before cosmic

        mask = remove_cosmics(bs_image, self.fwc, self.sat_thresh,
                              self.plat_thresh, self.cosm_filter)
        np.testing.assert_array_equal(mask, check_mask)


    def test_no_rows_mask(self):
        """Assert mask array is blank if no cosmic rows."""
        mask = remove_cosmics(self.bs_image_below_thresh, self.fwc,
                              self.sat_thresh, self.plat_thresh,
                              self.cosm_filter)
        np.testing.assert_array_equal(mask, np.zeros(self.bs_image.shape))


class TestFindPlateaus(unittest.TestCase):
    """Unit tests for find_plateaus function.

    Remember that the value within i_beg is one pixel beyond the plateau edge.
    """

    def setUp(self):
        self.streak_row = copy.copy(streak_row)
        self.fwc = fwc
        self.sat_thresh = sat_thresh
        self.plat_thresh = plat_thresh
        self.cosm_filter = cosm_filter

        self.p_basic = p_basic
        self.p_small = p_small
        self.p_large = p_large
        self.p_dip = p_dip
        self.p_dip_deep = p_dip_deep
        self.p_uneven = p_uneven
        self.p_below_min = p_below_min
        self.tail = tail
        self.cosm = np.append(self.p_basic, self.tail)

    def test_i_begs(self):
        """Verify that function returns correct i_begs result."""
        beg = 50
        self.streak_row[beg:beg+len(self.cosm)] = self.cosm
        i_beg = find_plateaus(self.streak_row, self.fwc, self.sat_thresh,
                              self.plat_thresh, self.cosm_filter)
        self.assertEqual(i_beg, beg-1)


    def test_left_edge_i_begs(self):
        """Verify that function returns correct i_begs result at left edge."""
        beg = 0
        self.streak_row[beg:len(self.cosm)] = self.cosm
        i_beg = find_plateaus(self.streak_row, self.fwc, self.sat_thresh,
                              self.plat_thresh, self.cosm_filter)
        self.assertEqual(i_beg, beg)


    def test_right_edge_i_begs(self):
        """Verify that function returns correct i_begs result at right edge."""
        cosm = self.p_basic
        beg = len(self.streak_row)-len(cosm)
        self.streak_row[beg:] = cosm
        i_beg = find_plateaus(self.streak_row, self.fwc, self.sat_thresh,
                              self.plat_thresh, self.cosm_filter)
        self.assertEqual(i_beg, beg-1)


    def test_two_cosm_i_begs(self):
        """Verify that function returns correct i_begs result for two cosm."""
        beg1 = len(self.streak_row) - len(self.cosm)*2
        beg2 = beg1 + len(self.cosm)
        self.streak_row[beg1:beg1 + len(self.cosm)] = self.cosm
        self.streak_row[beg2:beg2 + len(self.cosm)] = self.cosm

        i_beg = find_plateaus(self.streak_row, self.fwc, self.sat_thresh,
                              self.plat_thresh, self.cosm_filter)
        self.assertEqual(i_beg, beg1-1)


    def test_p_small(self):
        """Verify that function ignores plateaus smaller than filter size."""
        cosm = np.append(self.p_small, self.tail)
        beg = 50
        self.streak_row[beg:beg+len(cosm)] = cosm
        i_beg = find_plateaus(self.streak_row, self.fwc,
                              self.sat_thresh, self.plat_thresh,
                              self.cosm_filter)
        self.assertIsNone(i_beg)


    def test_p_large(self):
        """Verify that function returns correct results for large plateaus."""
        cosm = np.append(self.p_large, self.tail)
        beg = 50
        self.streak_row[beg:beg+len(cosm)] = cosm
        i_beg = find_plateaus(self.streak_row, self.fwc,
                              self.sat_thresh, self.plat_thresh,
                              self.cosm_filter)
        self.assertEqual(i_beg, beg-1)


    def test_p_dip(self):
        """Verify that function still recognizes a plateau with a dip."""
        cosm = np.append(self.p_dip, self.tail)
        beg = 50
        self.streak_row[beg:beg+len(cosm)] = cosm
        i_beg = find_plateaus(self.streak_row, self.fwc,
                              self.sat_thresh, self.plat_thresh,
                              self.cosm_filter)
        self.assertEqual(i_beg, beg-1)


    def test_p_dip_deep(self):
        """Verify that the function recognizes plateau with a single pixel dip
        below plat_thresh and does not set the end at the dip."""
        cosm = np.append(self.p_dip_deep, self.tail)
        beg = 50
        self.streak_row[beg:beg+len(cosm)] = cosm
        i_beg = find_plateaus(self.streak_row, self.fwc,
                              self.sat_thresh, self.plat_thresh,
                              self.cosm_filter)
        self.assertEqual(i_beg, beg-1)


    def test_p_uneven(self):
        """Verify that function still recognizes an uneven plateau."""
        cosm = np.append(self.p_uneven, self.tail)
        beg = 50
        self.streak_row[beg:beg+len(cosm)] = cosm
        i_beg = find_plateaus(self.streak_row, self.fwc,
                              self.sat_thresh, self.plat_thresh,
                              self.cosm_filter)
        self.assertEqual(i_beg, beg-1)


    def test_p_below_min(self):
        """Verify that function ignores plateaus below saturation thresh."""
        cosm = np.append(self.p_below_min, self.tail)
        beg = 50
        self.streak_row[beg:beg+len(cosm)] = cosm
        i_beg = find_plateaus(self.streak_row, self.fwc,
                              self.sat_thresh, self.plat_thresh,
                              self.cosm_filter)
        self.assertIsNone(i_beg)



if __name__ == '__main__':
    unittest.main()
