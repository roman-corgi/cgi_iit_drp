# -*- coding: utf-8 -*-
"""Unit tests for gsw_remove_cosmics."""
from __future__ import absolute_import, division, print_function

import copy
import unittest
from unittest.mock import patch

import numpy as np

from .gsw_remove_cosmics import find_plateaus, remove_cosmics
from . import gsw_remove_cosmics
from .read_metadata import Metadata

# Input parameters
fwc = 10000.
sat_thresh = 0.99
plat_thresh = 0.85
cosm_filter = 2
cosm_tail = 1024 # cosmic mask goes to end of row in all cases
cosm_box = 0

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
bs_image_box = np.ones(bs_image.shape)
bs_image[i_streak_rows_t[0], 0:len(cosm_bs)] = cosm_bs
bs_image[i_streak_rows_t[1], 50:50+len(cosm_bs)] = cosm_bs
bs_image[i_streak_rows_t[1], 50+len(cosm_bs):50+len(cosm_bs)*2] = cosm_bs
bs_image[i_streak_rows_t[2], 51:51+len(cosm_bs)] = cosm_bs
bs_image[i_streak_rows_t[3], bs_image.shape[1]-len(p_basic):] = p_basic
bs_image_below_thresh[500, 50:50+len(not_cosm_bs)] = not_cosm_bs
bs_image_single_pix[500, 500] = fwc
bs_image_box[i_streak_rows_t[1], 50:50+len(cosm_bs)] = cosm_bs
# these pixels surrounding the cosmic head would not get masked
# unless cosm_box > 0; doesn't form a full box,
# but the whole box should get masked
bs_image_box[i_streak_rows_t[1]-2:i_streak_rows_t[1], 50-2:50+2+1] = \
    0.6*fwc
meta = Metadata() # using metadata.yaml
im_num_rows = meta.geom['image']['rows']
im_num_cols = meta.geom['image']['cols']
im_starting_row = meta.geom['image']['r0c0'][0]
im_ending_row = im_starting_row + im_num_rows
im_starting_col = meta.geom['image']['r0c0'][1]
im_ending_col = im_starting_col + im_num_cols

class TestRemoveCosmics(unittest.TestCase):
    """Unit tests for remove_cosmics function."""

    def setUp(self):
        self.bs_image = bs_image
        self.fwc = fwc
        self.sat_thresh = sat_thresh
        self.plat_thresh = plat_thresh
        self.cosm_filter = cosm_filter
        self.cosm_box = cosm_box
        self.cosm_tail = cosm_tail

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
        # i.e. index 0 starts each plateau
        mock_find_plateaus.return_value = np.array([0])
        remove_cosmics(self.bs_image, self.fwc, self.sat_thresh,
                       self.plat_thresh, self.cosm_filter, self.cosm_box,
                       self.cosm_tail)
        self.assertEqual(mock_find_plateaus.call_count,
                         len(self.i_streak_rows_t))

    @patch.object(gsw_remove_cosmics, 'find_plateaus')
    def test_not_called_find_plateaus(self, mock_find_plateaus):
        """Assert that find_plateaus not called if no cosmic rows."""
        mock_find_plateaus.return_value = (np.array([], dtype=int),
                                           np.array([], dtype=int))
        remove_cosmics(self.bs_image_below_thresh, self.fwc, self.sat_thresh,
                       self.plat_thresh, self.cosm_filter, self.cosm_box,
                       self.cosm_tail)
        self.assertFalse(mock_find_plateaus.called)

    def test_mask(self):
        """Assert correct elements are masked."""
        bs_image = np.ones_like(self.bs_image)
        bs_image[1, 2:2+len(self.cosm)] = self.cosm
        check_mask = np.zeros_like(self.bs_image, dtype=int)
        check_mask[1, 2:] = 1

        mask = remove_cosmics(bs_image, self.fwc, self.sat_thresh,
                            self.plat_thresh, self.cosm_filter, self.cosm_box,
                            self.cosm_tail)
        np.testing.assert_array_equal(mask, check_mask)


    def test_no_rows_mask(self):
        """Assert mask array is blank if no cosmic rows."""
        mask = remove_cosmics(self.bs_image_below_thresh, self.fwc,
                              self.sat_thresh, self.plat_thresh,
                              self.cosm_filter, self.cosm_box,
                              self.cosm_tail)
        np.testing.assert_array_equal(mask, np.zeros(self.bs_image.shape))

    def test_mask_box(self):
        """Assert correct elements are masked, including the box around
        the cosmic head and the specified cosmic tail.  This uses 'image' for 
        mode, and meta is not None, which makes no difference since mode is 
        'image'."""
        check_mask = np.zeros_like(self.bs_image, dtype=int)
        check_mask[i_streak_rows_t[1]-2:i_streak_rows_t[1]+2+1,
                   50-2:50+2+1] = 1
        # choose cosm_tail >= effective length of simulated tail
        # using cosm_filter=2 and cosm_tail=20:
        check_mask[i_streak_rows_t[1], 50:50+2+20+1] = 1
        check_mask = check_mask.astype(int)
        mask = remove_cosmics(bs_image_box, self.fwc, self.sat_thresh,
                            self.plat_thresh, self.cosm_filter, cosm_box=0,
                            cosm_tail=20, meta=meta)

        self.assertFalse(np.array_equal(mask, check_mask)) # since cosm_box=0

        # now use cosm_box=2 to catch pixels surrounding head, and let meta be
        # None
        mask2 = remove_cosmics(bs_image_box, self.fwc, self.sat_thresh,
                            self.plat_thresh, self.cosm_filter, cosm_box=2,
                            cosm_tail=20)

        self.assertTrue(np.array_equal(mask2, check_mask))

    def test_mask_box_corners(self):
        """Assert correct elements are masked, including the box around
        the cosmic head, when cosmic heads appear in corners."""
        check_mask = np.zeros((10,10), dtype=int)
        image = np.zeros((10,10), dtype=float)
        # lower left corner (head #1)
        image[-1,0:4] = fwc
        # near lower left corner (head #2)
        image[-2,1:4] = fwc
        # upper right corner (head #3)
        image[0,-1] = fwc

        # cosmic head #1
        check_mask[-1,0:] = 1
        # tries for a 2x2 box around head in corner
        check_mask[-3:,0:2] = 1
        # cosmic head #2
        check_mask[-2,1:] = 1
        # tries for a 2x2 box around head
        check_mask[-4:,0:4] = 1
        # cosmic head #3 and attempted box around it
        check_mask[0:3,-3:] = 1

        mask = remove_cosmics(image, self.fwc, self.sat_thresh,
                            self.plat_thresh, self.cosm_filter, cosm_box=2,
                            cosm_tail=self.cosm_tail)

        self.assertTrue(np.array_equal(mask, check_mask))

    def test_cosm_tail_2(self):
        """Assert correct elements are masked when 2 cosmic rays are in
        a single row.  cosm_box=0 for simplicity."""
        check_mask = np.zeros((10,10), dtype=int)
        image = np.zeros((10,10), dtype=float)
        # head #1
        image[-2,0:4] = fwc
        # head #2
        image[-2,6:9] = fwc

        # for cosm_filter=2 and cosm_tail=1:
        # head #1
        check_mask[-2,0:0+2+1+1] = 1
        # cosmic head #2
        check_mask[-2,6:6+2+1+1] = 1

        mask = remove_cosmics(image, self.fwc, self.sat_thresh,
                            self.plat_thresh, cosm_filter=2, cosm_box=0,
                            cosm_tail=1)

        self.assertTrue(np.array_equal(mask, check_mask))

        # for cosm_filter=2 and cosm_tail=3 (overlap due to masked tails):
        # head #1
        check_mask[-2,0:0+2+3+1] = 1
        # cosmic head #2
        check_mask[-2,6:6+2+3+1] = 1

        mask = remove_cosmics(image, self.fwc, self.sat_thresh,
                            self.plat_thresh, cosm_filter=2, cosm_box=0,
                            cosm_tail=3)

        self.assertTrue(np.array_equal(mask, check_mask))

    def test_cosm_tail_bleed_over(self):
        """Assert correct elements are masked when 2 cosmic rays are in
        a single row with bleed over into next row."""
        check_mask = np.zeros((10,10), dtype=int)
        image = np.zeros((10,10), dtype=float)
        # head
        image[-2,6:9] = fwc

        # cosmic head
        check_mask[-2,6:] = 1
        check_mask[-1,0:12] = 1 #bleed over 2+14-(1st row of 4)
        check_mask[-4:,4:9] = 1 # cosm_box=2

        mask = remove_cosmics(image, self.fwc, self.sat_thresh,
                            self.plat_thresh, cosm_filter=2, cosm_box=2,
                            cosm_tail=14, mode='full')

        self.assertTrue(np.array_equal(mask, check_mask))

        # when mode not "full", no bleed over
        check_mask[-1,0:12] = 0 # undo the bleed over
        # cosm_box=2 again since I undid some in previous line
        check_mask[-4:,4:9] = 1
        mask = remove_cosmics(image, self.fwc, self.sat_thresh,
                            self.plat_thresh, cosm_filter=2, cosm_box=2,
                            cosm_tail=14)

        self.assertTrue(np.array_equal(mask, check_mask))


    def test_cosm_tail_bleed_over_meta(self):
        """Assert correct elements are masked when 2 cosmic rays are in
        a single row with bleed over into next row, while taking into account
        the use of meta to prevent detections outside of image area from being 
        flagged."""
        check_mask = np.zeros((meta.frame_rows,meta.frame_cols), dtype=int)
        image = np.zeros((meta.frame_rows,meta.frame_cols), dtype=float)
        # head
        image[im_ending_row-1,im_ending_col-4:im_ending_col-1] = fwc
        # would normally trigger a detection, but not inside image area:
        image[-2,6:9] = fwc

        # cosmic head
        check_mask[im_ending_row-1,im_ending_col-4:] = 1
        # with cosm_tail=100, and (88-2) left in row after cosm_filter, 
        # so bleed 12-2 over next row
        check_mask[im_ending_row,0:10] = 1
        # cosm_box gets cut short one row since the end of the image area is 
        # reached with only 1 extra row of masking below the cosmic head
        check_mask[im_ending_row-3:im_ending_row+1,
                   im_ending_col-6:im_ending_col-1] = 1 # cosm_box=2

        mask = remove_cosmics(image, self.fwc, self.sat_thresh,
                            self.plat_thresh, cosm_filter=2, cosm_box=2,
                            cosm_tail=100, meta=meta, mode='full')

        self.assertTrue(np.array_equal(mask, check_mask))

    def test_meta(self):
        '''Input meta should be an instance of the Metadata class.'''
        image = np.zeros((meta.frame_rows,meta.frame_cols), dtype=float)
        with self.assertRaises(Exception):
            remove_cosmics(image, self.fwc, self.sat_thresh,
                            self.plat_thresh, cosm_filter=2, cosm_box=2,
                            cosm_tail=100, meta='foo', mode='full')
            

class TestFindPlateaus(unittest.TestCase):
    """Unit tests for find_plateaus function.
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
        self.assertEqual(i_beg, beg)


    def test_left_edge_i_begs(self):
        """Verify that function returns correct i_begs result at left edge."""
        beg = 0
        self.streak_row[beg:len(self.cosm)] = self.cosm
        i_beg = find_plateaus(self.streak_row, self.fwc, self.sat_thresh,
                              self.plat_thresh, self.cosm_filter)
        self.assertEqual(i_beg, np.array([beg]))


    def test_right_edge_i_begs(self):
        """Verify that function returns correct i_begs result at right edge."""
        cosm = self.p_basic
        beg = len(self.streak_row)-len(cosm)
        self.streak_row[beg:] = cosm
        i_beg = find_plateaus(self.streak_row, self.fwc, self.sat_thresh,
                              self.plat_thresh, self.cosm_filter)
        self.assertEqual(i_beg, np.array([beg]))


    def test_two_cosm_i_begs(self):
        """Verify that function returns correct i_begs result for two cosm."""
        beg1 = len(self.streak_row) - len(self.cosm)*2
        beg2 = beg1 + len(self.cosm)
        self.streak_row[beg1:beg1 + len(self.cosm)] = self.cosm
        self.streak_row[beg2:beg2 + len(self.cosm)] = self.cosm

        i_begs = find_plateaus(self.streak_row, self.fwc, self.sat_thresh,
                              self.plat_thresh, self.cosm_filter)
        self.assertTrue(np.array_equal(i_begs, np.array([beg1, beg2])))

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
        self.assertEqual(i_beg, np.array([beg]))


    def test_p_dip(self):
        """Verify that function still recognizes a plateau with a dip."""
        cosm = np.append(self.p_dip, self.tail)
        beg = 50
        self.streak_row[beg:beg+len(cosm)] = cosm
        i_beg = find_plateaus(self.streak_row, self.fwc,
                              self.sat_thresh, self.plat_thresh,
                              self.cosm_filter)
        self.assertEqual(i_beg, np.array([beg]))


    def test_p_dip_deep(self):
        """Verify that the function recognizes plateau with a single pixel dip
        below plat_thresh and does not set the end at the dip."""
        cosm = np.append(self.p_dip_deep, self.tail)
        beg = 50
        self.streak_row[beg:beg+len(cosm)] = cosm
        i_beg = find_plateaus(self.streak_row, self.fwc,
                              self.sat_thresh, self.plat_thresh,
                              self.cosm_filter)
        self.assertEqual(i_beg[0], beg)
        # also finds where the dip is when cosm_filter=2, and the dip is
        # 2 away, which is 1 before the next plateau
        self.assertEqual(i_beg[1], beg+3)


    def test_p_uneven(self):
        """Verify that function still recognizes an uneven plateau."""
        cosm = np.append(self.p_uneven, self.tail)
        beg = 50
        self.streak_row[beg:beg+len(cosm)] = cosm
        i_beg = find_plateaus(self.streak_row, self.fwc,
                              self.sat_thresh, self.plat_thresh,
                              self.cosm_filter)
        self.assertEqual(i_beg, np.array([beg]))


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
