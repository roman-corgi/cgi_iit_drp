# -*- coding: utf-8 -*-
"""Unit tests for calibrate_hvon."""

import os
import unittest
from pathlib import Path

import numpy as np

from .gsw_emccd_frame import EMCCDFrame
from .gsw_emccd_frame import EMCCDFrameException
from .read_metadata import Metadata

# Read metadata from path
here = Path(os.path.dirname(os.path.abspath(__file__)))
meta_path = Path(here, 'metadata.yaml')
meta = Metadata(meta_path)

class TestEMCCDFrame(unittest.TestCase):
    """Unit tests for __init__ method."""

    def setUp(self):
        self.frame_dn = 50*np.ones((meta.frame_rows, meta.frame_cols))
        eperdn = 6
        self.em_gain = 10
        self.fwc_pp = 50000/eperdn
        self.fwc_em = 90000/eperdn
        self.bias_offset = 0
        self.frame = EMCCDFrame(self.frame_dn, meta, self.fwc_em, self.fwc_pp,
                                self.em_gain, self.bias_offset)

    def test_inputs(self):
        """Verify input values are assigned correctly."""
        self.assertTrue(np.array_equal(self.frame.frame_dn, self.frame_dn))
        self.assertEqual(self.frame.meta, meta)
        self.assertEqual(self.frame.fwc_em, self.fwc_em)
        self.assertEqual(self.frame.fwc_pp, self.fwc_pp)
        self.assertEqual(self.frame.em_gain, self.em_gain)
        self.assertEqual(self.frame.bias_offset, self.bias_offset)


    def test_attributes(self):
        """Verify other attributes are correct/exist."""
        self.assertTrue(hasattr(self.frame, 'image'))
        self.assertTrue(hasattr(self.frame, 'prescan'))
        self.assertTrue(hasattr(self.frame, 'al_prescan'))
        self.assertTrue(hasattr(self.frame, 'frame_bias'))
        self.assertTrue(hasattr(self.frame, 'bias'))
        self.assertTrue(hasattr(self.frame, 'frame_bias0'))
        self.assertTrue(hasattr(self.frame, 'image_bias0'))

    def test_zeros_frame(self):
        """Verify class does not break for zeros frame."""
        # EMCCDFrame does not expect a zeros frame, but if it got one it
        # shouldn't have a problem with it and just return a zeros frame back
        zeros_frame = np.zeros((meta.frame_rows, meta.frame_cols))
        frame = EMCCDFrame(zeros_frame,
                           meta,
                           self.fwc_em,
                           self.fwc_pp,
                           self.em_gain,
                           self.bias_offset,
        )
        self.assertEqual(frame.frame_dn.tolist(), zeros_frame.tolist())

    def test_bias_hvoff(self):
        """Verify that function finds frame_bias and  bias for hvoff
        distribution."""
        tol = 1

        bval = 100
        bias = bval*np.ones((meta.frame_rows, 1))

        i_r0 = meta.geom['image']['r0c0'][0]
        p_r0 = meta.geom['prescan']['r0c0'][0]
        i_nrow = meta.geom['image']['rows']
        bias_aligned = bias[(i_r0-p_r0):(i_r0-p_r0+i_nrow), :]

        seed = 12345
        rng = np.random.default_rng(seed)
        frame_dn_hvoff = rng.normal(bval, 1,
                                    size=(meta.frame_rows, meta.frame_cols))
        frame = EMCCDFrame(frame_dn_hvoff,
                           meta,
                           self.fwc_em,
                           self.fwc_pp,
                           self.em_gain,
                           self.bias_offset,
        )
        self.assertTrue(np.max(np.abs(frame.bias - bias_aligned)) < tol)
        self.assertTrue(np.max(np.abs(frame.frame_bias - bias)) < tol)


    def test_bias_hvon(self):
        """Verify that function finds frame_bias and bias for hvon
        distribution.  Also tests that only the good columns are used for
        the bias."""
        tol = 6

        bias = 100*np.ones((meta.frame_rows, 1))
        bias_m = bias @ np.ones((1, meta.frame_cols))
        st = meta.geom['prescan']['col_start']
        bias_m[:, 0:st] = 500 # this shouldn't affect result below

        i_r0 = meta.geom['image']['r0c0'][0]
        p_r0 = meta.geom['prescan']['r0c0'][0]
        i_nrow = meta.geom['image']['rows']
        bias_aligned = bias[(i_r0-p_r0):(i_r0-p_r0+i_nrow), :]

        expmean = 10
        seed = 678910
        rng = np.random.default_rng(seed)
        frame_dn_hvon = (rng.normal(0, 1, bias_m.shape)
                         + rng.exponential(expmean, bias_m.shape)
                         - expmean # to keep DC contribution 0
                         + bias_m)
        frame = EMCCDFrame(frame_dn_hvon,
                           meta,
                           self.fwc_em,
                           self.fwc_pp,
                           self.em_gain,
                           self.bias_offset,
        )
        self.assertTrue(np.max(np.abs(frame.bias - bias_aligned)) < tol)
        self.assertTrue(np.max(np.abs(frame.frame_bias - bias)) < tol)

    def test_bias_uniform_value(self):
        """Verify function finds frame_bias and bias for uniform
        distribution."""
        frame_dn_zeros = np.zeros((meta.frame_rows, meta.frame_cols))
        frame = EMCCDFrame(frame_dn_zeros,
                           meta,
                           self.fwc_em,
                           self.fwc_pp,
                           self.em_gain,
                           self.bias_offset,
        )
        self.assertTrue((frame.bias == 0).all())
        self.assertTrue((frame.frame_bias == 0).all())

    def test_bias0(self):
        """Verify property subtracts frame_bias from frame and bias from
        image."""
        tol = 1

        bias = 100
        seed = 55555
        rng = np.random.default_rng(seed)
        frame_dn_hvoff = rng.normal(bias, 1,
                                    size=(meta.frame_rows, meta.frame_cols))
        frame = EMCCDFrame(frame_dn_hvoff,
                           meta,
                           self.fwc_em,
                           self.fwc_pp,
                           self.em_gain,
                           self.bias_offset,
        )

        self.assertTrue(np.max(np.abs(frame.image-bias - frame.image_bias0))
                        < tol)

        p_r0 = frame.meta.geom['prescan']['r0c0'][0]
        self.assertTrue(np.max(np.abs(frame.frame_dn[p_r0:, :]-bias -
                        frame.frame_bias0)) < tol)

    def test_bias_offset(self):
        """Verify bias offset incorporated as expected"""
        # bias_offset = 10 means the bias, as measured in the prescan, is
        # 10 counts higher than the bias in the image region.
        b = 10
        tol = 1e-13

        frame_dn_zeros = np.zeros((meta.frame_rows, meta.frame_cols))
        frame_10 = np.zeros((meta.frame_rows, meta.frame_cols))
        rows, cols, r0c0 = meta._unpack_geom('prescan')
        frame_10[r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols] += b

        f0 = EMCCDFrame(frame_dn_zeros,
                        meta,
                        self.fwc_em,
                        self.fwc_pp,
                        self.em_gain,
                        bias_offset=0,
        )
        f1 = EMCCDFrame(frame_10,
                        meta,
                        self.fwc_em,
                        self.fwc_pp,
                        self.em_gain,
                        bias_offset=b,
        )

        # same frame after bias subtraction
        self.assertTrue(np.max(np.abs(f1.image_bias0 - f0.image_bias0)) < tol)

        pass


    def test_exception_inconsistent_frame_size(self):
        """Verify exception is thrown for inconsistent frame size."""
        small_frame = np.zeros((10, 10))
        with self.assertRaises(EMCCDFrameException):
            EMCCDFrame(small_frame,
                       meta,
                       self.fwc_em,
                       self.fwc_pp,
                       self.em_gain,
                       self.bias_offset,
            )


class TestRemoveCosmics(unittest.TestCase):
    """Unit tests for remove_cosmics method.  Bad inputs are not tested here
    since are inherited inputs that are tested elsewhere."""

    def setUp(self):
        eperdn = 6
        self.fwc_em = 90000/eperdn
        self.fwc_pp = 50000/eperdn
        self.em_gain = 10
        self.bias_offset = 0
        self.frame = EMCCDFrame(np.zeros((meta.frame_rows, meta.frame_cols)),
                                meta,
                                self.fwc_em,
                                self.fwc_pp,
                                self.em_gain,
                                self.bias_offset,
        )
        self.sat_thresh = 0.99
        self.plat_thresh = 0.85
        self.cosm_filter = 2
        pass


    def test_mask(self):
        """Verify method returns correct mask with a CR."""
        # full frame
        frame_bias0 = np.zeros((meta.frame_rows, meta.frame_cols))
        frame_bias0[1:3, 2:4] = self.fwc_em  # Fake hit
        frame_bias0[1:3, 4] = self.fwc_em/2  # Trigger end cosmic thresh
        frame_bias0[1:3, 5:] = 1.  # Fake tail
        self.frame.frame_bias0 = frame_bias0
        frame_mask = np.zeros_like(frame_bias0, dtype=int)
        frame_mask[1:3, 1:] = 1 # Mask starts 1 before cosmic

        # image area
        image_bias0 = np.zeros((meta.geom['image']['rows'],
                                meta.geom['image']['cols']))
        image_bias0[1:3, 2:4] = self.fwc_em  # Fake hit
        image_bias0[1:3, 4] = self.fwc_em/2  # Trigger end cosmic thresh
        image_bias0[1:3, 5:] = 1.  # Fake tail
        self.frame.image_bias0 = image_bias0
        image_mask = np.zeros_like(image_bias0, dtype=int)
        image_mask[1:3, 1:] = 1  # Mask starts 1 before cosmic

        image_outmask, frame_outmask= self.frame.remove_cosmics(
                self.sat_thresh, self.plat_thresh, self.cosm_filter)
        self.assertTrue((frame_mask == frame_outmask).all())
        self.assertTrue((image_mask == image_outmask).all())
        pass


    def test_sub_thresh_em(self):
        """
        Verify method returns correct mask with a CR just below the EM
        gain register threshold

        Use sat_thresh = 1.0 so we test the inequality
        """
        self.frame.em_gain = 1.0 # fwc_em is definitely larger in this case

        # full frame
        frame_bias0 = np.zeros((meta.frame_rows, meta.frame_cols))
        frame_bias0[1:3, 2:4] = self.fwc_pp - 1  # Fake hit *below threshold*
        frame_bias0[1:3, 4] = self.fwc_pp/2  # Trigger end cosmic thresh
        frame_bias0[1:3, 5:] = 1.  # Fake tail
        self.frame.frame_bias0 = frame_bias0

        # image area
        image_bias0 = np.zeros((meta.geom['image']['rows'],
                                meta.geom['image']['cols']))
        image_bias0[1:3, 2:4] = self.fwc_pp - 1  # Fake hit *below threshold*
        image_bias0[1:3, 4] = self.fwc_pp/2  # Trigger end cosmic thresh
        image_bias0[1:3, 5:] = 1.  # Fake tail
        self.frame.image_bias0 = image_bias0

        # Should be nothing bad here
        frame_mask = np.zeros_like(frame_bias0, dtype=int)
        image_mask = np.zeros_like(image_bias0, dtype=int)

        image_outmask, frame_outmask = self.frame.remove_cosmics(
            sat_thresh=1.0, plat_thresh=self.plat_thresh,
            cosm_filter=self.cosm_filter)
        self.assertTrue((frame_mask == frame_outmask).all())
        self.assertTrue((image_mask == image_outmask).all())
        pass


    def test_thresh_em_pix(self):
        """
        Verify method returns correct mask with a CR and a pixel at the EM
        gain register threshold

        Use sat_thresh = 1.0 so we test the inequality
        """
        self.frame.em_gain = 1.0 # fwc_em is definitely larger in this case

        # full frame
        frame_bias0 = np.zeros((meta.frame_rows, meta.frame_cols))
        frame_bias0[1:3, 2:4] = self.fwc_em  # Fake hit
        frame_bias0[1:3, 4] = self.fwc_em/2  # Trigger end cosmic thresh
        frame_bias0[1:3, 5:] = 1.  # Fake tail
        frame_bias0[8, 8] = self.fwc_em # isolated pixel
        self.frame.frame_bias0 = frame_bias0

        # image area
        image_bias0 = np.zeros((meta.geom['image']['rows'],
                                meta.geom['image']['cols']))
        image_bias0[1:3, 2:4] = self.fwc_em  # Fake hit
        image_bias0[1:3, 4] = self.fwc_em/2  # Trigger end cosmic thresh
        image_bias0[1:3, 5:] = 1.  # Fake tail
        image_bias0[8, 8] = self.fwc_em # isolated pixel
        self.frame.image_bias0 = image_bias0

        frame_mask = np.zeros_like(frame_bias0, dtype=int)
        image_mask = np.zeros_like(image_bias0, dtype=int)

        frame_mask[1:3, 1:] = 1  # Mask starts 1 before cosmic
        frame_mask[8, 8] = 1  # Isolated pixels remove only those pixels
        image_mask[1:3, 1:] = 1  # Mask starts 1 before cosmic
        image_mask[8, 8] = 1  # Isolated pixels remove only those pixels

        image_outmask, frame_outmask = self.frame.remove_cosmics(
            sat_thresh=1.0, plat_thresh=self.plat_thresh,
            cosm_filter=self.cosm_filter)
        self.assertTrue((frame_mask == frame_outmask).all())
        self.assertTrue((image_mask == image_outmask).all())
        pass


    def test_sub_thresh_em_pix(self):
        """
        Verify method returns correct mask with a CR below threshold and a
        pixel at the EM gain register threshold

        Use sat_thresh = 1.0 so we test the inequality
        """
        self.frame.em_gain = 1.0 # fwc_em is definitely larger in this case

        # full frame
        frame_bias0 = np.zeros((meta.frame_rows, meta.frame_cols))
        frame_bias0[1:3, 2:4] = self.fwc_pp - 1 # Fake hit *below threshold*
        frame_bias0[1:3, 4] = self.fwc_pp/2  # Trigger end cosmic thresh
        frame_bias0[1:3, 5:] = 1.  # Fake tail
        frame_bias0[8, 8] = self.fwc_em # isolated pixel
        self.frame.frame_bias0 = frame_bias0

        # image area
        image_bias0 = np.zeros((meta.geom['image']['rows'],
                                meta.geom['image']['cols']))
        image_bias0[1:3, 2:4] = self.fwc_pp - 1 # Fake hit *below threshold*
        image_bias0[1:3, 4] = self.fwc_pp/2  # Trigger end cosmic thresh
        image_bias0[1:3, 5:] = 1.  # Fake tail
        image_bias0[8, 8] = self.fwc_em # isolated pixel
        self.frame.image_bias0 = image_bias0

        frame_mask = np.zeros_like(frame_bias0, dtype=int)
        frame_mask[8, 8] = 1  # Isolated pixels remove only those pixels

        image_mask = np.zeros_like(image_bias0, dtype=int)
        image_mask[8, 8] = 1  # Isolated pixels remove only those pixels

        image_outmask, frame_outmask = self.frame.remove_cosmics(
            sat_thresh=1.0, plat_thresh=self.plat_thresh,
            cosm_filter=self.cosm_filter)
        self.assertTrue((frame_mask == frame_outmask).all())
        self.assertTrue((image_mask == image_outmask).all())
        pass


    def test_thresh_pp(self):
        """
        Verify method returns correct mask with a CR at the gain register
        threshold and a pixel below the EM
        gain register threshold but saturating the per-pixel full well

        Use sat_thresh = 1.0 so we test the inequality
        """
        self.frame.em_gain = 1.2 # greater than 1, less than fwc_em/fwc_pp

        # full frame
        frame_bias0 = np.zeros((meta.frame_rows, meta.frame_cols))
        frame_bias0[1:3, 2:4] = self.fwc_em  # Fake hit
        frame_bias0[1:3, 4] = self.fwc_em/2  # Trigger end cosmic thresh
        frame_bias0[1:3, 5:] = 1.  # Fake tail
        frame_bias0[8, 8] = self.fwc_pp*self.em_gain # isolated pixel
        self.frame.frame_bias0 = frame_bias0

        # image area
        image_bias0 = np.zeros((meta.geom['image']['rows'],
                                meta.geom['image']['cols']))
        image_bias0[1:3, 2:4] = self.fwc_em  # Fake hit
        image_bias0[1:3, 4] = self.fwc_em/2  # Trigger end cosmic thresh
        image_bias0[1:3, 5:] = 1.  # Fake tail
        image_bias0[8, 8] = self.fwc_pp*self.em_gain # isolated pixel
        self.frame.image_bias0 = image_bias0

        frame_mask = np.zeros_like(frame_bias0, dtype=int)
        frame_mask[1:3, 1:] = 1  # Mask starts 1 before cosmic
        frame_mask[8, 8] = 1  # Isolated pixels remove only those pixels

        image_mask = np.zeros_like(image_bias0, dtype=int)
        image_mask[1:3, 1:] = 1  # Mask starts 1 before cosmic
        image_mask[8, 8] = 1  # Isolated pixels remove only those pixels

        image_outmask, frame_outmask = self.frame.remove_cosmics(
            sat_thresh=1.0, plat_thresh=self.plat_thresh,
            cosm_filter=self.cosm_filter)
        self.assertTrue((frame_mask == frame_outmask).all())
        self.assertTrue((image_mask == image_outmask).all())
        pass


if __name__ == '__main__':
    unittest.main()
