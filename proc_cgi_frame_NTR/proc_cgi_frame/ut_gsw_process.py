# -*- coding: utf-8 -*-
# pylint: disable=unsubscriptable-object # pylint handles .shape poorly
"""Unit tests for Process."""
# NOTE TO FSW: THIS MODULE IS ONLY USED BY THE GROUND PIPELINE, DO NOT PORT

import copy
import os
import unittest
from pathlib import Path

import numpy as np

from .gsw_process import (Process, median_combine, mean_combine)
from .read_metadata import Metadata
from .gsw_nonlin import _parse_file
from . import ut_check

# Metadata path
here = Path(os.path.dirname(os.path.abspath(__file__)))
meta_path = Path(here, 'metadata.yaml')

# Read metadata
meta = Metadata(meta_path)
image_rows, image_cols, image_ul = meta._unpack_geom('image')

# Very rough approximations of data distributions
frame_dn_zeros = np.zeros((meta.frame_rows, meta.frame_cols))
frame_dn_hvoff = np.random.normal(0, 1, size=frame_dn_zeros.shape)
frame_dn_hvon = (np.random.exponential(100, size=frame_dn_zeros.shape)
                 + frame_dn_hvoff)

nframes = 5
frames = np.random.normal(0, 1, size=(nframes, frame_dn_zeros.shape[0],
                                      frame_dn_zeros.shape[1]))
frames += np.random.exponential(100, size=frames.shape)
darks = (np.random.normal(0, 1, size=frames.shape)
         + np.random.exponential(100, size=frames.shape))
flat = np.ones((image_rows, image_cols))
bad_pix = np.zeros(flat.shape)

dark = np.zeros_like(flat)
eperdn = 1.
fwc_em_e = 90000
fwc_pp_e = 50000
bias_offset = 0
em_gain = 1
exptime = 1

# Create nonlin array to be written to csv
# # Row headers are counts
check_count_ax = np.array([1, 1000, 2000, 3000])
# Column headers are em gains
check_gain_ax = np.array([1, 10, 100, 1000])
# Relative gain data, valid curves
curve1 = np.array([0.900, 0.910, 0.950, 1.000])  # Last value is exactly 1
curve2 = np.array([0.950, 0.950, 1.000, 1.001])  # Mid value is exactly 1
curve3 = np.array([0.989, 0.990, 1.010, 1.011])  # Values straddle 1
curve4 = np.array([1.000, 1.010, 1.050, 1.060])  # First value is exactly 1
# Fill array with relative gain curves
check_relgains = np.zeros((len(check_count_ax), len(check_gain_ax)))
check_relgains[:, 0] = curve1
check_relgains[:, 1] = curve2
check_relgains[:, 2] = curve3
check_relgains[:, 3] = curve4

# Build nonlin array
nonlin_array = np.ones((check_relgains.shape[0]+1, check_relgains.shape[1]+1))
nonlin_array[0, 0] = np.nan  # Top left corner is set to nan
nonlin_array[1:, 0] = check_count_ax
nonlin_array[0, 1:] = check_gain_ax
# Save one array with all relative gains set to 1 (perfect linearity)
nonlin_array_ones = nonlin_array.copy()
nonlin_array[1:, 1:] = check_relgains

sat_thresh = 0.99
plat_thresh = 0.85
cosm_filter = 2

localpath = os.path.dirname(os.path.abspath(__file__))

class TestProcess(unittest.TestCase):
    """Unit tests for __init__ method."""

    def setUp(self):
        self.nonlin_path_ones = os.path.join(localpath, 'testdata',
                                             'ut_nonlin_array_ones.txt')

    def test_inputs(self):
        """Verify input values are assigned correctly."""
        proc = Process(bad_pix, eperdn, fwc_em_e, fwc_pp_e, bias_offset,
                       em_gain, exptime, self.nonlin_path_ones, meta_path,
                       dark, flat, sat_thresh, plat_thresh, cosm_filter)

        self.assertTrue(np.array_equal(proc.bad_pix, bad_pix))
        self.assertTrue(np.array_equal(proc.eperdn, eperdn))
        self.assertTrue(np.array_equal(proc.fwc_em_dn, fwc_em_e/eperdn))
        self.assertTrue(np.array_equal(proc.fwc_pp_dn, fwc_pp_e/eperdn))
        self.assertEqual(proc.meta_path, meta_path)
        self.assertEqual(proc.nonlin_path, self.nonlin_path_ones)
        self.assertTrue(np.array_equal(proc.bias_offset, bias_offset))
        self.assertTrue(proc.em_gain == em_gain)
        self.assertTrue(proc.exptime == exptime)
        self.assertTrue(np.array_equal(proc.dark, dark))
        self.assertTrue(np.array_equal(proc.flat, flat))
        self.assertTrue(np.array_equal(proc.sat_thresh, sat_thresh))
        self.assertTrue(np.array_equal(proc.plat_thresh, plat_thresh))
        self.assertTrue(np.array_equal(proc.cosm_filter, cosm_filter))

    def test_bad_pix(self):
        """bad_pix input bad."""
        for perr in ut_check.twoDlist:
            with self.assertRaises(TypeError):
                Process(perr, eperdn, fwc_em_e, fwc_pp_e, bias_offset,
                        em_gain, exptime, self.nonlin_path_ones,
                        meta_path, dark, flat)

    def test_eperdn(self):
        """eperdn input bad."""
        for perr in ut_check.rpslist:
            with self.assertRaises(TypeError):
                Process(bad_pix, perr, fwc_em_e, fwc_pp_e, bias_offset,
                        em_gain, exptime, self.nonlin_path_ones,
                        meta_path, dark, flat)

    def test_fwc_em_e(self):
        """fwc_em_e input bad."""
        for perr in ut_check.psilist:
            with self.assertRaises(TypeError):
                Process(bad_pix, eperdn, perr, fwc_pp_e, bias_offset,
                        em_gain, exptime, self.nonlin_path_ones,
                        meta_path, dark, flat)

    def test_fwc_pp_e(self):
        """fwc_pp_e input bad."""
        for perr in ut_check.psilist:
            with self.assertRaises(TypeError):
                Process(bad_pix, eperdn, fwc_em_e, perr, bias_offset,
                        em_gain, exptime, self.nonlin_path_ones,
                        meta_path, dark, flat)

    def test_sat_thresh(self):
        """sat_thresh input bad."""
        for perr in ut_check.rpslist:
            with self.assertRaises(TypeError):
                Process(bad_pix, eperdn, fwc_em_e, fwc_pp_e, bias_offset,
                        em_gain, exptime, self.nonlin_path_ones,
                        meta_path, dark, flat,
                        sat_thresh=perr)

    def test_plat_thresh(self):
        """plat_thresh input bad."""
        for perr in ut_check.rpslist:
            with self.assertRaises(TypeError):
                Process(bad_pix, eperdn, fwc_em_e, fwc_pp_e, bias_offset,
                        em_gain, exptime, self.nonlin_path_ones,
                        meta_path, dark, flat,
                        plat_thresh=perr)

    def test_cosm_filter(self):
        """cosm_filter input bad."""
        for perr in ut_check.psilist:
            with self.assertRaises(TypeError):
                Process(bad_pix, eperdn, fwc_em_e, fwc_pp_e, bias_offset,
                        em_gain, exptime, self.nonlin_path_ones,
                        meta_path, dark, flat,
                        cosm_filter=perr)

    def test_em_gain(self):
        """em_gain input bad."""
        for perr in ut_check.rpslist:
            with self.assertRaises(TypeError):
                Process(bad_pix, eperdn, fwc_em_e, fwc_pp_e, bias_offset,
                        perr, exptime, self.nonlin_path_ones,
                        meta_path, dark, flat,
                        cosm_filter)

    def test_invalid_em_gain(self):
        """Invalid inputs caught"""
        perrlist = [
                    0.99999, 0.01, 0.5 # real positive scalar that is < 1
                    ]
        for perr in perrlist:
            with self.assertRaises(ValueError):
                Process(bad_pix, eperdn, fwc_em_e, fwc_pp_e,
                        bias_offset,
                        perr, exptime, self.nonlin_path_ones,
                        meta_path, dark, flat)
        pass

    def test_exptime(self):
        """exptime input bad."""
        for perr in ut_check.rpslist:
            with self.assertRaises(TypeError):
                Process(bad_pix, eperdn, fwc_em_e, fwc_pp_e, bias_offset,
                        em_gain, perr, self.nonlin_path_ones,
                        meta_path, dark, flat,
                        cosm_filter)

    def test_dark(self):
        """dark input bad."""
        for perr in ut_check.twoDlist:
            with self.assertRaises(TypeError):
                Process(bad_pix, eperdn, fwc_em_e, fwc_pp_e, bias_offset,
                        em_gain, exptime, self.nonlin_path_ones,
                        meta_path, perr, flat)

    def test_flat(self):
        """flat input bad."""
        for perr in ut_check.twoDlist:
            with self.assertRaises(TypeError):
                Process(bad_pix, eperdn, fwc_em_e, fwc_pp_e, bias_offset,
                        em_gain, exptime, self.nonlin_path_ones,
                        meta_path, dark, perr)

    def test_shape_dark(self):
        """The dimensions of dark and bad_pix should agree."""
        dark_x = np.shape(bad_pix)[0] - 1
        dark_y = np.shape(bad_pix)[1]
        bad_dark = np.zeros((dark_x, dark_y))
        with self.assertRaises(ValueError):
            Process(bad_pix, eperdn, fwc_em_e, fwc_pp_e, bias_offset,
                        em_gain, exptime, self.nonlin_path_ones,
                        meta_path, bad_dark, flat)

    def test_shape_flat(self):
        """The dimensions of flat and bad_pix should agree."""
        flat_x = np.shape(bad_pix)[0]
        flat_y = np.shape(bad_pix)[1] + 1
        bad_flat = np.zeros((flat_x, flat_y))
        with self.assertRaises(ValueError):
            Process(bad_pix, eperdn, fwc_em_e, fwc_pp_e, bias_offset,
                        em_gain, exptime, self.nonlin_path_ones,
                        meta_path, dark, bad_flat)

    def test_bias_offset(self):
        """bias_offsetinput bad."""
        for perr in ut_check.rnslist:
            with self.assertRaises(TypeError):
                Process(bad_pix, eperdn, fwc_em_e, fwc_pp_e, perr,
                        em_gain, exptime, self.nonlin_path_ones,
                        meta_path, dark, flat)

    def test_meta_path(self):
        """meta_path should be actual file path."""
        with self.assertRaises(FileNotFoundError):
            Process(bad_pix, eperdn, fwc_em_e, fwc_pp_e, bias_offset,
                        em_gain, exptime, self.nonlin_path_ones,
                        'foo', dark, flat)

    def test_defaults(self):
        """Verify default input values are correct."""
        proc = Process(bad_pix, eperdn, fwc_em_e, fwc_pp_e, bias_offset,
                       em_gain, exptime, self.nonlin_path_ones)

        self.assertTrue(proc.meta_path == Path(here, 'metadata.yaml'))
        self.assertTrue(np.array_equal(proc.dark, np.zeros_like(dark)))
        self.assertTrue(np.array_equal(proc.flat, np.ones_like(flat)))

    def test_attributes(self):
        """Verify attributes are correct/exist."""
        proc = Process(bad_pix, eperdn, fwc_em_e, fwc_pp_e, bias_offset,
                       em_gain, exptime, self.nonlin_path_ones,
                       meta_path, dark, flat)

        self.assertIsInstance(proc.meta, Metadata)


class TestL1ToL2a(unittest.TestCase):
    """Unit tests for L1_to_L2a method"""

    def setUp(self):
        self.mask_fixed_tails = False

        # full frame________
        self.full_active = copy.copy(frame_dn_zeros)
        self.full_active[0:2, 2:4] = fwc_em_e  # Fake hit
        self.full_active[0:2, 4] = fwc_em_e / 2  # Trigger end cosmic thresh
        self.full_active[0:2, 5:12] = 1.  # Fake tail
        # start 1 pixel before hit as remove_cosmics oversizes the plateau by 1
        self.full_row_mask = np.zeros_like(frame_dn_zeros, dtype='int')
        self.full_row_mask[0:2, 1:] = 1

        self.full_frame = copy.copy(frame_dn_zeros)
        self.full_frame = self.full_active

        self.full_dark = np.zeros_like(frame_dn_zeros)
        self.full_flat = np.ones_like(frame_dn_zeros)
        self.full_bad_pix = np.zeros_like(frame_dn_zeros, dtype='bool')

        # image area_________
        self.active = np.zeros((image_rows, image_cols))
        self.active[0:2, 2:4] = fwc_em_e  # Fake hit
        self.active[0:2, 4] = fwc_em_e / 2  # Trigger end cosmic thresh
        self.active[0:2, 5:12] = 1.  # Fake tail
        # start 1 pixel before hit as remove_cosmics oversizes the plateau by 1
        self.row_mask = np.zeros((image_rows, image_cols), dtype='int')
        self.row_mask[0:2, 1:] = 1

        self.frame = copy.copy(frame_dn_zeros)
        self.frame[image_ul[0]:image_ul[0]+image_rows,
                   image_ul[1]:image_ul[1]+image_cols] = self.active

        self.dark = np.zeros((image_rows, image_cols))
        self.flat = np.ones((image_rows, image_cols))
        self.bad_pix = np.zeros((image_rows, image_cols), dtype='bool')

        self.nonlin_path_ones = os.path.join(localpath, 'testdata',
                                             'ut_nonlin_array_ones.txt')


    def test_exception_not_array(self):
        """Verify method throws exception if input is not a 2d array."""
        proc = Process(bad_pix, eperdn, fwc_em_e, fwc_pp_e, bias_offset,
                       em_gain, exptime, self.nonlin_path_ones,
                       meta_path, dark, flat)

        with self.assertRaises(TypeError):
            proc.L1_to_L2a(np.zeros([10]), proc.em_gain)  # 1d array
        with self.assertRaises(TypeError):
            proc.L1_to_L2a(np.zeros([10, 10, 10]), proc.em_gain)  # 3d array


    def test_mask(self):
        """
        Verify everything after CR hit is masked, and one pixel before
        """
        proc = Process(self.bad_pix, eperdn, fwc_em_e, fwc_pp_e, bias_offset,
                       em_gain, exptime, self.nonlin_path_ones, meta_path,
                       self.dark, self.flat)
        _, bad_mask, image_r, _, _, _, _ = proc.L1_to_L2a(self.frame)

        self.assertTrue((bad_mask == self.row_mask).all())
        # ensure the image with CR removed is as expected:
        masked_rows, masked_cols = np.where(self.row_mask)
        self.assertTrue((image_r[masked_rows, masked_cols] == 0).all())

    def test_full_mask(self):
        """
        Verify everything after CR hit is masked, and one pixel before (for
        full frame)
        """
        proc = Process(self.full_bad_pix, eperdn, fwc_em_e, fwc_pp_e,
                       bias_offset, em_gain, exptime,
                       self.nonlin_path_ones, meta_path,
                       self.full_dark, self.full_flat)
        _, _, _, _, _, bad_mask, _ = proc.L1_to_L2a(self.full_frame)

        self.assertTrue((bad_mask == self.full_row_mask).all())


    def test_nonlin_fixed_gain(self):
        """Verify function multiplies by relative gain."""
        nonlin_path_notones = os.path.join(localpath, 'testdata',
                                           'ut_nonlin_array_notones.txt')

        tol = 1e-13

        # using counts=1 and gain=1 uses the element in the [1, 1] slot for
        # the above relative-gain array
        frame = np.ones_like(self.frame)

        proc = Process(self.bad_pix, eperdn, fwc_em_e, fwc_pp_e, bias_offset,
                       em_gain, exptime, self.nonlin_path_ones, meta_path,
                       self.dark, self.flat)
        image, _, _, _, _, _, _ = proc.L1_to_L2a(frame)
        proc0 = Process(self.bad_pix, eperdn, fwc_em_e, fwc_pp_e, bias_offset,
                        em_gain, exptime,
                        nonlin_path_notones, meta_path,
                        self.dark, self.flat)
        image0, _, _, _, _, _, _ = proc0.L1_to_L2a(frame)

        self.assertTrue(np.max(np.abs(image - image0/1.5)) < tol)
        pass


    def test_full_nonlin_fixed_gain(self):
        """Verify function multiplies by relative gain."""
        nonlin_path_notones = os.path.join(localpath, 'testdata',
                                           'ut_nonlin_array_notones.txt')

        tol = 1e-13

        # using counts=1 and gain=1 uses the element in the [1, 1] slot for
        # the above relative-gain array
        frame = np.ones_like(self.full_frame)

        proc = Process(self.full_bad_pix, eperdn, fwc_em_e, fwc_pp_e,
                    bias_offset, em_gain, exptime, self.nonlin_path_ones,
                    meta_path, self.full_dark, self.full_flat)
        _, _, _, _, image, _, _ = proc.L1_to_L2a(frame)
        proc0 = Process(self.full_bad_pix, eperdn, fwc_em_e, fwc_pp_e,
                        bias_offset, em_gain, exptime,
                        nonlin_path_notones, meta_path,
                        self.full_dark, self.full_flat)
        _, _, _, _, image0, _, _ = proc0.L1_to_L2a(frame)

        self.assertTrue(np.max(np.abs(image - image0/1.5)) < tol)
        pass

    def test_nonlin_full_array(self):
        """
        Verify function multiplies by relative gain and that the various
        possible configurations of the relative gain array are cycled.
        """
        nonlin_path = os.path.join(localpath, 'testdata',
                                   'ut_nonlin_array.txt')

        tol = 1e-13

        # using counts=1 and gain=1 uses the element in the [1, 1] slot for
        # the above relative-gain array
        frame = np.ones_like(self.frame)

        proc = Process(self.bad_pix, eperdn, fwc_em_e, fwc_pp_e, bias_offset,
                       em_gain, exptime, nonlin_path, meta_path,
                       self.dark, self.flat)
        image, _, _, _, _, _, _ = proc.L1_to_L2a(frame)

        for j, count in enumerate(check_count_ax):
            for k, gain in enumerate(check_gain_ax):
                proc_t = Process(self.bad_pix, eperdn, fwc_em_e, fwc_pp_e,
                       bias_offset, gain, exptime, nonlin_path, meta_path,
                       self.dark, self.flat)
                image0, _, _, _, _, _, _ = proc_t.L1_to_L2a(frame*count)

                rat = nonlin_array[1, 1]/nonlin_array[j+1, k+1]
                self.assertTrue(np.max(np.abs(image - image0*rat)) < tol)
                pass
            pass

    def test_full_nonlin_full_array(self):
        """
        Verify function multiplies by relative gain and that the various
        possible configurations of the relative gain array are cycled.
        """
        nonlin_path = os.path.join(localpath, 'testdata',
                                   'ut_nonlin_array.txt')

        tol = 1e-13

        # using counts=1 and gain=1 uses the element in the [1, 1] slot for
        # the above relative-gain array
        frame = np.ones_like(self.frame)

        proc = Process(self.full_bad_pix, eperdn, fwc_em_e, fwc_pp_e,
                    bias_offset, em_gain, exptime, nonlin_path,
                    meta_path, self.full_dark,
                    self.full_flat)
        _, _, _, _, image, _, _ = proc.L1_to_L2a(frame)

        for j, count in enumerate(check_count_ax):
            for k, gain in enumerate(check_gain_ax):
                proc_t = Process(self.full_bad_pix, eperdn, fwc_em_e, fwc_pp_e,
                       bias_offset, gain, exptime, nonlin_path, meta_path,
                       self.full_dark, self.full_flat)
                _, _, _, _, image0, _, _ = proc_t.L1_to_L2a(frame*count)

                rat = nonlin_array[1, 1]/nonlin_array[j+1, k+1]
                self.assertTrue(np.max(np.abs(image - image0*rat)) < tol)
                pass
            pass

        pass


    def test_dimensions(self):
        """Verify method returns correct array dimensions."""
        proc = Process(bad_pix, eperdn, fwc_em_e, fwc_pp_e, bias_offset,
                       em_gain, exptime, self.nonlin_path_ones,
                       meta_path, dark, flat)
        i0, b0, _, bias, _, _, _ = proc.L1_to_L2a(np.squeeze(frames[0, :, :]))

        self.assertEqual(i0.shape, (image_rows, image_cols))
        self.assertEqual(b0.shape, (image_rows, image_cols))
        self.assertEqual(bias.shape, (image_rows, 1))


    def test_full_dimensions(self):
        """Verify method returns correct array dimensions."""
        proc = Process(self.full_bad_pix, eperdn, fwc_em_e, fwc_pp_e,
                    bias_offset, em_gain, exptime, self.nonlin_path_ones,
                    meta_path, self.full_dark, self.full_flat)
        _, _, _, _, i0, b0, bias = proc.L1_to_L2a(np.squeeze(frames[0, :, :]))

        self.assertEqual(i0.shape, (meta.frame_rows, meta.frame_cols))
        self.assertEqual(b0.shape, (meta.frame_rows, meta.frame_cols))
        self.assertEqual(bias.shape, (meta.geom['prescan']['rows'], 1))

    def test_bias(self):
        """bias returned as expected"""
        proc = Process(bad_pix, eperdn, fwc_em_e, fwc_pp_e, bias_offset,
                       em_gain, exptime, self.nonlin_path_ones,
                       meta_path, dark, flat)

        bval = 100
        _, _, _, bias, _, _, _ = proc.L1_to_L2a(bval*np.ones_like(self.frame))
        self.assertTrue((bias == bval).all())



class TestL2aToL2b(unittest.TestCase):
    """Unit tests for L2a_to_L2b method"""

    def setUp(self):
        self.mask_fixed_tails = False

        self.active = np.zeros((image_rows, image_cols))
        self.active[0:2, 2:4] = fwc_em_e  # Fake hit
        self.active[0:2, 4] = fwc_em_e / 2  # Trigger end cosmic thresh
        self.active[0:2, 5:12] = 1.  # Fake tail
        # start 1 pixel before hit as remove_cosmics oversizes the plateau by 1
        self.row_mask = np.zeros((image_rows, image_cols), dtype='int')
        self.row_mask[0:2, 1:] = 1

        self.frame = copy.copy(frame_dn_zeros)
        self.frame[image_ul[0]:image_ul[0]+image_rows,
                   image_ul[1]:image_ul[1]+image_cols] = self.active

        self.dark = np.zeros((image_rows, image_cols))
        self.flat = np.ones((image_rows, image_cols))
        self.bad_pix = np.zeros((image_rows, image_cols), dtype='bool')

        self.nonlin_path_ones = os.path.join(localpath, 'testdata',
                                             'ut_nonlin_array_ones.txt')


    def test_exception_not_array(self):
        """Verify method throws exception if input is not a 2d array."""
        proc = Process(bad_pix, eperdn, fwc_em_e, fwc_pp_e, bias_offset,
                       em_gain, exptime, self.nonlin_path_ones,
                       meta_path, dark, flat)
        i0, b0, _, _, _, _, _ = proc.L1_to_L2a(self.frame)

        with self.assertRaises(TypeError):
            proc.L2a_to_L2b(np.zeros([10]), b0)  # 1d array
        with self.assertRaises(TypeError):
            proc.L2a_to_L2b(np.zeros([10, 10, 10]), b0)  # 3d array
        with self.assertRaises(TypeError):
            proc.L2a_to_L2b(i0, np.zeros([10]))  # 1d array
        with self.assertRaises(TypeError):
            proc.L2a_to_L2b(i0, np.zeros([10, 10, 10]))  # 3d array


    def test_dark(self):
        """Verify dark frame is being subtracted."""
        tol = 1e-13

        # only apply dark current to actual pixels, otherwise it's bias
        dark_value = 0.10
        extra_dark = np.zeros_like(self.frame)
        extra_dark[image_ul[0]:image_ul[0]+image_rows,
                   image_ul[1]:image_ul[1]+image_cols] += dark_value

        proc = Process(self.bad_pix, eperdn, fwc_em_e, fwc_pp_e, bias_offset,
                       em_gain, exptime, self.nonlin_path_ones, meta_path,
                       self.dark, self.flat)
        i0, b0, _, _, _, _, _ = proc.L1_to_L2a(self.frame)
        image, _, _ = proc.L2a_to_L2b(i0, b0)
        proc0 = Process(self.bad_pix, eperdn, fwc_em_e, fwc_pp_e, bias_offset,
                        em_gain, exptime, self.nonlin_path_ones, meta_path,
                        self.dark + dark_value, self.flat)
        i0, b0, _, _, _, _, _ = proc0.L1_to_L2a(self.frame + extra_dark)
        image0, _, _ = proc0.L2a_to_L2b(i0, b0)

        self.assertTrue(np.max(np.abs(image - image0)) < tol)


    def test_flat(self):
        """Verify flat is being divided."""
        tol = 1e-13

        flat_val = 10
        check_flat = self.flat * flat_val
        proc = Process(self.bad_pix, eperdn, fwc_em_e, fwc_pp_e, bias_offset,
                       em_gain, exptime, self.nonlin_path_ones, meta_path,
                       self.dark, self.flat)
        i0, b0, _, _, _, _, _ = proc.L1_to_L2a(self.frame)
        image, _, _ = proc.L2a_to_L2b(i0, b0)
        proc0 = Process(self.bad_pix, eperdn, fwc_em_e, fwc_pp_e, bias_offset,
                        em_gain, exptime, self.nonlin_path_ones, meta_path,
                        self.dark, check_flat)
        i0, b0, _, _, _, _, _ = proc0.L1_to_L2a(self.frame)
        image0, _, _ = proc0.L2a_to_L2b(i0, b0)

        self.assertTrue(np.max(np.abs(image - image0*flat_val)) < tol)


    def test_flat_zeros(self):
        """Verify flat zero values are handled."""
        tol = 1e-13

        check_flat = copy.copy(self.flat)
        check_flat[0, 0] = 0

        proc = Process(self.bad_pix, eperdn, fwc_em_e, fwc_pp_e, bias_offset,
                       em_gain, exptime, self.nonlin_path_ones, meta_path,
                       self.dark, self.flat)
        i0, b0, _, _, _, _, _ = proc.L1_to_L2a(self.frame)
        image, _, _ = proc.L2a_to_L2b(i0, b0)

        check_image = copy.copy(image)
        check_image[0, 0] = 0

        proc0 = Process(self.bad_pix, eperdn, fwc_em_e, fwc_pp_e, bias_offset,
                        em_gain, exptime, self.nonlin_path_ones, meta_path,
                        self.dark, check_flat)
        i0, b0, _, _, _, _, _ = proc0.L1_to_L2a(self.frame)
        image0, _, _ = proc0.L2a_to_L2b(i0, b0)

        self.assertTrue(np.max(np.abs(image0 - check_image)) < tol)


    def test_bad_mask(self):
        """Verify bad pix is being applied."""

        proc = Process(self.bad_pix, eperdn, fwc_em_e, fwc_pp_e, bias_offset,
                       em_gain, exptime, self.nonlin_path_ones, meta_path,
                       self.dark, self.flat)
        i0, b0, _, _, _, _, _ = proc.L1_to_L2a(self.frame)
        _, bad_mask, _ = proc.L2a_to_L2b(i0, b0)

        check_bad_pix = copy.copy(self.bad_pix)
        check_bad_pix[-1, -1] = 1

        proc0 = Process(check_bad_pix, eperdn, fwc_em_e, fwc_pp_e, bias_offset,
                        em_gain, exptime, self.nonlin_path_ones, meta_path,
                        self.dark, self.flat)
        i0, b0, _, _, _, _, _ = proc0.L1_to_L2a(self.frame)
        _, bad_mask0, _ = proc0.L2a_to_L2b(i0, b0)

        self.assertTrue((bad_mask0 == np.logical_or(bad_mask,
                                                    check_bad_pix)).all())


    def test_bad_mask_no_add(self):
        """Verify bad pix is acting as mask and not adding."""
        check_bad_pix = copy.copy(self.row_mask)  # Same locations as cosm
        proc = Process(check_bad_pix, eperdn, fwc_em_e, fwc_pp_e, bias_offset,
                       em_gain, exptime, self.nonlin_path_ones, meta_path,
                       self.dark, self.flat)
        i0, b0, _, _, _, _, _ = proc.L1_to_L2a(self.frame)
        _, bad_mask, image_r = proc.L2a_to_L2b(i0, b0)

        self.assertTrue((bad_mask == self.row_mask).all())
        # ensure the image with CR removed is as expected:
        masked_rows, masked_cols = np.where(self.row_mask)
        self.assertTrue((image_r[masked_rows, masked_cols] == 0).all())


    def test_em_gain_in(self):
        """Verify image is being divided by em_gain when it is provided."""

        tol = 1e-13
        proc = Process(self.bad_pix, eperdn, fwc_em_e, fwc_pp_e, bias_offset,
                       em_gain, exptime, self.nonlin_path_ones, meta_path,
                       self.dark, self.flat)
        i0, b0, _, _, _, _, _ = proc.L1_to_L2a(self.frame)
        image1, _, _ = proc.L2a_to_L2b(i0, b0)
        proc1 = Process(self.bad_pix, eperdn, fwc_em_e, fwc_pp_e, bias_offset,
                       10, exptime, self.nonlin_path_ones, meta_path,
                       self.dark, self.flat)
        i0, b0, _, _, _, _, _ = proc1.L1_to_L2a(self.frame)
        image10, _, _ = proc1.L2a_to_L2b(i0, b0)

        self.assertTrue(np.max(np.abs(image1 - image10*10)) < tol)


    def test_eperdn(self):
        """Verify function multiplies by eperdn."""
        eperdn0 = 0.5*eperdn
        tol = 1e-13

        proc = Process(self.bad_pix, eperdn, fwc_em_e, fwc_pp_e, bias_offset,
                       em_gain, exptime, self.nonlin_path_ones, meta_path,
                       self.dark, self.flat)
        i0, b0, _, _, _, _, _ = proc.L1_to_L2a(self.frame)
        image, _, _ = proc.L2a_to_L2b(i0, b0)
        proc0 = Process(self.bad_pix, eperdn0, fwc_em_e, fwc_pp_e, bias_offset,
                        em_gain, exptime, self.nonlin_path_ones, meta_path,
                        self.dark, self.flat)
        i0, b0, _, _, _, _, _ = proc0.L1_to_L2a(self.frame)
        image0, _, _ = proc0.L2a_to_L2b(i0, b0)

        self.assertTrue(np.max(np.abs(image - image0/0.5)) < tol)


    def test_bpmap_not_0_or_1(self):
        """Catch case where bpmap is not 0 or 1"""
        proc = Process(bad_pix, eperdn, fwc_em_e, fwc_pp_e, bias_offset,
                       em_gain, exptime, self.nonlin_path_ones,
                       meta_path, dark, flat)
        i0, b0, _, _, _, _, _ = proc.L1_to_L2a(self.frame)

        bbad = np.zeros_like(b0, dtype=int)
        for val in [-1, 2]:
            bbad[0, 0] = val
            with self.assertRaises(TypeError):
                proc.L2a_to_L2b(i0, bbad)
            pass
        pass




class TestMedianCombine(unittest.TestCase):
    """Unit tests for median_combine method."""

    def setUp(self):
        ones = np.ones((10, 10))
        zeros = np.zeros((10, 10), dtype=int)
        self.vals = [1, 2, 4]
        self.check_ims = []
        self.check_masks = []
        for val in self.vals:
            self.check_ims.append(ones*val)
            self.check_masks.append(zeros.copy())
            pass

        self.all_masks_i = (0, 0)
        self.single_mask_i = (0, 1)
        # Mask one pixel across all frames
        for m in self.check_masks:
            m[self.all_masks_i[0], self.all_masks_i[1]] = 1
            pass

        # Mask another pixel in only one frame
        self.check_masks[0][self.single_mask_i[0], self.single_mask_i[1]] = 1
        # median value across frames for that single-masked pixel
        self.med_for_single = np.median([self.vals[1], self.vals[2]])

    def test_median_im(self):
        """Verify method calculates median image."""

        combined_im, _ = median_combine(self.check_ims, self.check_masks)

        check_med_frame = np.median(self.check_ims, axis=0)
        # insert 0 for the pixel where all frames masked
        check_med_frame[self.all_masks_i] = 0
        # insert the expected median value for the single-masked pixel
        check_med_frame[self.single_mask_i] = self.med_for_single

        # now check that combined_im agrees with the check
        self.assertTrue((combined_im == check_med_frame).all())


    def test_median_mask(self):
        """Verify method calculates correct median mask."""

        _, combined_mask = median_combine(self.check_ims, self.check_masks)
        check_combined_mask = np.zeros_like(combined_mask)
        # Only this pixel should be combined
        check_combined_mask[self.all_masks_i] = 1

        self.assertTrue((combined_mask == check_combined_mask).all())


    def test_invalid_image_list(self):
        """Invalid inputs caught"""

        bpmap_list = [np.zeros((3, 3), dtype=int), np.eye(3, dtype=int)]

        # for image_list
        perrlist = [
            'txt', None, 1j, 0, (5,), # not list
            (np.ones((3, 3)), np.ones((3, 3))), # not list
            [np.eye(3), np.eye(3), np.eye(3)], # length mismatch
            [np.eye(3)], # length mismatch
            [np.eye(4), np.eye(3)], # array size mismatch
            [np.eye(4), np.eye(4)], # array size mismatch
            [np.ones((1, 3, 3)), np.ones((1, 3, 3))], # not 2D
            ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                median_combine(perr, bpmap_list)
            pass

        # special empty case
        with self.assertRaises(TypeError):
            median_combine([], [])

        pass


    def test_invalid_bpmap_list(self):
        """Invalid inputs caught"""

        image_list = [np.zeros((3, 3)), np.eye(3)]

        # for image_list
        perrlist = [
            'txt', None, 1j, 0, (5,), # not list
            (np.ones((3, 3), dtype=int), np.ones((3, 3), dtype=int)), #not list
            [np.eye(3, dtype=int), np.eye(3, dtype=int),
             np.eye(3, dtype=int)], # length mismatch
            [np.eye(3, dtype=int)], # length mismatch
            [np.eye(4, dtype=int), np.eye(3, dtype=int)], # array size mismatch
            [np.eye(4, dtype=int), np.eye(4, dtype=int)], # array size mismatch
            [np.ones((1, 3, 3), dtype=int),
             np.ones((1, 3, 3), dtype=int)], # not 2D
            [np.eye(3)*1.0, np.eye(3)*1.0], # not int
            ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                median_combine(image_list, perr)
            pass
        pass

    def test_bpmap_list_element_not_0_or_1(self):
        """Catch case where bpmap is not 0 or 1"""

        image_list = [np.zeros((3, 3))]
        bpmap_list = [np.zeros((3, 3))]

        for val in [-1, 2]:
            bpmap_list[0][0, 0] = val
            with self.assertRaises(TypeError):
                median_combine(image_list, bpmap_list)
            pass
        pass

    def test_accommodation_of_ndarray_inputs(self):
        """If inputs is a array_like (whether a single frame or a stack),
        the function will accommodate and convert each to a list of arrays."""

        # single frame
        image_list = np.ones((3,3))
        bpmap_list = np.zeros((3,3)).astype(int)
        # runs with no issues:
        median_combine(image_list, bpmap_list)

        # stack
        image_list = np.stack([np.ones((3,3)), 2*np.ones((3,3))])
        bpmap_list = np.stack([np.zeros((3,3)).astype(int),
                                np.zeros((3,3)).astype(int)])
        # runs with no issues:
        median_combine(image_list, bpmap_list)



class TestMeanCombine(unittest.TestCase):
    """Unit tests for mean_combine method."""

    def setUp(self):
        ones = np.ones((10, 10))
        zeros = np.zeros((10, 10), dtype=int)
        self.vals = [1, 0, 4]
        self.check_ims = []
        self.check_masks = []
        self.one_fr_mask = []
        for val in self.vals:
            self.check_ims.append(ones*val)
            self.check_masks.append(zeros.copy())
            self.one_fr_mask.append(zeros.copy())
            pass

        self.all_masks_i = (0, 0)
        self.single_mask_i = (0, 1)
        # Mask one pixel across all frames
        for m in self.check_masks:
            m[self.all_masks_i[0], self.all_masks_i[1]] = 1
            pass

        # Mask another pixel in only one frame
        self.check_masks[0][self.single_mask_i[0], self.single_mask_i[1]] = 1
        # Mask one pixel in one frame for one_fr_mask
        self.one_fr_mask[0][self.single_mask_i[0], self.single_mask_i[1]] = 1


    def test_mean_im(self):
        """Verify method calculates mean image."""
        tol = 1e-13

        check_combined_im = np.mean(self.check_ims, axis=0)
        # For the pixel that is masked throughout
        check_combined_im[self.all_masks_i] = 0
        unmasked_vals = np.delete(self.vals, self.single_mask_i[0])
        # For pixel that is only masked once
        check_combined_im[self.single_mask_i] = np.mean(unmasked_vals)

        combined_im, _, _, _ = mean_combine(self.check_ims, self.check_masks)

        self.assertTrue(np.max(np.abs(combined_im - check_combined_im)) < tol)


    def test_mean_mask(self):
        """Verify method calculates correct mean mask."""
        _, combined_mask, _, _ = mean_combine(self.check_ims, self.check_masks)
        check_combined_mask = np.zeros_like(combined_mask)
        # Only this pixel should be combined
        check_combined_mask[self.all_masks_i] = 1

        self.assertTrue((combined_mask == check_combined_mask).all())


    def test_darks_exception(self):
        """Half or more of the frames for a given pixel are masked."""
        # all frames for pixel (0,0) masked for inputs below
        _, _, _, enough_for_rn = mean_combine(self.check_ims, self.check_masks)
        self.assertTrue(enough_for_rn == False)


    def test_darks_mean_num_good_fr(self):
        """mean_num_good_fr as expected."""
        _, _, mean_num_good_fr, _  = mean_combine(self.check_ims,
                                                self.one_fr_mask)
        # 99 pixels with no mask on any of the 3 frames, one with one
        # frame masked
        expected_mean_num_good_fr = (3*99 + 2)/100
        self.assertEqual(mean_num_good_fr, expected_mean_num_good_fr)


    def test_invalid_image_list(self):
        """Invalid inputs caught"""

        bpmap_list = [np.zeros((3, 3), dtype=int), np.eye(3, dtype=int)]

        # for image_list
        perrlist = [
            'txt', None, 1j, 0, (5,), # not list
            (np.ones((3, 3)), np.ones((3, 3))), # not list
            [np.eye(3), np.eye(3), np.eye(3)], # length mismatch
            [np.eye(3)], # length mismatch
            [np.eye(4), np.eye(3)], # array size mismatch
            [np.eye(4), np.eye(4)], # array size mismatch
            [np.ones((1, 3, 3)), np.ones((1, 3, 3))], # not 2D
            ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                mean_combine(perr, bpmap_list)
            pass

        # special empty case
        with self.assertRaises(TypeError):
            mean_combine([], [])

        pass


    def test_invalid_bpmap_list(self):
        """Invalid inputs caught"""

        image_list = [np.zeros((3, 3)), np.eye(3)]

        # for image_list
        perrlist = [
            'txt', None, 1j, 0, (5,), # not list
            (np.ones((3, 3), dtype=int), np.ones((3, 3), dtype=int)), #not list
            [np.eye(3, dtype=int), np.eye(3, dtype=int),
             np.eye(3, dtype=int)], # length mismatch
            [np.eye(3, dtype=int)], # length mismatch
            [np.eye(4, dtype=int), np.eye(3, dtype=int)], # array size mismatch
            [np.eye(4, dtype=int), np.eye(4, dtype=int)], # array size mismatch
            [np.ones((1, 3, 3), dtype=int),
             np.ones((1, 3, 3), dtype=int)], # not 2D
            [np.eye(3)*1.0, np.eye(3)*1.0], # not int
            ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                mean_combine(image_list, perr)
            pass
        pass

    def test_bpmap_list_element_not_0_or_1(self):
        """Catch case where bpmap is not 0 or 1"""

        image_list = [np.zeros((3, 3))]
        bpmap_list = [np.zeros((3, 3))]

        for val in [-1, 2]:
            bpmap_list[0][0, 0] = val
            with self.assertRaises(TypeError):
                mean_combine(image_list, bpmap_list)
            pass
        pass

    def test_accommodation_of_ndarray_inputs(self):
        """If inputs is a array_like (whether a single frame or a stack),
        the function will accommodate and convert each to a list of arrays."""

        # single frame
        image_list = np.ones((3,3))
        bpmap_list = np.zeros((3,3)).astype(int)
        # runs with no issues:
        mean_combine(image_list, bpmap_list)

        # stack
        image_list = np.stack([np.ones((3,3)), 2*np.ones((3,3))])
        bpmap_list = np.stack([np.zeros((3,3)).astype(int),
                                np.zeros((3,3)).astype(int)])
        # runs with no issues:
        mean_combine(image_list, bpmap_list)


if __name__ == '__main__':
    unittest.main()
