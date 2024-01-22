"""Unit test suite for trap_id.py.
"""
import unittest

import numpy as np

import cal.util.ut_check as ut_check
from cal.tpumpanalysis.trap_id import illumination_correction, trap_id

class TestIlluminationCorrection(unittest.TestCase):
    """Unit tests for illumination_correction()."""
    def setUp(self):
        # make a frame with uneven dimensions (just for generality)
        # median of 20, delta of 3 on either side
        self.img = np.random.uniform(17, 23, size=(100,120))
        # area with defect, lining up with binsize of 10.  (If it didn't line
        # up with bin, it would just subtract some of the region, so easier
        # this way for testing):
        self.img[10:20,20:30] = 100
        # making a dipole above 3 sigma (sigma = sqrt(3) for this uniform
        #distribution) in this defect region
        self.img[20,20] += 10
        self.img[21,20] -= 10
        # more dipoles outside the defect region, +/- 3sigma of median, 20
        self.img[50,50] = 20+12
        self.img[49,50] = 20-11
        self.img[71,84] = 20+10
        self.img[72,84] = 20-12
        self.img[98,33] = 20+10
        self.img[97,33] = 20-10
        # choose reasonable bin size given density of traps
        self.binsize = 10

    def test_frame_False(self):
        """Verify that expected result is returned for ill_corr = False."""
        cor_img, local_ill = illumination_correction(self.img, self.binsize,
                                                    False)
        expected = self.img - 20
        exp_local_ill = self.img - expected
        # difference for all pixels should be roughly within delta (which is 3)
        self.assertTrue(np.isclose(expected, cor_img, atol=3).all())
        self.assertTrue(np.isclose(exp_local_ill, local_ill, atol=3).all())
        # test equal size
        img2 = self.img[:,0:100]
        expected2 = expected[:,0:100]
        exp_local_ill2 = img2 - expected2
        cor_img2, local_ill2 = illumination_correction(img2, self.binsize,
                                                    False)
        self.assertTrue(np.isclose(expected2, cor_img2, atol=3).all())
        self.assertTrue(np.isclose(exp_local_ill2, local_ill2, atol=3).all())

    def test_frame_None(self):
        """Verify that expected result is returned for binsize = None (same as
        when ill_corr = False)."""
        cor_img, local_ill = illumination_correction(self.img, None,
                                                    True)
        expected = self.img - 20
        exp_local_ill = self.img - expected
        # difference for all pixels should be roughly within delta (which is 3)
        self.assertTrue(np.isclose(expected, cor_img, atol=3).all())
        self.assertTrue(np.isclose(exp_local_ill, local_ill, atol=3).all())
        # test equal size
        img2 = self.img[:,0:100]
        expected2 = expected[:,0:100]
        exp_local_ill2 = img2 - expected2
        cor_img2, local_ill2 = illumination_correction(img2, None,
                                                    True)
        self.assertTrue(np.isclose(expected2, cor_img2, atol=3).all())
        self.assertTrue(np.isclose(exp_local_ill2, local_ill2, atol=3).all())

    def test_frame_None_False(self):
        """Verify that expected result is returned for binsize = None and
        when ill_corr = False (should be same as previous two tests)."""
        cor_img, local_ill = illumination_correction(self.img, None,
                                                    False)
        expected = self.img - 20
        exp_local_ill = self.img - expected
        # difference for all pixels should be roughly within delta (which is 3)
        self.assertTrue(np.isclose(expected, cor_img, atol=3).all())
        self.assertTrue(np.isclose(exp_local_ill, local_ill, atol=3).all())
        # test equal size
        img2 = self.img[:,0:100]
        expected2 = expected[:,0:100]
        exp_local_ill2 = img2 - expected2
        cor_img2, local_ill2 = illumination_correction(img2, None,
                                                    False)
        self.assertTrue(np.isclose(expected2, cor_img2, atol=3).all())
        self.assertTrue(np.isclose(exp_local_ill2, local_ill2, atol=3).all())

    def test_frame_True(self):
        """Verify that expected result is returned for ill_corr = True.  Should
        also detect local defect region now."""
        cor_img, local_ill = illumination_correction(self.img, self.binsize,
                                                    True)
        expected = self.img - 20
        # for defect region, minus 80 more for a total of 100
        expected[10:20, 20:30] -= 80
        exp_local_ill = self.img - expected
        # difference for all pixels should be roughly within delta (which is 3)
        self.assertTrue(np.isclose(expected, cor_img, atol=3).all())
        self.assertTrue(np.isclose(exp_local_ill, local_ill, atol=3).all())
        # test equal size
        img2 = self.img[:,0:100]
        expected2 = expected[:,0:100]
        exp_local_ill2 = img2 - expected2
        cor_img2, local_ill2 = illumination_correction(img2, self.binsize,
                                                    True)
        self.assertTrue(np.isclose(expected2, cor_img2, atol=3).all())
        self.assertTrue(np.isclose(exp_local_ill2, local_ill2, atol=3).all())


    def test_weird_bins(self):
        """Verify that weird bin sizes work, whether ill_corr is True or
        False."""
        img = np.ones([100,120])
        cor_img, loc = illumination_correction(img, binsize=1, ill_corr=True)
        cor_img2, loc2 = illumination_correction(img, binsize=23,
                                            ill_corr=False)
        cor_img3, loc3 = illumination_correction(img, binsize=120,
                                            ill_corr=True)
        cor_img4, loc4 = illumination_correction(img, binsize=100,
                                            ill_corr=False)
        expected = np.zeros([100,120])
        exp_loc = img - expected
        self.assertTrue(np.array_equal(expected, cor_img))
        self.assertTrue(np.array_equal(expected, cor_img2))
        self.assertTrue(np.array_equal(expected, cor_img3))
        self.assertTrue(np.array_equal(expected, cor_img4))
        self.assertTrue(np.array_equal(exp_loc, loc))
        self.assertTrue(np.array_equal(exp_loc, loc2))
        self.assertTrue(np.array_equal(exp_loc, loc3))
        self.assertTrue(np.array_equal(exp_loc, loc4))

    def test_img_input(self):
        """Verify that exception is raised."""
        for perr in ut_check.twoDlist:
            with self.assertRaises(TypeError):
                illumination_correction(perr, self.binsize, True)

    def test_binsize_input(self):
        """Verify that exception is raised."""
        for perr in ut_check.psilist:
            if perr is None:
                continue
            with self.assertRaises(TypeError):
                illumination_correction(self.img, perr, True)

    def test_ill_corr_input(self):
        """Verify that exception is raised."""
        for perr in ['foo', 0, 1, -2.3]:
            with self.assertRaises(TypeError):
                illumination_correction(self.img, self.binsize, perr)

class TestTrapID(unittest.TestCase):
    """Unit tests for trap_id()."""
    def setUp(self):
        #frame 1, median of 20
        self.img = np.random.uniform(17, 23, size=(100,120))
        # frame 2, median of 0 (shouldn't matter since delta of 3 threshold
        # determined for each frame; with ill_corr, such an amp discrepancy
        # b/w adjacent phase times wouldn't matter)
        self.img2 = np.random.uniform(-3, 3, size=(100,120))
        #frame 3, median of 20
        self.img3 = np.random.uniform(17, 23, size=(100,120))
        #frame 4, median of 20
        self.img4 = np.random.uniform(17, 23, size=(100,120))
        #frame 5, median of 20 (same phase time as that of img4)
        self.img5 = np.random.uniform(17, 23, size=(100,120))
        # area with defect (signal above 5 sigma, which is 5*sqrt(3),
        # but no dipole that stands out enough without illumination
        # correction), but defect not present in img5
        self.img[12:22,17:25] = 30+10
        self.img2[12:22,17:25] = 10+10
        self.img3[12:22,17:25] = 30+10
        self.img4[12:22,17:25] = 30+10
        # now a dipole that meets threshold around local mean doesn't meet
        # threshold around frame median; would be detected only after
        # illumination correction
        self.img[13,19] = 30+10
        self.img[14,19] = 30-10
        self.img2[13,19] = 10+10
        self.img2[14,19] = 10-10
        self.img3[13,19] = 30+10
        self.img3[14,19] = 30-10
        self.img4[13,19] = 30+10
        self.img4[14,19] = 30-10
        # making some dipoles above 5 sigma
        # 'above' dipole for all 4 times (all 5 frames)
        self.img[20,20] = 20+10
        self.img[21,20] = 20-10
        self.img2[20,20] = 0+10
        self.img2[21,20] = 0-10
        self.img3[20,20] = 20+10
        self.img3[21,20] = 20-10
        self.img4[20,20] = 20+10
        self.img4[21,20] = 20-10
        self.img5[20,20] = 20+10
        self.img5[21,20] = 20-10
        # 'below' dipole for first 3 times
        self.img[50,50] = 20+12
        self.img[49,50] = 20-11
        self.img2[50,50] = 0+12
        self.img2[49,50] = 0-10
        self.img3[50,50] = 20+12
        self.img3[49,50] = 20-11
        # 'above' dipole for first 2 times (doesn't meet length_limit)
        self.img[71,84] = 20+10
        self.img[72,84] = 20-12
        self.img2[71,84] = 0+10
        self.img2[72,84] = 0-12
        # 'below' dipole for first 2 and 5th frames (np.ceil(2+1/2) = 3, meets
        # length_limit
        self.img[68,67] = 20+10
        self.img[67,67] = 20-12
        self.img2[68,67] = 0+10
        self.img2[67,67] = 0-12
        self.img5[68,67] = 20+10
        self.img5[67,67] = 20-12
        # 'both' dipole present at the same phase time; 'above' for first 3,
        # 'below' for last 2-4
        self.img[98,33] = 20+10
        self.img[99,33] = 20-11
        self.img2[98,33] = 0+10
        self.img2[97,33] = 0-10
        self.img2[99,33] = 0-11
        self.img3[98,33] = 20+10
        self.img3[99,33] = 20-11
        self.img3[97,33] = 20-10
        self.img4[98,33] = 20+11
        self.img4[97,33] = 20-10
        # 'both' dipole present at the same phase time; 'above' for first 3,
        # 'below' only for 3-4 (doesn't meet length_limit)
        self.img[41,22] = 20+11
        self.img[42,22] = 20-11
        self.img2[41,22] = 0+11
        self.img2[42,22] = 0-11
        self.img3[41,22] = 20+11
        self.img3[42,22] = 20-11
        self.img3[40,22] = 20-10
        self.img4[41,22] = 20+10
        self.img4[40,22] = 20-10
        # 'above' dipole present at bottom (see that false positive caught)
        self.img[-1,55] = 20+10
        self.img[0,55] = 20-10
        self.img2[-1,55] = 0+10
        self.img2[0,55] = 0-10
        self.img3[-1,55] = 20+10
        self.img3[0,55] = 20-10
        self.img4[-1,55] = 20+10
        self.img4[0,55] = 20-10
        # 'below' dipole present at top (see that false positive caught)
        self.img[0,70] = 20+11
        self.img[-1,70] = 20-10
        self.img2[0,70] = 0+11
        self.img2[-1,70] = 0-10
        self.img3[0,70] = 20+11
        self.img3[-1,70] = 20-10
        self.img4[0,70] = 20+11
        self.img4[-1,70] = 20-10
        # 'above' only for the first 2 times, 'below' for the 3-4
        self.img[7,7] = 20+10
        self.img[8,7] = 20-10
        self.img2[7,7] = 0+10
        self.img2[8,7] = 0-10
        self.img3[7,7] = 20+10
        self.img3[6,7] = 20-10
        self.img4[7,7] = 20+10
        self.img4[6,7] = 20-10
        self.cor_img_stack = np.stack([self.img,self.img2,self.img3,self.img4,
            self.img5])
        self.timings = np.array([1,2,3,4,4]) # 1 duplicate set of phase times
        # all dipole amps are above 5sigma IF the mean in the randint
        # distributions that made the frames is actually the mean, but in
        # fact the mean is skewed upwards by the dipoles themselves.  So set
        # the thresh_factor a bit lower, to 4
        self.thresh_factor = 4
        self.length_limit = 3
        # making 2D arrays for these; their contents don't actually matter.
        # trap_id() simply takes them and puts them into the rc_* dictoinaries
        self.ill_corr_min = np.ones_like(self.img)
        self.ill_corr_max = 5*np.ones_like(self.img)
        self.rc_above, self.rc_below, self.rc_both = \
            trap_id(self.cor_img_stack, self.ill_corr_min, self.ill_corr_max,
                self.timings, self.thresh_factor,
                self.length_limit)

    def test_pass_length(self):
        """Catches 'above' and 'below' that pass length limit."""
        self.assertTrue(np.array_equal(self.rc_above[(20,20)]['amps_above'],
            np.array([30,10,30,30,30])))
        # only meets threshold for first 3 times, but all amps should be taken
        self.assertTrue(np.array_equal(self.rc_below[(50,50)]['amps_below'],
            np.array([32,12,32,self.img4[50,50], self.img5[50,50]])))
        self.assertTrue(np.array_equal(self.rc_below[(68,67)]['amps_below'],
            np.array([30,10,self.img3[68,67], self.img4[68,67], 30])))

    def test_both_same_time(self):
        """Handles 'both' case when 'above' and 'below' share the bright
        pixel at the same phase time."""
        self.assertTrue(np.array_equal(self.rc_both[(98,33)]['amps_both'],
           np.array([30,10,30,31, self.img5[98,33]])))
        self.assertTrue(np.array_equal(self.rc_both[(98,33)]['above']['amp'],
            np.array([30,10,30])))
        self.assertTrue(np.array_equal(self.rc_both[(98,33)]['above']['t'],
            np.array([1,2,3])))
        self.assertTrue(np.array_equal(self.rc_both[(98,33)]['below']['amp'],
            np.array([10,30,31])))
        self.assertTrue(np.array_equal(self.rc_both[(98,33)]['below']['t'],
            np.array([2,3,4])))

    def test_both_one_fail_length(self):
        """Handles 'both' case when 'above' passes length limit while 'below'
        doesn't. So the 'above' gets sorted into rc_above with all 5 amps."""
        self.assertTrue(np.array_equal(self.rc_above[(41,22)]['amps_above'],
            np.array([31,11,31,30, self.img5[41,22]])))
        self.assertFalse((41,22) in self.rc_both.keys())

    def test_false_positives(self):
        """Makes sure false-positive dipoles are ignored."""
        # checking false 'above'
        self.assertFalse((100,55) in self.rc_above.keys())
        # extra check
        self.assertFalse((0,55) in self.rc_above.keys())
        # checking false 'below'
        self.assertFalse((0,70) in self.rc_below.keys())
        # extra check
        self.assertFalse((100,70) in self.rc_below.keys())

    def test_both_fail_length(self):
        """Makes sure 'both' dipoles that fail length_limit don't show up."""
        self.assertFalse((7,7) in self.rc_above.keys())
        self.assertFalse((7,7) in self.rc_below.keys())

    def test_full_output(self):
        """Confirm full output of function is as expected."""
        self.assertEqual(set([(20,20),(41,22)]), set(self.rc_above.keys()))
        self.assertEqual(set([(50,50),(68,67)]), set(self.rc_below.keys()))
        self.assertEqual(set([(98,33)]), set(self.rc_both.keys()))

    def test_loc_med(self):
        """Confirm loc_med_min and loc_med_max recorded as expected."""
        for rc in self.rc_above:
            self.assertTrue(self.rc_above[rc]['loc_med_min'] ==
                                self.ill_corr_min[rc[0],rc[1]])
            self.assertTrue(self.rc_above[rc]['loc_med_max'] ==
                                self.ill_corr_max[rc[0],rc[1]])
        for rc in self.rc_below:
            self.assertTrue(self.rc_below[rc]['loc_med_min'] ==
                                self.ill_corr_min[rc[0],rc[1]])
            self.assertTrue(self.rc_below[rc]['loc_med_max'] ==
                                self.ill_corr_max[rc[0],rc[1]])
        for rc in self.rc_both:
            self.assertTrue(self.rc_both[rc]['loc_med_min'] ==
                                self.ill_corr_min[rc[0],rc[1]])
            self.assertTrue(self.rc_both[rc]['loc_med_max'] ==
                                self.ill_corr_max[rc[0],rc[1]])


    def test_cor_img_stack(self):
        """Verify that exception is raised."""
        for perr in ut_check.threeDlist:
            with self.assertRaises(TypeError):
                trap_id(perr, self.timings, self.ill_corr_min,
                    self.ill_corr_max, self.thresh_factor,
                    self.length_limit)

    def test_cor_img_stack_shape(self):
        '''Each frame should be castable to a stack (i.e., should have same
        shape).'''
        # input 5 frames, same number as in self.cor_img_stack
        perr = [np.ones([100,120]), np.ones([101,119]), np.ones([100,120]),
                np.ones([100,120]), np.ones([100,120])]
        with self.assertRaises(TypeError):
                trap_id(perr, self.timings, self.ill_corr_min,
                    self.ill_corr_max, self.thresh_factor,
                    self.length_limit)

    def test_ill_corr_min(self):
        '''Verify that exception is raised.'''
        for perr in ut_check.twoDlist:
            with self.assertRaises(TypeError):
                trap_id(self.cor_img_stack, self.timings, perr,
                    self.ill_corr_max, self.thresh_factor,
                    self.length_limit)

    def test_ill_corr_min_shape(self):
        '''Shape of ill_corr_min should agree with shape of frames in
        cor_img_stack.'''
        perr_list = [np.ones([100,121]), np.ones([99,120])]
        for perr in perr_list:
            with self.assertRaises(TypeError):
                trap_id(self.cor_img_stack, self.timings, perr,
                    self.ill_corr_max, self.thresh_factor,
                    self.length_limit)

    def test_ill_corr_max(self):
        '''Verify that exception is raised.'''
        for perr in ut_check.twoDlist:
            with self.assertRaises(TypeError):
                trap_id(self.cor_img_stack, self.timings, self.ill_corr_min,
                    perr, self.thresh_factor,
                    self.length_limit)

    def test_ill_corr_max_shape(self):
        '''Shape of ill_corr_max should agree with shape of frames in
        cor_img_stack.'''
        perr_list = [np.ones([100,121]), np.ones([99,120])]
        for perr in perr_list:
            with self.assertRaises(TypeError):
                trap_id(self.cor_img_stack, self.timings, self.ill_corr_min,
                    perr, self.thresh_factor,
                    self.length_limit)

    def test_timings(self):
        """Verify that exception is raised."""
        for perr in ut_check.oneDlist:
            with self.assertRaises(TypeError):
                trap_id(self.cor_img_stack, self.ill_corr_min,
                    self.ill_corr_max, perr, self.thresh_factor,
                    self.length_limit)

    def test_len_timings(self):
        """len(timings) should equal len(cor_img_stack)."""
        timings = [1,2,3]
        with self.assertRaises(ValueError):
            trap_id(self.cor_img_stack, self.ill_corr_min, self.ill_corr_max,
                    timings, self.thresh_factor,
                    self.length_limit)

    def test_thresh_factor(self):
        """Verify that exception is raised."""
        for perr in ut_check.rpslist:
            with self.assertRaises(TypeError):
                trap_id(self.cor_img_stack, self.ill_corr_min,
                    self.ill_corr_max, self.timings, perr,
                    self.length_limit)

    def test_length_limit(self):
        """Verify that exception is raised."""
        for perr in ut_check.psilist:
            with self.assertRaises(TypeError):
                trap_id(self.cor_img_stack, self.ill_corr_min,
                    self.ill_corr_max, self.timings, self.thresh_factor,
                    perr)

    def test_length_limit_gtr_len_cor_img_stack(self):
        """Verify exception raised if length_limit > len(cor_img_stack)."""
        length_limit = len(self.cor_img_stack)+1
        with self.assertRaises(ValueError):
            trap_id(self.cor_img_stack, self.ill_corr_min, self.ill_corr_max,
                self.timings, self.thresh_factor,
                length_limit)

if __name__ == '__main__':
    unittest.main()