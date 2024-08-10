"""Unit test suite for tpump_final.py.
"""
import unittest
import os
from pathlib import Path
import warnings
import tempfile

import numpy as np

import cal.util.ut_check as ut_check
from cal.util.read_metadata import Metadata as MetadataWrapper
from cal.tpumpanalysis.tpump_final import tpump_analysis, TPumpAnException
from cal.tpumpanalysis.trap_fitting import tau_temp

here = os.path.abspath(os.path.dirname(__file__))

class TestTPumpAnalysis(unittest.TestCase):
    """Unit tests for tpump_analysis().  Tests simulated data made
    with generate_test_data.py.  trap_fit() does not find all the traps for at
    least 3 temperatures (which is required for a fit for the cross section),
    so it is tested only in the theoretical cases where it may
    outperform. And I could probably be more precise in my trap generation in
    generate_test_data.py for better detection (see TODO in tpump_final.py).
    For example, for 2-traps in which one trap has a very thin peak in phase
    time while the other dominates the phase times, distinguishing
    which curve fits best becomes difficult, especially with noise. But for
    real data, I expect the code to work well for detectable traps."""
    def setUp(self):
        self.base_dir = Path(here, 'test_data')
        self.sub_no_noise_dir = Path(here, 'test_data_sub_frame_no_noise')
        self.sub_noise_dir = Path(here, 'test_data_sub_frame_noise')
        self.sub_noise_one_temp = Path(here,
            'test_data_sub_frame_noise_one_temp')
        self.sub_noise_no_prob1 = Path(here,
            'test_data_sub_frame_noise_no_prob1')
        self.sub_noise_no_sch1_traps = Path(here,
            'test_data_sub_frame_noise_no_sch1_traps')
        self.sub_noise_mean_field = Path(here,
            'test_data_sub_frame_noise_mean_field')
        self.test_data_sch2first = Path(here, 'test_data_sch2first')
        self.test_data_sample_data = Path(here, 'test_data_sample_data')
        self.test_data_bad_temp_label = Path(here, 'test_data_bad_temp_label')
        self.test_data_bad_sch_label = Path(here, 'test_data_bad_sch_label')
        self.test_data_duplicate_sch = Path(here, 'test_data_duplicate_sch')
        self.test_data_bad_emgain_header = Path(here,
            'test_data_bad_emgain_header')
        self.test_data_bad_emgain_value = Path(here,
            'test_data_bad_emgain_value')
        self.test_data_bad_ptime_header = Path(here,
            'test_data_bad_ptime_header')
        self.test_data_bad_ptime_value = Path(here,
            'test_data_bad_ptime_value')
        self.test_data_empty_temp_folder = Path(here,
            'test_data_empty_temp_folder')
        self.test_data_empty_sch_folder = Path(here,
            'test_data_empty_sch_folder')
        self.test_data_empty_base_dir = Path(here,
            'test_data_empty_base_dir')
        self.test_data_mat_wrong_shape = Path(here,
            'test_data_mat_wrong_shape')
        self.test_data_mat_bad_ptime = Path(here,
            'test_data_mat_bad_ptime')
        self.test_data_mat_no_us = Path(here,
            'test_data_mat_no_us')
        self.time_head = 'PHASE_T'
        self.emgain_head = 'EM_GAIN'
        self.num_pumps = {1:10000,2:10000,3:10000,4:10000}
        self.input_T = 160 #in K; outside range of temps in test data, for fun
        # for sub frames:
        self.meta_path_sub = Path(here, '..', 'util', 'metadata_test.yaml')
        self.meta_sub = MetadataWrapper(self.meta_path_sub)
        # for full frames (takes more storage, runs more slowly):
        self.meta_path_full = Path(here, '..', 'util', 'metadata.yaml')
        self.meta_full = MetadataWrapper(self.meta_path_full)
        self.nonlin_path = Path(here, '..', 'util', 'nonlin_sample.csv')
        self.length_lim = 5 # intended for generate_test_data.py
        self.temp_list = [180, 190, 200, 210, 220] # from generate_test_data.py
        # from generate_test_data.py, without the trap meant for finding only
        #after illumination correction. Special case of (10,10) where the
        #assignment of tau or tau2 to the '0' or '1' trap is somewhat random;
        # if one has an smaller starting temperature, then it will be assigned
        #tau.
        self.trap_dict_keys_no_ill = [((26, 28), 'CENel1', 0),
            ((50, 50), 'RHSel1', 0), ((60, 80), 'LHSel2', 0),
            ((68, 67), 'CENel2', 0), ((98, 33), 'LHSel3', 0),
            ((98, 33), 'RHSel2', 0), ((41, 15), 'CENel3', 0),
            ((89, 2), 'RHSel3', 0), ((89, 2), 'LHSel4', 0),
            [((10, 10), 'LHSel4', 0), ((10, 10), 'LHSel4', 1)],
            ((56, 56), 'CENel4', 0), ((77, 90), 'RHSel4', 0),
            ((77, 90), 'CENel2', 0)]
        # in the same order as above:
        self.trap_dict_E = [0.32, 0.32, 0.32, 0.32, 0.28, 0.32, 0.32, 0.32,
            0.28, [0.32, 0.28], 0.32, 0.28, 0.32]
        self.trap_dict_cs = [2e-15, 2e-15, 2e-15, 2e-15, 12e-15, 2e-15, 2e-15,
            2e-15, 12e-15, [2e-15, 12e-15], 2e-15, 12e-15, 2e-15]
        # when ill_corr=True, an additional, cloaked trap found:
        self.trap_dict_keys_ill = self.trap_dict_keys_no_ill + \
            [((13, 21), 'LHSel1', 0)]
        self.trap_dict_E_ill = self.trap_dict_E + [0.32]
        self.trap_dict_cs_ill = self.trap_dict_cs + [2e-15]
        # when no P1 traps present in scheme 1:
        self.trap_dict_keys_kprob2_ill = [((13, 21), 'LHSel2', 0),
            ((26, 28), 'CENel1', 0), ((41, 15), 'CENel3', 0)]
        self.trap_dict_keys_mean_field = [((68, 67), 'CENel2', 0),
            ((41, 15), 'CENel3', 0)]
        # filter out expected warnings
        warnings.filterwarnings('ignore', category=UserWarning,
                        module='cal.tpumpanalysis.tpump_final')
        warnings.filterwarnings('ignore', category=UserWarning,
                        module='cal.tpumpanalysis.trap_fitting')

    def test_tfit_const_True_sub_no_noise(self):
        '''No gain or noise present. Using tfit_const=True
        (uses trap_fit_const()), ill_corr = False.  Successfully
        finds all expected traps and stats on them, for every temperature.
        See generate_test_data.py for details on the simulated traps.'''
        #for no gain or variance or bias or offset of any kind:
        base_dir = self.sub_no_noise_dir
        # higher standards needed for no noise so that 1-trap fit can fail
        #when it needs to if the pixel is actually a 2-trap
        tau_fit_thresh = 0.9
        cs_fit_thresh = 0.8
        thresh_factor = 1.5
        input_T = 185
        ill_corr = False
        tfit_const = True
        # since we are dealing a no-noise case, we pad our uncertainty checks
        # with the machine-precision amount.  The number of terms on the
        # measured side of each inequality determines how many factors of eps
        # we pad by.
        # Introduce it here:
        eps = np.finfo(np.float64).eps

        (trap_dict, trap_densities, bad_fit_counter, pre_sub_el_count,
        unused_fit_data, unused_temp_fit_data, two_or_less_count,
        noncontinuous_count) = tpump_analysis(base_dir, self.time_head,
        self.emgain_head, self.num_pumps, self.meta_path_sub, nonlin_path=None,
        length_lim = self.length_lim, thresh_factor = thresh_factor,
        ill_corr = ill_corr, tfit_const = tfit_const, save_temps = None,
        tau_min = 0.7e-6, tau_max = 1.3e-2, tau_fit_thresh = tau_fit_thresh,
        tauc_min = 0, tauc_max = 1e-5, offset_min = 10, offset_max = 10,
        pc_min=0, pc_max=2,
        cs_fit_thresh = cs_fit_thresh, input_T=input_T)
        pass
        for i in range(len(self.trap_dict_keys_no_ill)):
            if i!= 9:
                t = self.trap_dict_keys_no_ill[i]
                self.assertTrue(t in trap_dict)
                # now make sure they appear for all temperatures, even
                # the harder-to-fit 2-traps, since there's no noise, assuming
                # the curve peak gets high enough for each temp; it doesn't for
                # (77,90), 'CENel2', but most temps have a detection:
                #self.assertTrue(set(trap_dict[t]['T']) == set(self.temp_list))
                self.assertTrue(len(trap_dict[t]['T']) >=len(self.temp_list)-1)
                self.assertTrue(np.isclose(trap_dict[t]['E'],
                                self.trap_dict_E[i], atol = 0.05))
                #generally, std dev is decent estimate of uncertainty, if
                # random error.  But let's do 2 sigma to account for other
                # potential fitting errors in first fit and any other machine
                # precision errors.  (The std devs below are >~1e-16, slightly
                # above machnine precision.)
                self.assertTrue(self.trap_dict_E[i] - 2*eps <=
                                trap_dict[t]['E'] + 2*trap_dict[t]['sig_E'])
                self.assertTrue(self.trap_dict_E[i] + 2*eps >=
                                trap_dict[t]['E'] - 2*trap_dict[t]['sig_E'])
                self.assertTrue(np.isclose(trap_dict[t]['cs'],
                                self.trap_dict_cs[i], rtol = 0.1))
                self.assertTrue(self.trap_dict_cs[i] - 2*eps <=
                                trap_dict[t]['cs'] + 2*trap_dict[t]['sig_cs'])
                self.assertTrue(self.trap_dict_cs[i] + 2*eps >=
                                trap_dict[t]['cs'] - 2*trap_dict[t]['sig_cs'])
                # must multiply cs (in cm^2) by 1e15 to get the cs input to
                # tau_temp() to be as expected, which is 1e-19 m^2
                self.assertTrue(np.isclose(trap_dict[t]['tau at input T'],
                    tau_temp(input_T, self.trap_dict_E[i],
                             self.trap_dict_cs[i]*1e15), rtol = 0.1))
                self.assertTrue(tau_temp(input_T, self.trap_dict_E[i],
                             self.trap_dict_cs[i]*1e15) - 2*eps <=
                             trap_dict[t]['tau at input T'] +
                             2*trap_dict[t]['sig_tau at input T'])
                self.assertTrue(tau_temp(input_T, self.trap_dict_E[i],
                             self.trap_dict_cs[i]*1e15) + 2*eps >=
                             trap_dict[t]['tau at input T'] -
                             2*trap_dict[t]['sig_tau at input T'])
            if i==9: #special case of (10,10)
                t1, t2 = self.trap_dict_keys_no_ill[i]
                self.assertTrue(t1 in trap_dict)
                self.assertTrue(t2 in trap_dict)
                # now make sure they appear for all temperatures
                # since there's no noise
                self.assertTrue(set(trap_dict[t1]['T']) == set(self.temp_list))
                self.assertTrue(set(trap_dict[t2]['T']) == set(self.temp_list))
                # check closeness and within error for t1
                self.assertTrue(np.isclose(trap_dict[t1]['E'],
                                self.trap_dict_E[i][0], atol = 0.05) or
                                np.isclose(trap_dict[t1]['E'],
                                self.trap_dict_E[i][1], atol = 0.05))
                self.assertTrue((self.trap_dict_E[i][0] - 2*eps <=
                            trap_dict[t1]['E'] + 2*trap_dict[t1]['sig_E']) or
                            (self.trap_dict_E[i][1] - 2*eps <=
                            trap_dict[t1]['E'] + 2*trap_dict[t1]['sig_E']))
                self.assertTrue((self.trap_dict_E[i][0] + 2*eps >=
                            trap_dict[t1]['E'] - 2*trap_dict[t1]['sig_E']) or
                            (self.trap_dict_E[i][1] + 2*eps >=
                            trap_dict[t1]['E'] - 2*trap_dict[t1]['sig_E']))
                self.assertTrue(np.isclose(trap_dict[t1]['cs'],
                                self.trap_dict_cs[i][0], rtol = 0.1) or
                                np.isclose(trap_dict[t1]['cs'],
                                self.trap_dict_cs[i][1], rtol = 0.1))
                self.assertTrue((self.trap_dict_cs[i][0] - 2*eps <=
                            trap_dict[t1]['cs'] + 2*trap_dict[t1]['sig_cs']) or
                                (self.trap_dict_cs[i][1] - 2*eps <=
                            trap_dict[t1]['cs'] + 2*trap_dict[t1]['sig_cs']))
                self.assertTrue((self.trap_dict_cs[i][0] + 2*eps >=
                            trap_dict[t1]['cs'] - 2*trap_dict[t1]['sig_cs']) or
                                (self.trap_dict_cs[i][1] + 2*eps >=
                            trap_dict[t1]['cs'] - 2*trap_dict[t1]['sig_cs']))
                self.assertTrue(np.isclose(trap_dict[t1]['tau at input T'],
                    tau_temp(input_T, self.trap_dict_E[i][0],
                             self.trap_dict_cs[i][0]*1e15), rtol = 0.1) or
                             np.isclose(trap_dict[t1]['tau at input T'],
                    tau_temp(input_T, self.trap_dict_E[i][1],
                             self.trap_dict_cs[i][1]*1e15), rtol = 0.1))
                self.assertTrue((tau_temp(input_T, self.trap_dict_E[i][0],
                             self.trap_dict_cs[i][0]*1e15) - 2*eps <=
                             trap_dict[t1]['tau at input T'] +
                             2*trap_dict[t1]['sig_tau at input T']) or
                             (tau_temp(input_T, self.trap_dict_E[i][1],
                             self.trap_dict_cs[i][1]*1e15) - 2*eps <=
                             trap_dict[t1]['tau at input T'] +
                             2*trap_dict[t1]['sig_tau at input T']))
                self.assertTrue((tau_temp(input_T, self.trap_dict_E[i][0],
                             self.trap_dict_cs[i][0]*1e15) + 2*eps >=
                             trap_dict[t1]['tau at input T'] -
                             2*trap_dict[t1]['sig_tau at input T']) or
                             (tau_temp(input_T, self.trap_dict_E[i][1],
                             self.trap_dict_cs[i][1]*1e15) + 2*eps >=
                             trap_dict[t1]['tau at input T'] -
                             2*trap_dict[t1]['sig_tau at input T']))
                # check closeness and within error for t2
                self.assertTrue(np.isclose(trap_dict[t2]['E'],
                                self.trap_dict_E[i][0], atol = 0.05) or
                                np.isclose(trap_dict[t2]['E'],
                                self.trap_dict_E[i][1], atol = 0.05))
                self.assertTrue((self.trap_dict_E[i][0] - 2*eps <=
                            trap_dict[t2]['E'] + 2*trap_dict[t2]['sig_E']) or
                            (self.trap_dict_E[i][1] - 2*eps <=
                            trap_dict[t2]['E'] + 2*trap_dict[t2]['sig_E']))
                self.assertTrue((self.trap_dict_E[i][0] + 2*eps >=
                            trap_dict[t2]['E'] - 2*trap_dict[t2]['sig_E']) or
                            (self.trap_dict_E[i][1] + 2*eps >=
                            trap_dict[t2]['E'] - 2*trap_dict[t2]['sig_E']))
                self.assertTrue(np.isclose(trap_dict[t2]['cs'],
                                self.trap_dict_cs[i][0], rtol = 0.1) or
                                np.isclose(trap_dict[t2]['cs'],
                                self.trap_dict_cs[i][1], rtol = 0.1))
                self.assertTrue((self.trap_dict_cs[i][0] - 2*eps <=
                            trap_dict[t2]['cs'] + 2*trap_dict[t2]['sig_cs']) or
                                (self.trap_dict_cs[i][1] - 2*eps <=
                            trap_dict[t2]['cs'] + 2*trap_dict[t2]['sig_cs']))
                self.assertTrue((self.trap_dict_cs[i][0] + 2*eps >=
                            trap_dict[t2]['cs'] - 2*trap_dict[t2]['sig_cs']) or
                                (self.trap_dict_cs[i][1] + 2*eps >=
                            trap_dict[t2]['cs'] - 2*trap_dict[t2]['sig_cs']))
                self.assertTrue(np.isclose(trap_dict[t2]['tau at input T'],
                    tau_temp(input_T, self.trap_dict_E[i][0],
                             self.trap_dict_cs[i][0]*1e15), rtol = 0.1) or
                             np.isclose(trap_dict[t2]['tau at input T'],
                    tau_temp(input_T, self.trap_dict_E[i][1],
                             self.trap_dict_cs[i][1]*1e15), rtol = 0.1))
                self.assertTrue((tau_temp(input_T, self.trap_dict_E[i][0],
                             self.trap_dict_cs[i][0]*1e15) - 2*eps <=
                             trap_dict[t2]['tau at input T'] +
                             2*trap_dict[t2]['sig_tau at input T']) or
                             (tau_temp(input_T, self.trap_dict_E[i][1],
                             self.trap_dict_cs[i][1]*1e15) - 2*eps <=
                             trap_dict[t2]['tau at input T'] +
                             2*trap_dict[t2]['sig_tau at input T']))
                self.assertTrue((tau_temp(input_T, self.trap_dict_E[i][0],
                             self.trap_dict_cs[i][0]*1e15) + 2*eps >=
                             trap_dict[t2]['tau at input T'] -
                             2*trap_dict[t2]['sig_tau at input T']) or
                             (tau_temp(input_T, self.trap_dict_E[i][1],
                             self.trap_dict_cs[i][1]*1e15) + 2*eps >=
                             trap_dict[t2]['tau at input T'] -
                             2*trap_dict[t2]['sig_tau at input T']))
        # since no noise at all:
        self.assertTrue(two_or_less_count == 0)
        self.assertTrue(noncontinuous_count == 0)
        self.assertTrue(unused_temp_fit_data == 0)
        #bad_fit_counter, pre_sub_el_count: for testing, and these aren't
        #strictly necessary for getting the desired result; happens to be no
        #bad fits in bad_fit_counter for this, though
        self.assertTrue(bad_fit_counter == 0)

        nrows, ncols, _ = self.meta_sub._imaging_area_geom()
        self.assertTrue(len(trap_densities) == 2)
        for tr in trap_densities:
            # self.assertTrue(np.isclose(tr[0], 11/(nrows*ncols), atol=1e-4) \
            #     or np.isclose(tr[0], 4/(nrows*ncols), atol=1e-4))
            self.assertTrue(tr[0] == 10/(nrows*ncols) or
                tr[0] == 4/(nrows*ncols))
            # 10 traps that have 0.32eV, 2e-15 cm^2
            #if np.isclose(tr[0], 11/(nrows*ncols), atol=1e-4):
            if tr[0] == 10/(nrows*ncols):
                # with atol aligning with bins input to tpump_final()
                self.assertTrue(np.isclose(tr[1], 0.32, atol=0.05))
                self.assertTrue(np.isclose(tr[2], 2e-15, rtol=0.1))
            # 4 traps that have 0.28eV, 12e-15 cm^2
            #if np.isclose(tr[0], 4/(nrows*ncols), atol=1e-4):
            if tr[0] == 4/(nrows*ncols):
                self.assertTrue(np.isclose(tr[1], 0.28, atol=0.05))
                self.assertTrue(np.isclose(tr[2], 12e-15, rtol=0.1))

    def test_tfit_const_True_sub_no_noise_ill(self):
        '''No gain or noise present. Using tfit_const=True
        (uses trap_fit_const()) and ill_corr = True.  Successfully
        finds all expected traps (as well as the one that is uncovered only
        with ill_corr=True, at (13,21)) and stats on them,
        for every temperature.
        See generate_test_data.py for details on the simulated traps.'''
        #for no gain or variance or bias or offset of any kind:
        base_dir = self.sub_no_noise_dir
        # higher standards needed for no noise so that 1-trap fit can fail
        #when it needs to if the pixel is actually a 2-trap
        tau_fit_thresh = 0.9
        cs_fit_thresh = 0.8
        thresh_factor = 1.5
        ill_corr = True
        tfit_const = True
        input_T = 185
        # since we are dealing a no-noise case, we pad our uncertainty checks
        # with the machine-precision amount.  The number of terms on the
        # measured side of each inequality determines how many factors of eps
        # we pad by.
        # Introduce it here:
        eps = np.finfo(np.float32).eps

        (trap_dict, trap_densities, bad_fit_counter, pre_sub_el_count,
        unused_fit_data, unused_temp_fit_data, two_or_less_count,
        noncontinuous_count) = tpump_analysis(base_dir, self.time_head,
        self.emgain_head, self.num_pumps, self.meta_path_sub, nonlin_path=None,
        length_lim = self.length_lim, thresh_factor = thresh_factor,
        ill_corr = ill_corr, tfit_const = tfit_const, save_temps = None,
        tau_min = 0.7e-6, tau_max = 1.3e-2, tau_fit_thresh = tau_fit_thresh,
        tauc_min = 0, tauc_max = 1e-5, offset_min = 10, offset_max = 10,
        pc_min=0, pc_max=2,
        cs_fit_thresh = cs_fit_thresh, input_T=input_T)
        pass
        for i in range(len(self.trap_dict_keys_ill)):
            if i!= 9:
                t = self.trap_dict_keys_ill[i]
                self.assertTrue(t in trap_dict)
                # now make sure they appear for all temperatures, even
                # the harder-to-fit 2-traps, since there's no noise, assuming
                # the curve peak gets high enough for each temp; it doesn't for
                # (77,90), 'CENel2', but most temps have a detection:
                #self.assertTrue(set(trap_dict[t]['T']) == set(self.temp_list))
                self.assertTrue(len(trap_dict[t]['T']) >=len(self.temp_list)-1)
                self.assertTrue(np.isclose(trap_dict[t]['E'],
                                self.trap_dict_E_ill[i], atol = 0.05))
                #generally, std dev is decent estimate of uncertainty, if
                # random error.  But let's do 2 sigma to account for other
                # potential fitting errors in first fit and any other machine
                # precision errors.  (The std devs below are >~1e-16, slightly
                # above machnine precision.)
                self.assertTrue(self.trap_dict_E_ill[i] - 2*eps <=
                                trap_dict[t]['E'] + 2*trap_dict[t]['sig_E'])
                self.assertTrue(self.trap_dict_E_ill[i] + 2*eps >=
                                trap_dict[t]['E'] - 2*trap_dict[t]['sig_E'])
                self.assertTrue(np.isclose(trap_dict[t]['cs'],
                                self.trap_dict_cs_ill[i], rtol = 0.1))
                self.assertTrue(self.trap_dict_cs_ill[i] - 2*eps <=
                                trap_dict[t]['cs'] + 2*trap_dict[t]['sig_cs'])
                self.assertTrue(self.trap_dict_cs_ill[i] + 2*eps >=
                                trap_dict[t]['cs'] - 2*trap_dict[t]['sig_cs'])
                # must multiply cs (in cm^2) by 1e15 to get the cs input to
                # tau_temp() to be as expected, which is 1e-19 m^2
                self.assertTrue(np.isclose(trap_dict[t]['tau at input T'],
                    tau_temp(input_T, self.trap_dict_E_ill[i],
                             self.trap_dict_cs_ill[i]*1e15), rtol = 0.1))
                self.assertTrue(tau_temp(input_T, self.trap_dict_E_ill[i],
                             self.trap_dict_cs_ill[i]*1e15) - 2*eps <=
                             trap_dict[t]['tau at input T'] +
                             2*trap_dict[t]['sig_tau at input T'])
                self.assertTrue(tau_temp(input_T, self.trap_dict_E_ill[i],
                             self.trap_dict_cs_ill[i]*1e15) + 2*eps >=
                             trap_dict[t]['tau at input T'] -
                             2*trap_dict[t]['sig_tau at input T'])
            if i==9: #special case of (10,10)
                t1, t2 = self.trap_dict_keys_ill[i]
                self.assertTrue(t1 in trap_dict)
                self.assertTrue(t2 in trap_dict)
                # now make sure they appear for all temperatures
                # since there's no noise
                self.assertTrue(set(trap_dict[t1]['T']) == set(self.temp_list))
                self.assertTrue(set(trap_dict[t2]['T']) == set(self.temp_list))
                # check closeness and within error for t1
                self.assertTrue(np.isclose(trap_dict[t1]['E'],
                                self.trap_dict_E_ill[i][0], atol = 0.05) or
                                np.isclose(trap_dict[t1]['E'],
                                self.trap_dict_E_ill[i][1], atol = 0.05))
                self.assertTrue((self.trap_dict_E_ill[i][0] - 2*eps <=
                            trap_dict[t1]['E'] + 2*trap_dict[t1]['sig_E']) or
                            (self.trap_dict_E_ill[i][1] - 2*eps <=
                            trap_dict[t1]['E'] + 2*trap_dict[t1]['sig_E']))
                self.assertTrue((self.trap_dict_E_ill[i][0] + 2*eps >=
                            trap_dict[t1]['E'] - 2*trap_dict[t1]['sig_E']) or
                            (self.trap_dict_E_ill[i][1] + 2*eps >=
                            trap_dict[t1]['E'] - 2*trap_dict[t1]['sig_E']))
                self.assertTrue(np.isclose(trap_dict[t1]['cs'],
                                self.trap_dict_cs_ill[i][0], rtol = 0.1) or
                                np.isclose(trap_dict[t1]['cs'],
                                self.trap_dict_cs_ill[i][1], rtol = 0.1))
                self.assertTrue((self.trap_dict_cs_ill[i][0] - 2*eps <=
                            trap_dict[t1]['cs'] + 2*trap_dict[t1]['sig_cs']) or
                                (self.trap_dict_cs_ill[i][1] - 2*eps <=
                            trap_dict[t1]['cs'] + 2*trap_dict[t1]['sig_cs']))
                self.assertTrue((self.trap_dict_cs_ill[i][0] + 2*eps >=
                            trap_dict[t1]['cs'] - 2*trap_dict[t1]['sig_cs']) or
                                (self.trap_dict_cs_ill[i][1] + 2*eps >=
                            trap_dict[t1]['cs'] - 2*trap_dict[t1]['sig_cs']))
                self.assertTrue(np.isclose(trap_dict[t1]['tau at input T'],
                    tau_temp(input_T, self.trap_dict_E_ill[i][0],
                             self.trap_dict_cs_ill[i][0]*1e15), rtol = 0.1) or
                             np.isclose(trap_dict[t1]['tau at input T'],
                    tau_temp(input_T, self.trap_dict_E_ill[i][1],
                             self.trap_dict_cs_ill[i][1]*1e15), rtol = 0.1))
                self.assertTrue((tau_temp(input_T, self.trap_dict_E_ill[i][0],
                             self.trap_dict_cs_ill[i][0]*1e15) - 2*eps <=
                             trap_dict[t1]['tau at input T'] +
                             2*trap_dict[t1]['sig_tau at input T']) or
                             (tau_temp(input_T, self.trap_dict_E_ill[i][1],
                             self.trap_dict_cs_ill[i][1]*1e15) - 2*eps <=
                             trap_dict[t1]['tau at input T'] +
                             2*trap_dict[t1]['sig_tau at input T']))
                self.assertTrue((tau_temp(input_T, self.trap_dict_E_ill[i][0],
                             self.trap_dict_cs_ill[i][0]*1e15) + 2*eps >=
                             trap_dict[t1]['tau at input T'] -
                             2*trap_dict[t1]['sig_tau at input T']) or
                             (tau_temp(input_T, self.trap_dict_E_ill[i][1],
                             self.trap_dict_cs_ill[i][1]*1e15) + 2*eps >=
                             trap_dict[t1]['tau at input T'] -
                             2*trap_dict[t1]['sig_tau at input T']))
                # check closeness and within error for t2
                self.assertTrue(np.isclose(trap_dict[t2]['E'],
                                self.trap_dict_E_ill[i][0], atol = 0.05) or
                                np.isclose(trap_dict[t2]['E'],
                                self.trap_dict_E_ill[i][1], atol = 0.05))
                self.assertTrue((self.trap_dict_E_ill[i][0] - 2*eps <=
                            trap_dict[t2]['E'] + 2*trap_dict[t2]['sig_E']) or
                            (self.trap_dict_E_ill[i][1] - 2*eps <=
                            trap_dict[t2]['E'] + 2*trap_dict[t2]['sig_E']))
                self.assertTrue((self.trap_dict_E_ill[i][0] + 2*eps >=
                            trap_dict[t2]['E'] - 2*trap_dict[t2]['sig_E']) or
                            (self.trap_dict_E_ill[i][1] + 2*eps >=
                            trap_dict[t2]['E'] - 2*trap_dict[t2]['sig_E']))
                self.assertTrue(np.isclose(trap_dict[t2]['cs'],
                                self.trap_dict_cs_ill[i][0], rtol = 0.1) or
                                np.isclose(trap_dict[t2]['cs'],
                                self.trap_dict_cs_ill[i][1], rtol = 0.1))
                self.assertTrue((self.trap_dict_cs_ill[i][0] - 2*eps <=
                            trap_dict[t2]['cs'] + 2*trap_dict[t2]['sig_cs']) or
                                (self.trap_dict_cs_ill[i][1] - 2*eps <=
                            trap_dict[t2]['cs'] + 2*trap_dict[t2]['sig_cs']))
                self.assertTrue((self.trap_dict_cs_ill[i][0] + 2*eps >=
                            trap_dict[t2]['cs'] - 2*trap_dict[t2]['sig_cs']) or
                                (self.trap_dict_cs_ill[i][1] + 2*eps >=
                            trap_dict[t2]['cs'] - 2*trap_dict[t2]['sig_cs']))
                self.assertTrue(np.isclose(trap_dict[t2]['tau at input T'],
                    tau_temp(input_T, self.trap_dict_E_ill[i][0],
                             self.trap_dict_cs_ill[i][0]*1e15), rtol = 0.1) or
                             np.isclose(trap_dict[t2]['tau at input T'],
                    tau_temp(input_T, self.trap_dict_E_ill[i][1],
                             self.trap_dict_cs_ill[i][1]*1e15), rtol = 0.1))
                self.assertTrue((tau_temp(input_T, self.trap_dict_E_ill[i][0],
                             self.trap_dict_cs_ill[i][0]*1e15) - 2*eps <=
                             trap_dict[t2]['tau at input T'] +
                             2*trap_dict[t2]['sig_tau at input T']) or
                             (tau_temp(input_T, self.trap_dict_E_ill[i][1],
                             self.trap_dict_cs_ill[i][1]*1e15) - 2*eps <=
                             trap_dict[t2]['tau at input T'] +
                             2*trap_dict[t2]['sig_tau at input T']))
                self.assertTrue((tau_temp(input_T, self.trap_dict_E_ill[i][0],
                             self.trap_dict_cs_ill[i][0]*1e15) + 2*eps >=
                             trap_dict[t2]['tau at input T'] -
                             2*trap_dict[t2]['sig_tau at input T']) or
                             (tau_temp(input_T, self.trap_dict_E_ill[i][1],
                             self.trap_dict_cs_ill[i][1]*1e15) + 2*eps >=
                             trap_dict[t2]['tau at input T'] -
                             2*trap_dict[t2]['sig_tau at input T']))
        # since no noise at all:
        self.assertTrue(two_or_less_count == 0)
        self.assertTrue(noncontinuous_count == 0)
        self.assertTrue(unused_temp_fit_data == 0)
        #bad_fit_counter, pre_sub_el_count: for testing, and these aren't
        #strictly necessary for getting the desired result; happens to be no
        #bad fits in bad_fit_counter for this, though
        self.assertTrue(bad_fit_counter == 0)

        nrows, ncols, _ = self.meta_sub._imaging_area_geom()
        self.assertTrue(len(trap_densities) == 2)
        for tr in trap_densities:
            # self.assertTrue(np.isclose(tr[0], 11/(nrows*ncols), atol=1e-4) \
            #     or np.isclose(tr[0], 4/(nrows*ncols), atol=1e-4))
            self.assertTrue(tr[0] == 11/(nrows*ncols) or
                tr[0] == 4/(nrows*ncols))
            # 11 traps that have 0.32eV, 2e-15 cm^2
            #if np.isclose(tr[0], 11/(nrows*ncols), atol=1e-4):
            if tr[0] == 11/(nrows*ncols):
                # with atol aligning with bins input to tpump_final()
                self.assertTrue(np.isclose(tr[1], 0.32, atol=0.05))
                self.assertTrue(np.isclose(tr[2], 2e-15, rtol=0.1))
            # 4 traps that have 0.28eV, 12e-15 cm^2
            #if np.isclose(tr[0], 4/(nrows*ncols), atol=1e-4):
            if tr[0] == 4/(nrows*ncols):
                self.assertTrue(np.isclose(tr[1], 0.28, atol=0.05))
                self.assertTrue(np.isclose(tr[2], 12e-15, rtol=0.1))

    def test_tfit_const_True_sub_noise_ill(self):
        '''Gain and noise present. Using tfit_const=True
        (uses trap_fit_const()) with ill_corr = True (which is needed
        to get all the traps right).  Successfully
        finds all expected traps for at least 3 temperatures, and finds the
        expected stats on them.  (If too much noise is added, some temperatures
        may be missed by the code, especially for the 2-trap pixels.)
        See generate_test_data.py
        for details on the simulated traps. save_temps is set to True, so we
        then load into the function the file temps.npy that was created, and
        the function runs successfully.'''

        tmp_fd, tmp_npy = tempfile.mkstemp(suffix='.npy')

        #ill_corr=False: calls (77,90) a single trap 'LHSel1' instead of
        # 'RHSel4' and 'CENel2'
        # and using trap_fit() catches everything, but several of them have
        #fewer than 3 temperatures
        base_dir = self.sub_noise_dir
        tau_fit_thresh = 0.8
        cs_fit_thresh = 0.8
        thresh_factor = 1.5
        input_T = 185
        ill_corr = True
        tfit_const = True
        #since I want to catch some hard-to-catch-and-fit 2-trap and have only
        # 2 non-zero bins, I choose bins with widths that roughly correspond
        # to the level of uncertainties
        bins_E = 50 # at 50% of default since noisy with inj charge
        bins_cs = 5 # at 50% of default since noisy with inj charge

        # warning that some traps have fewer than 3 temps
        with self.assertWarns(UserWarning):
            (trap_dict, trap_densities, bad_fit_counter, pre_sub_el_count,
            unused_fit_data, unused_temp_fit_data, two_or_less_count,
            noncontinuous_count) = tpump_analysis(base_dir, self.time_head,
            self.emgain_head, self.num_pumps, self.meta_path_sub,
            nonlin_path=None, length_lim = self.length_lim,
            thresh_factor = thresh_factor, ill_corr = ill_corr,
            tfit_const = tfit_const, save_temps=tmp_npy,
            tau_min = 0.7e-6, tau_max = 1.3e-2,
            tau_fit_thresh = tau_fit_thresh,
            tauc_min = 0, tauc_max = 1e-5, offset_min = 10, offset_max = 10,
            pc_min=0, pc_max=2,
            cs_fit_thresh = cs_fit_thresh, bins_E=bins_E, bins_cs=bins_cs,
            input_T=input_T)

        self.assertTrue(unused_fit_data > 0)
        # with noise, there will be some 2-traps that were not fitted as
        # intended for all temperatures, and they will not get matched up.
        # They will be deleted.  So unused_temp_fit_data should be 0.
        self.assertTrue(unused_temp_fit_data == 0)
        # there will be cases of missing temperatures due to the noise
        self.assertTrue(two_or_less_count > 0)
        self.assertTrue(noncontinuous_count > 0)
        # depending on thresh_factor, bad_fit_counter could be > 0
        self.assertTrue(pre_sub_el_count > 0)
        pass
        for i in range(len(self.trap_dict_keys_ill)):
            if i!= 9:
                t = self.trap_dict_keys_ill[i]
                self.assertTrue(t in trap_dict)
                # now make sure they appear for all temperatures, even
                # the harder-to-fit 2-traps, since there's no noise

                # A good uncertainty for a single-measured value (e.g., 1 set
                # of trap-pumped frames from which we extract 1 tau per trap)
                # is the standard deviation (std dev) from the fit for tau,
                # assuming random sources of noise.  However, since we have
                # non-normal noise from the detector, some of the tests below
                # occasionally fail if we only consider 1 std dev.  In light
                # of that, we use 2 standard deviations instead.
                self.assertTrue(len(trap_dict[t]['T']) >= 3)
                self.assertTrue(np.isclose(trap_dict[t]['E'],
                                self.trap_dict_E_ill[i], atol = 0.05))
                self.assertTrue(self.trap_dict_E_ill[i] <=
                                trap_dict[t]['E'] + 2*trap_dict[t]['sig_E'])
                self.assertTrue(self.trap_dict_E_ill[i] >=
                                trap_dict[t]['E'] - 2*trap_dict[t]['sig_E'])
                self.assertTrue(np.isclose(trap_dict[t]['cs'],
                                self.trap_dict_cs_ill[i], rtol = 0.1))
                self.assertTrue(self.trap_dict_cs_ill[i] <=
                                trap_dict[t]['cs'] + 2*trap_dict[t]['sig_cs'])
                self.assertTrue(self.trap_dict_cs_ill[i] >=
                                trap_dict[t]['cs'] - 2*trap_dict[t]['sig_cs'])
                # must multiply cs (in cm^2) by 1e15 to get the cs input to
                # tau_temp() to be as expected, which is 1e-19 m^2
                self.assertTrue(np.isclose(trap_dict[t]['tau at input T'],
                    tau_temp(input_T, self.trap_dict_E_ill[i],
                             self.trap_dict_cs_ill[i]*1e15), rtol = 0.1))
                self.assertTrue(tau_temp(input_T, self.trap_dict_E_ill[i],
                             self.trap_dict_cs_ill[i]*1e15) <=
                             trap_dict[t]['tau at input T'] +
                                2*trap_dict[t]['sig_tau at input T'])
                self.assertTrue(tau_temp(input_T, self.trap_dict_E_ill[i],
                             self.trap_dict_cs_ill[i]*1e15) >=
                             trap_dict[t]['tau at input T'] -
                                2*trap_dict[t]['sig_tau at input T'])
            if i==9: #special case of (10,10)
                t1, t2 = self.trap_dict_keys_ill[i]
                self.assertTrue(t1 in trap_dict)
                self.assertTrue(t2 in trap_dict)
                # now make sure they appear for all temperatures
                # since there's no noise
                self.assertTrue(len(trap_dict[t1]['T']) >= 3)
                self.assertTrue(len(trap_dict[t2]['T']) >= 3)
                # check closeness and within error for t1
                self.assertTrue(np.isclose(trap_dict[t1]['E'],
                                self.trap_dict_E_ill[i][0], atol = 0.05) or
                                np.isclose(trap_dict[t1]['E'],
                                self.trap_dict_E_ill[i][1], atol = 0.05))
                self.assertTrue((self.trap_dict_E_ill[i][0] <=
                            trap_dict[t1]['E'] + 2*trap_dict[t1]['sig_E']) or
                            (self.trap_dict_E_ill[i][1] <=
                            trap_dict[t1]['E'] + 2*trap_dict[t1]['sig_E']))
                self.assertTrue((self.trap_dict_E_ill[i][0] >=
                            trap_dict[t1]['E'] - 2*trap_dict[t1]['sig_E']) or
                            (self.trap_dict_E_ill[i][1] >=
                            trap_dict[t1]['E'] - 2*trap_dict[t1]['sig_E']))
                self.assertTrue(np.isclose(trap_dict[t1]['cs'],
                                self.trap_dict_cs_ill[i][0], rtol = 0.1) or
                                np.isclose(trap_dict[t1]['cs'],
                                self.trap_dict_cs_ill[i][1], rtol = 0.1))
                self.assertTrue((self.trap_dict_cs_ill[i][0] <=
                            trap_dict[t1]['cs'] + 2*trap_dict[t1]['sig_cs']) or
                                (self.trap_dict_cs_ill[i][1] <=
                            trap_dict[t1]['cs'] + 2*trap_dict[t1]['sig_cs']))
                self.assertTrue((self.trap_dict_cs_ill[i][0] >=
                            trap_dict[t1]['cs'] - 2*trap_dict[t1]['sig_cs']) or
                                (self.trap_dict_cs_ill[i][1] >=
                            trap_dict[t1]['cs'] - 2*trap_dict[t1]['sig_cs']))
                self.assertTrue(np.isclose(trap_dict[t1]['tau at input T'],
                    tau_temp(input_T, self.trap_dict_E_ill[i][0],
                           self.trap_dict_cs_ill[i][0]*1e15), rtol = 0.1) or
                           np.isclose(trap_dict[t1]['tau at input T'],
                    tau_temp(input_T, self.trap_dict_E_ill[i][1],
                             self.trap_dict_cs_ill[i][1]*1e15), rtol = 0.1))
                self.assertTrue((tau_temp(input_T, self.trap_dict_E_ill[i][0],
                             self.trap_dict_cs_ill[i][0]*1e15) <=
                             trap_dict[t1]['tau at input T'] +
                             2*trap_dict[t1]['sig_tau at input T']) or
                             (tau_temp(input_T, self.trap_dict_E_ill[i][1],
                             self.trap_dict_cs_ill[i][1]*1e15) <=
                             trap_dict[t1]['tau at input T'] +
                             2*trap_dict[t1]['sig_tau at input T']))
                self.assertTrue((tau_temp(input_T, self.trap_dict_E_ill[i][0],
                             self.trap_dict_cs_ill[i][0]*1e15) >=
                             trap_dict[t1]['tau at input T'] -
                             2*trap_dict[t1]['sig_tau at input T']) or
                             (tau_temp(input_T, self.trap_dict_E_ill[i][1],
                             self.trap_dict_cs_ill[i][1]*1e15) >=
                             trap_dict[t1]['tau at input T'] -
                             2*trap_dict[t1]['sig_tau at input T']))
                # check closeness and within error for t2
                self.assertTrue(np.isclose(trap_dict[t2]['E'],
                                self.trap_dict_E_ill[i][0], atol = 0.05) or
                                np.isclose(trap_dict[t2]['E'],
                                self.trap_dict_E_ill[i][1], atol = 0.05))
                self.assertTrue((self.trap_dict_E_ill[i][0] <=
                            trap_dict[t2]['E'] + 2*trap_dict[t2]['sig_E']) or
                            (self.trap_dict_E_ill[i][1] <=
                            trap_dict[t2]['E'] + 2*trap_dict[t2]['sig_E']))
                self.assertTrue((self.trap_dict_E_ill[i][0] >=
                            trap_dict[t2]['E'] - 2*trap_dict[t2]['sig_E']) or
                            (self.trap_dict_E_ill[i][1] >=
                            trap_dict[t2]['E'] - 2*trap_dict[t2]['sig_E']))
                self.assertTrue(np.isclose(trap_dict[t2]['cs'],
                                self.trap_dict_cs_ill[i][0], rtol = 0.1) or
                                np.isclose(trap_dict[t2]['cs'],
                                self.trap_dict_cs_ill[i][1], rtol = 0.1))
                self.assertTrue((self.trap_dict_cs_ill[i][0] <=
                            trap_dict[t2]['cs'] + 2*trap_dict[t2]['sig_cs']) or
                                (self.trap_dict_cs_ill[i][1] <=
                            trap_dict[t2]['cs'] + 2*trap_dict[t2]['sig_cs']))
                self.assertTrue((self.trap_dict_cs_ill[i][0] >=
                            trap_dict[t2]['cs'] - 2*trap_dict[t2]['sig_cs']) or
                                (self.trap_dict_cs_ill[i][1] >=
                            trap_dict[t2]['cs'] - 2*trap_dict[t2]['sig_cs']))
                self.assertTrue(np.isclose(trap_dict[t2]['tau at input T'],
                    tau_temp(input_T, self.trap_dict_E_ill[i][0],
                         self.trap_dict_cs_ill[i][0]*1e15), rtol = 0.1) or
                          np.isclose(trap_dict[t2]['tau at input T'],
                 tau_temp(input_T, self.trap_dict_E_ill[i][1],
                          self.trap_dict_cs_ill[i][1]*1e15), rtol = 0.1))
                self.assertTrue((tau_temp(input_T, self.trap_dict_E_ill[i][0],
                             self.trap_dict_cs_ill[i][0]*1e15) <=
                             trap_dict[t2]['tau at input T'] +
                             2*trap_dict[t2]['sig_tau at input T']) or
                             (tau_temp(input_T, self.trap_dict_E_ill[i][1],
                             self.trap_dict_cs_ill[i][1]*1e15) <=
                             trap_dict[t2]['tau at input T'] +
                             2*trap_dict[t2]['sig_tau at input T']))
                self.assertTrue((tau_temp(input_T, self.trap_dict_E_ill[i][0],
                             self.trap_dict_cs_ill[i][0]*1e15) >=
                             trap_dict[t2]['tau at input T'] -
                             2*trap_dict[t2]['sig_tau at input T']) or
                             (tau_temp(input_T, self.trap_dict_E_ill[i][1],
                             self.trap_dict_cs_ill[i][1]*1e15) >=
                             trap_dict[t2]['tau at input T'] -
                             2*trap_dict[t2]['sig_tau at input T']))

        nrows, ncols, _ = self.meta_sub._imaging_area_geom()
        self.assertTrue(len(trap_densities) == 2)
        for tr in trap_densities:
            # self.assertTrue(np.isclose(tr[0], 11/(nrows*ncols), atol=1e-4) \
            #    tra or np.isclose(tr[0], 4/(nrows*ncols), atol=1e-4))
            self.assertTrue(tr[0] == 11/(nrows*ncols) or
                tr[0] == 4/(nrows*ncols))
            # 11 traps that have 0.32eV, 2e-15 cm^2
            #if np.isclose(tr[0], 11/(nrows*ncols), atol=1e-4):
            if tr[0] == 11/(nrows*ncols):
                # with atol aligning with bins trainput to tpump_final()
                self.assertTrue(np.isclose(tr[1], 0.32, atol=0.05))
                self.assertTrue(np.isclose(tr[2], 2e-15, rtol=0.1))
            # 4 traps that have 0.28eV, 12e-15 cm^2
            #if np.isclose(tr[0], 4/(nrows*ncols), atol=1e-4):
            if tr[0] == 4/(nrows*ncols):
                self.assertTrue(np.isclose(tr[1], 0.28, atol=0.05))
                self.assertTrue(np.isclose(tr[2], 12e-15, rtol=0.1))

        # now load the file in
        (trap_dict, trap_densities, bad_fit_counter, pre_sub_el_count,
        unused_fit_data, unused_temp_fit_data, two_or_less_count,
        noncontinuous_count) = tpump_analysis(base_dir, self.time_head,
        self.emgain_head, self.num_pumps, self.meta_path_sub,
        nonlin_path=self.nonlin_path,
        length_lim = self.length_lim, thresh_factor = thresh_factor,
        ill_corr = ill_corr, tfit_const = tfit_const, save_temps = None,
        tau_min = 0.7e-6, tau_max = 1.3e-2,
        tau_fit_thresh = tau_fit_thresh,
        load_temps=tmp_npy,
        tauc_min = 0, tauc_max = 1e-5, offset_min = 10, offset_max = 10,
        pc_min=0, pc_max=2,
        cs_fit_thresh = cs_fit_thresh)

        #this happens when load_temps is not None
        self.assertTrue(unused_fit_data is None)
        # now delete the file created
        os.close(tmp_fd)
        os.unlink(tmp_npy)

    def test_tfit_const_True_sample_data(self):
        '''Gain and noise present. Using tfit_const=True
        (uses trap_fit_const()) with ill_corr = False and no nonlinearity
        correction. sample_data = True tested with a few files downloaded from
        Alfresco (a few from 170K, scheme 1). Then test that I can save
        temps.npy and use load_temps successfully.'''

        tmp_fd, tmp_npy = tempfile.mkstemp(suffix='.npy')

        base_dir = self.test_data_sample_data
        tau_fit_thresh = 0.8
        cs_fit_thresh = 0.8
        #set this high just to minimize how long code takes to run on these
        # bigger frames; just needs to find at least one trap
        thresh_factor = 10
        ill_corr = False
        tfit_const = True

        tpump_analysis(base_dir, self.time_head,
        self.emgain_head, self.num_pumps, self.meta_path_sub,
        nonlin_path=self.nonlin_path,
        length_lim = self.length_lim, thresh_factor = thresh_factor,
        ill_corr = ill_corr, tfit_const = tfit_const,
        tau_min = 0.7e-6, tau_max = 1.3e-2,
        tau_fit_thresh = tau_fit_thresh,
        tauc_min = 0, tauc_max = 1e-5, offset_min = 10, offset_max = 10,
        pc_min=0, pc_max=2, save_temps=tmp_npy,
        cs_fit_thresh = cs_fit_thresh, sample_data=True)
        pass

        # now load the file in
        (trap_dict, trap_densities, bad_fit_counter, pre_sub_el_count,
        unused_fit_data, unused_temp_fit_data, two_or_less_count,
        noncontinuous_count) = tpump_analysis(base_dir, self.time_head,
        self.emgain_head, self.num_pumps, self.meta_path_sub,
        nonlin_path=self.nonlin_path,
        length_lim = self.length_lim, thresh_factor = thresh_factor,
        ill_corr = ill_corr, tfit_const = tfit_const, save_temps = None,
        tau_min = 0.7e-6, tau_max = 1.3e-2,
        tau_fit_thresh = tau_fit_thresh, load_temps=tmp_npy,
        tauc_min = 0, tauc_max = 1e-5, offset_min = 10, offset_max = 10,
        pc_min=0, pc_max=2,
        cs_fit_thresh = cs_fit_thresh)

        #this happens when load_temps is not None
        self.assertTrue(unused_fit_data is None)
        # now remove temps.npy
        os.close(tmp_fd)
        os.unlink(tmp_npy)

    def test_tfit_const_True_sub_noise_ill_nonlin(self):
        '''Gain and noise present. Using tfit_const=True
        (uses trap_fit_const()) with ill_corr = True (which is needed
        to get all the traps right).  Nonlinearity correction is also turned on
        with a sample file (and this only affects EM gain > 1).  Not all
        simulated traps will be found necessarily since the generated data were
        made with emccd_detect, which cannot currently produce frames with
        nonlinearity.  This is why the estimation for eperdn (k gain) is
        slightly different from what is expected from how the data were
        generated.  Just a check to make sure everything runs fine.
        Uses 180K, scheme 1 from the sub frames with noise.  See
        generate_test_data.py for details on the simulated traps.'''
        base_dir = self.sub_noise_one_temp
        tau_fit_thresh = 0.8
        cs_fit_thresh = 0.8
        thresh_factor = 1.5
        ill_corr = True
        tfit_const = True

        (trap_dict, trap_densities, bad_fit_counter, pre_sub_el_count,
        unused_fit_data, unused_temp_fit_data, two_or_less_count,
        noncontinuous_count) = tpump_analysis(base_dir, self.time_head,
        self.emgain_head, self.num_pumps, self.meta_path_sub,
        nonlin_path=self.nonlin_path,
        length_lim = self.length_lim, thresh_factor = thresh_factor,
        ill_corr = ill_corr, tfit_const = tfit_const, save_temps = None,
        tau_min = 0.7e-6, tau_max = 1.3e-2,
        tau_fit_thresh = tau_fit_thresh,
        tauc_min = 0, tauc_max = 1e-5, offset_min = 10, offset_max = 10,
        pc_min=0, pc_max=2,
        cs_fit_thresh = cs_fit_thresh)
        pass

        self.assertTrue(pre_sub_el_count > 0)
        # only 1 temp, 1 scheme
        self.assertTrue(trap_dict == {})
        self.assertTrue(trap_densities == [])

    def test_tfit_const_True_sub_noise_nonlin(self):
        '''Gain and noise present. Using tfit_const=True
        (uses trap_fit_const()) with ill_corr = False.
        Nonlinearity correction is also turned on
        with a sample file (and this only affects EM gain > 1).  Not all
        simulated traps will be found necessarily since the generated data were
        made with emccd_detect, which cannot currently produce frames with
        nonlinearity.  This is why the estimation for eperdn (k gain) is
        slightly different from what is expected from how the data were
        generated.  Just a check to make sure everything runs fine.
        Uses 180K, scheme 1 from the sub frames with noise.  See
        generate_test_data.py for details on the simulated traps.'''
        base_dir = self.sub_noise_one_temp
        tau_fit_thresh = 0.8
        cs_fit_thresh = 0.8
        thresh_factor = 1.5
        ill_corr = False
        tfit_const = True

        (trap_dict, trap_densities, bad_fit_counter, pre_sub_el_count,
        unused_fit_data, unused_temp_fit_data, two_or_less_count,
        noncontinuous_count) = tpump_analysis(base_dir, self.time_head,
        self.emgain_head, self.num_pumps, self.meta_path_sub,
        nonlin_path=self.nonlin_path,
        length_lim = self.length_lim, thresh_factor = thresh_factor,
        ill_corr = ill_corr, tfit_const = tfit_const, save_temps = None,
        tau_min = 0.7e-6, tau_max = 1.3e-2,
        tau_fit_thresh = tau_fit_thresh,
        tauc_min = 0, tauc_max = 1e-5, offset_min = 10, offset_max = 10,
        pc_min=0, pc_max=2,
        cs_fit_thresh = cs_fit_thresh)
        pass
        self.assertTrue(pre_sub_el_count > 0)
        # only 1 temp, 1 scheme
        self.assertTrue(trap_dict == {})
        self.assertTrue(trap_densities == [])


    def test_tfit_const_True_sub_noise_kprob2(self):
        '''Gain and noise present. Using tfit_const=True
        (uses trap_fit_const()) with ill_corr = True.
        Nonlinearity correction is off.  No P1 traps in scheme 1 (not even
        in 'both' traps). Data for just 180K.
        We get the expected user warning for k_prob=1.
        Then we run the program with k_prob=2 and successfully find the traps.
        See generate_test_data.py for details on the simulated traps.'''
        base_dir = self.sub_noise_no_prob1
        tau_fit_thresh = 0.8
        cs_fit_thresh = 0.8
        thresh_factor = 1.5
        ill_corr = True
        tfit_const = True

        # using k_prob = 1 should raise exception
        with self.assertRaises(TPumpAnException):
            tpump_analysis(base_dir, self.time_head,
            self.emgain_head, self.num_pumps, self.meta_path_sub,
            nonlin_path=self.nonlin_path, k_prob = 1,
            length_lim = self.length_lim, thresh_factor = thresh_factor,
            ill_corr = ill_corr, tfit_const = tfit_const, save_temps = None,
            tau_min = 0.7e-6, tau_max = 1.3e-2,
            tau_fit_thresh = tau_fit_thresh,
            tauc_min = 0, tauc_max = 1e-5, offset_min = 10, offset_max = 10,
            pc_min=0, pc_max=2,
            cs_fit_thresh = cs_fit_thresh)

        (trap_dict, trap_densities, bad_fit_counter, pre_sub_el_count,
        unused_fit_data, unused_temp_fit_data, two_or_less_count,
        noncontinuous_count) = tpump_analysis(base_dir, self.time_head,
        self.emgain_head, self.num_pumps, self.meta_path_sub,
        nonlin_path=self.nonlin_path, k_prob = 2,
        length_lim = self.length_lim, thresh_factor = thresh_factor,
        ill_corr = ill_corr, tfit_const = tfit_const, save_temps = None,
        tau_min = 0.7e-6, tau_max = 1.3e-2,
        tau_fit_thresh = tau_fit_thresh,
        tauc_min = 0, tauc_max = 1e-5, offset_min = 10, offset_max = 10,
        pc_min=0, pc_max=2,
        cs_fit_thresh = cs_fit_thresh)

        self.assertTrue(unused_fit_data > 0)
        # with noise, there will be some 2-traps that were not fitted as
        # intended for all temperatures, and they will not get matched up.
        # They will be deleted.  So unused_temp_fit_data should be 0.
        self.assertTrue(unused_temp_fit_data == 0)
        self.assertTrue(two_or_less_count > 0)
        # tau_fit_thresh good; no bad fits
        self.assertTrue(bad_fit_counter == 0)
        self.assertTrue(pre_sub_el_count > 0)

        for i in range(len(self.trap_dict_keys_kprob2_ill)):
            t = self.trap_dict_keys_kprob2_ill[i]
            self.assertTrue(t in trap_dict)

    def test_tfit_const_True_sub_noise_no_sch1(self):
        '''Gain and noise present. Using tfit_const=True
        (uses trap_fit_const()) with ill_corr = True.
        Nonlinearity correction is off.  No P1 or P2 traps in scheme 1
        (not even in 'both' traps), which means there are no traps at all
        in scheme 1 that meet the length limit. (The one at (71,84) is a false
        trap that is present in all the data sets that doesn't meet the length
        limit requirement, self.length_limit = 5.)
        We get the expected exception.
        See generate_test_data.py for details on the simulated traps.'''
        base_dir = self.sub_noise_no_sch1_traps
        tau_fit_thresh = 0.8
        cs_fit_thresh = 0.8
        # with fewer traps, the standard deviation is less, so we increase the
        # threshold to ensure unintended dipoles aren't identified by chance
        thresh_factor = 3
        ill_corr = True
        tfit_const = True

        # using k_prob = 1 or 2 should raise exception
        for k_prob in [1,2]:
            with self.assertRaises(TPumpAnException):
                tpump_analysis(base_dir, self.time_head,
                self.emgain_head, self.num_pumps, self.meta_path_sub,
                nonlin_path=self.nonlin_path, k_prob = k_prob,
                length_lim = self.length_lim, thresh_factor = thresh_factor,
                ill_corr = ill_corr, tfit_const = tfit_const, save_temps=None,
                tau_min = 0.7e-6, tau_max = 1.3e-2,
                tau_fit_thresh = tau_fit_thresh,
                tauc_min = 0, tauc_max = 1e-5, offset_min = 10, offset_max=10,
                pc_min=0, pc_max=2,
                cs_fit_thresh = cs_fit_thresh)

    def test_tfit_const_False_sub_noise(self):
        '''Gain and noise present. Using tfit_const=False
        (uses trap_fit()) with ill_corr = False and no
        nonlinearity correction.  Even in the case of large tauc
        (i.e., Pc approximated as a constant not very good), trap_fit_const()
        outperforms.
        So this is just a check to make sure everything runs fine.
        Uses 180K, scheme 1 from the sub frames with noise.  See
        generate_test_data.py for details on the simulated traps.'''
        base_dir = self.sub_noise_one_temp
        tau_fit_thresh = 0.8
        cs_fit_thresh = 0.8
        thresh_factor = 1.5
        ill_corr = False
        tfit_const = False

        (trap_dict, trap_densities, bad_fit_counter, pre_sub_el_count,
        unused_fit_data, unused_temp_fit_data, two_or_less_count,
        noncontinuous_count) = tpump_analysis(base_dir, self.time_head,
        self.emgain_head, self.num_pumps, self.meta_path_sub,
        nonlin_path=None,
        length_lim = self.length_lim, thresh_factor = thresh_factor,
        ill_corr = ill_corr, tfit_const = tfit_const, save_temps = None,
        tau_min = 0.7e-6, tau_max = 1.3e-2,
        tau_fit_thresh = tau_fit_thresh,
        tauc_min = 0, tauc_max = 1e-5, offset_min = 10, offset_max = 10,
        pc_min=0, pc_max=2,
        cs_fit_thresh = cs_fit_thresh)

        pass
        self.assertTrue(pre_sub_el_count > 0)
        # only 1 temp, 1 scheme
        self.assertTrue(trap_dict == {})
        self.assertTrue(trap_densities == [])


    def test_tfit_const_False_sub_noise_ill(self):
        '''Gain and noise present. Using tfit_const=False
        (uses trap_fit()) with ill_corr = True and no
        nonlinearity correction.
        Just a check to make sure everything runs fine.
        Uses 180K, scheme 1 from the sub frames with noise.  See
        generate_test_data.py for details on the simulated traps.'''
        base_dir = self.sub_noise_one_temp
        tau_fit_thresh = 0.8
        cs_fit_thresh = 0.8
        thresh_factor = 1.5
        ill_corr = True
        tfit_const = False

        (trap_dict, trap_densities, bad_fit_counter, pre_sub_el_count,
        unused_fit_data, unused_temp_fit_data, two_or_less_count,
        noncontinuous_count) = tpump_analysis(base_dir, self.time_head,
        self.emgain_head, self.num_pumps, self.meta_path_sub,
        nonlin_path=None,
        length_lim = self.length_lim, thresh_factor = thresh_factor,
        ill_corr = ill_corr, tfit_const = tfit_const, save_temps = None,
        tau_min = 0.7e-6, tau_max = 1.3e-2,
        tau_fit_thresh = tau_fit_thresh,
        tauc_min = 0, tauc_max = 1e-5, offset_min = 10, offset_max = 10,
        pc_min=0, pc_max=2,
        cs_fit_thresh = cs_fit_thresh)
        pass
        self.assertTrue(pre_sub_el_count > 0)
        # only 1 temp, 1 scheme
        self.assertTrue(trap_dict == {})
        self.assertTrue(trap_densities == [])


    def test_tfit_const_False_sub_noise_nonlin(self):
        '''Gain and noise present. Using tfit_const=False
        (uses trap_fit()) with ill_corr = False and with
        nonlinearity correction.
        Just a check to make sure everything runs fine.
        Uses 180K, scheme 1 from the sub frames with noise.  See
        generate_test_data.py for details on the simulated traps.'''
        base_dir = self.sub_noise_one_temp
        tau_fit_thresh = 0.8
        cs_fit_thresh = 0.8
        thresh_factor = 1.5
        ill_corr = False
        tfit_const = False

        (trap_dict, trap_densities, bad_fit_counter, pre_sub_el_count,
        unused_fit_data, unused_temp_fit_data, two_or_less_count,
        noncontinuous_count) = tpump_analysis(base_dir, self.time_head,
        self.emgain_head, self.num_pumps, self.meta_path_sub,
        nonlin_path=self.nonlin_path,
        length_lim = self.length_lim, thresh_factor = thresh_factor,
        ill_corr = ill_corr, tfit_const = tfit_const, save_temps = None,
        tau_min = 0.7e-6, tau_max = 1.3e-2,
        tau_fit_thresh = tau_fit_thresh,
        tauc_min = 0, tauc_max = 1e-5, offset_min = 10, offset_max = 10,
        pc_min=0, pc_max=2,
        cs_fit_thresh = cs_fit_thresh)
        pass
        self.assertTrue(pre_sub_el_count > 0)
        # only 1 temp, 1 scheme
        self.assertTrue(trap_dict == {})
        self.assertTrue(trap_densities == [])


    def test_tfit_const_False_sub_noise_ill_nonlin(self):
        '''Gain and noise present. Using tfit_const=False
        (uses trap_fit()) with ill_corr = True and
        nonlinearity correction.
        Just a check to make sure everything runs fine.
        Uses 180K, scheme 1 from the sub frames with noise.  See
        generate_test_data.py for details on the simulated traps.'''
        base_dir = self.sub_noise_one_temp
        tau_fit_thresh = 0.8
        cs_fit_thresh = 0.8
        thresh_factor = 1.5
        ill_corr = False
        tfit_const = False

        (trap_dict, trap_densities, bad_fit_counter, pre_sub_el_count,
        unused_fit_data, unused_temp_fit_data, two_or_less_count,
        noncontinuous_count) = tpump_analysis(base_dir, self.time_head,
        self.emgain_head, self.num_pumps, self.meta_path_sub,
        nonlin_path=self.nonlin_path,
        length_lim = self.length_lim, thresh_factor = thresh_factor,
        ill_corr = ill_corr, tfit_const = tfit_const, save_temps = None,
        tau_min = 0.7e-6, tau_max = 1.3e-2,
        tau_fit_thresh = tau_fit_thresh,
        tauc_min = 0, tauc_max = 1e-5, offset_min = 10, offset_max = 10,
        pc_min=0, pc_max=2,
        cs_fit_thresh = cs_fit_thresh)
        pass
        self.assertTrue(pre_sub_el_count > 0)
        # only 1 temp, 1 scheme
        self.assertTrue(trap_dict == {})
        self.assertTrue(trap_densities == [])


    def test_tfit_const_False_sub_noise_mean_field(self):
        '''Gain and noise present. Using tfit_const=False
        (uses trap_fit()) with ill_corr = True and no
        nonlinearity correction.  Only data for 180K present.
        trap_fit_const() also successfully finds the
        traps, but in this regime of low mean field (low mean electron level
        per non-trap pixel), tau is no longer approximately constant over phase
        time, and it gives inaccurate values for tau in the curve fit.
        However, trap_fit_const() in general does a better job with curve
        fitting, and trap-pumped frames are typically by design for high charge
        packets per pixel.  Tests the case where the mean e- per pixel is
        less than 2500e-, the maximum e- that can be transferred to a P1 trap.
        (See doc string on mean_field in tpump_analysis() for details.)
        Uses only 180K from the sub frames with noise, and tauc is chosen to be
        large (3e-3 s), and only two traps present: (68, 67) and (41, 15).  See
        generate_test_data.py for details on the simulated traps.'''
        base_dir = self.sub_noise_mean_field
        tau_fit_thresh = 0.8
        cs_fit_thresh = 0.8
        #very few traps, so std dev not so big; need bigger thresh
        thresh_factor = 3
        ill_corr = True
        tfit_const = False
        #for the choice of tauc, E, and cs, peak P1 trap in scheme 1 peaks at
        #2090 e-, so that would approximate the mean field of e-
        mean_field = 2090 # e-
        tauc_max = 1e-2

        (trap_dict, trap_densities, bad_fit_counter, pre_sub_el_count,
        unused_fit_data, unused_temp_fit_data, two_or_less_count,
        noncontinuous_count) = tpump_analysis(base_dir, self.time_head,
        self.emgain_head, self.num_pumps, self.meta_path_sub,
        nonlin_path=self.nonlin_path,
        length_lim = self.length_lim, thresh_factor = thresh_factor,
        ill_corr = ill_corr, tfit_const = tfit_const, save_temps = None,
        tau_min = 0.7e-6, tau_max = 1.3e-2,
        tau_fit_thresh = tau_fit_thresh,
        tauc_min = 0, tauc_max = tauc_max,
        offset_min = 10, offset_max = 10, pc_min=0, pc_max=2,
        cs_fit_thresh = cs_fit_thresh)

        for i in range(len(self.trap_dict_keys_mean_field)):
            t = self.trap_dict_keys_mean_field[i]
            self.assertTrue(t in trap_dict)
            self.assertTrue(len(trap_dict[t]['T']) == 1)
        #no trap_densities since only 1 temp analyzed

    def test_folder_titles(self):
        '''Bad folder labels for temperature and scheme raise exceptions.'''
        base_dir = self.test_data_bad_temp_label
        with self.assertRaises(TPumpAnException):
            tpump_analysis(base_dir, self.time_head,
            self.emgain_head, self.num_pumps, self.meta_path_sub,
            nonlin_path=None)

        base_dir = self.test_data_bad_sch_label
        with self.assertRaises(TPumpAnException):
            tpump_analysis(base_dir, self.time_head,
            self.emgain_head, self.num_pumps, self.meta_path_sub,
            nonlin_path=None)

    def test_sch_folder_duplicate(self):
        '''Duplicate scheme number read off 2 different folders.'''
        base_dir = self.test_data_duplicate_sch
        with self.assertRaises(TPumpAnException):
            tpump_analysis(base_dir, self.time_head,
            self.emgain_head, self.num_pumps, self.meta_path_sub,
            nonlin_path=None)

    def test_not_fits_files(self):
        '''Files in scheme folder not .fits type.'''
        base_dir = self.test_data_sample_data #has .mat files instead
        with self.assertRaises(TPumpAnException):
            tpump_analysis(base_dir, self.time_head,
            self.emgain_head, self.num_pumps, self.meta_path_sub,
            nonlin_path=None)

    def test_emgain_header(self):
        '''Bad em_gain header key.'''
        base_dir = self.test_data_bad_emgain_header
        with self.assertRaises(TPumpAnException):
            tpump_analysis(base_dir, self.time_head,
            self.emgain_head, self.num_pumps, self.meta_path_sub,
            nonlin_path=None)

    def test_emgain_value(self):
        '''Bad em_gain header key value.'''
        base_dir = self.test_data_bad_emgain_value
        with self.assertRaises(TPumpAnException):
            tpump_analysis(base_dir, self.time_head,
            self.emgain_head, self.num_pumps, self.meta_path_sub,
            nonlin_path=None)

    def test_ptime_header(self):
        '''Bad phase time header key.'''
        base_dir = self.test_data_bad_ptime_header
        with self.assertRaises(TPumpAnException):
            tpump_analysis(base_dir, self.time_head,
            self.emgain_head, self.num_pumps, self.meta_path_sub,
            nonlin_path=None)

    def test_ptime_value(self):
        '''Bad phase time header key value.'''
        base_dir = self.test_data_bad_ptime_value
        with self.assertRaises(TPumpAnException):
            tpump_analysis(base_dir, self.time_head,
            self.emgain_head, self.num_pumps, self.meta_path_sub,
            nonlin_path=None)

    def test_empty_temp_folder(self):
        '''Empty temperature folder. Folder has 180K and an empty folder called
        190K.  Basically returns nothing useful for that
        temperature, but runs successfully otherwise.'''
        base_dir = self.test_data_empty_temp_folder
        _, _, _, _, _, _, _, _ = \
            tpump_analysis(base_dir, self.time_head,
            self.emgain_head, self.num_pumps, self.meta_path_sub,
            nonlin_path=None)

    def test_empty_sch_folder(self):
        '''Empty scheme folder.  Folder has Scheme 1 and an empty folder called
        Scheme 2.  Causes exception.'''
        base_dir = self.test_data_empty_sch_folder
        with self.assertRaises(TPumpAnException):
            tpump_analysis(base_dir, self.time_head,
            self.emgain_head, self.num_pumps, self.meta_path_sub,
            nonlin_path=None)

    def test_empty_base_dir(self):
        '''Empty base_dir folder. Basically returns nothing useful.'''
        base_dir = self.test_data_empty_base_dir
        trap_dict, _, _, _, _, _, _, _ = \
            tpump_analysis(base_dir, self.time_head,
            self.emgain_head, self.num_pumps, self.meta_path_sub,
            nonlin_path=None)
        self.assertTrue(trap_dict == {})

    def test_not_mat_files(self):
        '''Files in scheme folder not .mat type when sample_data = True.'''
        base_dir = self.sub_noise_one_temp #has .fits files in it instead
        with self.assertRaises(TPumpAnException):
            tpump_analysis(base_dir, self.time_head,
            self.emgain_head, self.num_pumps, self.meta_path_sub,
            nonlin_path=None, sample_data = True)

    def test_mat_wrong_shape(self):
        '''.mat file has wrong array shape (when sample_data = True).'''
        base_dir = self.test_data_mat_wrong_shape
        with self.assertRaises(TPumpAnException):
            tpump_analysis(base_dir, self.time_head,
            self.emgain_head, self.num_pumps, self.meta_path_sub,
            nonlin_path=None, sample_data = True)

    def test_mat_no_us(self):
        '''.mat file label has no '_us' in it (when sample_data = True).'''
        base_dir = self.test_data_mat_no_us
        with self.assertRaises(TPumpAnException):
            tpump_analysis(base_dir, self.time_head,
            self.emgain_head, self.num_pumps, self.meta_path_sub,
            nonlin_path=None, sample_data = True)

    def test_mat_bad_ptime(self):
        '''.mat file has bad phase time (a letter instead of a number)
        in the file label (when sample_data = True).'''
        base_dir = self.test_data_mat_bad_ptime
        with self.assertRaises(TPumpAnException):
            tpump_analysis(base_dir, self.time_head,
            self.emgain_head, self.num_pumps, self.meta_path_sub,
            nonlin_path=None, sample_data = True)

    def test_sch1_folder_absent(self):
        '''Scheme_1 folder absent.  Can't find eperdn as programmed.
        Raises exception.'''
        base_dir = self.test_data_sch2first
        with self.assertRaises(TPumpAnException):
            tpump_analysis(base_dir, self.time_head,
            self.emgain_head, self.num_pumps, self.meta_path_sub,
            nonlin_path=self.nonlin_path)

    def test_wrong_meta_file(self):
        '''meta_path specified inconsistent with frame size.
        Raises exception.'''
        base_dir = self.base_dir #sub frames
        meta_path = self.meta_path_full #for full frames

        with self.assertRaises(TPumpAnException):
            tpump_analysis(base_dir, self.time_head,
            self.emgain_head, self.num_pumps, meta_path,
            nonlin_path=self.nonlin_path)

    def test_wrong_nonlin_file(self):
        '''nonlin_path specified not valid.
        Raises FileNotFoundError.'''
        base_dir = self.base_dir
        nonlin_path = 'foo'
        with self.assertRaises(FileNotFoundError):
            tpump_analysis(base_dir, self.time_head,
            self.emgain_head, self.num_pumps, self.meta_path_sub,
            nonlin_path=nonlin_path)

    def test_base_dir(self):
        '''base_dir input bad.'''
        for perr in ['foo', 0, 1, -2.3]:
            with self.assertRaises(TypeError):
                tpump_analysis(self.base_dir, self.time_head, self.emgain_head,
                self.num_pumps, self.meta_path_sub, None, base_dir = perr)

    def test_load_temps(self):
        '''load_temps input bad.'''
        for perr in ['foo', 0, 1, -2.3]:
            with self.assertRaises(FileNotFoundError):
                tpump_analysis(self.base_dir, self.time_head, self.emgain_head,
                self.num_pumps, self.meta_path_sub, None, load_temps = perr)

    def test_nonlin_path(self):
        '''nonlin_path input bad.'''
        for perr in ['foo', 0, 1, -2.3]:
            with self.assertRaises(TypeError):
                tpump_analysis(self.base_dir, self.time_head, self.emgain_head,
                self.num_pumps, self.meta_path_sub, None, nonlin_path = perr)

    def test_sample_data(self):
        """sample_data input bad."""
        for perr in ['foo', 0, 1, -2.3]:
            with self.assertRaises(TypeError):
                tpump_analysis(self.base_dir, self.time_head, self.emgain_head,
                self.num_pumps, self.meta_path_sub, None, sample_data = perr)

    def test_head(self):
        """emgain_head or time_head inputs bad."""
        for perr in ut_check.strlist:
            with self.assertRaises(TypeError):
                tpump_analysis(self.base_dir, self.time_head, perr,
                self.num_pumps, self.meta_path_sub, None)
                tpump_analysis(self.base_dir, perr, self.emgain_head,
                self.num_pumps, self.meta_path_sub, None)

    def test_num_pumps_dict(self):
        '''num_pumps should be a dictionary.'''
        num_pumps_err = [10000,10000,10000,10000]
        with self.assertRaises(TypeError):
            tpump_analysis(self.base_dir, self.time_head, self.emgain_head,
                num_pumps_err, self.meta_path_sub, None)
            
    def test_num_pumps(self):
        """num_pumps input bad."""
        for perr in ut_check.psilist:
            num_pumps_err = {1:10000,2:perr,3:10000,4:10000}
            with self.assertRaises(TypeError):
                tpump_analysis(self.base_dir, self.time_head, self.emgain_head,
                num_pumps_err, self.meta_path_sub, None)

    def test_num_pumps_keys(self):
        """num_pumps keys should match scheme numbers, which are integers."""
        num_pumps_err = {1:1000,2:2000,3:3000,5:4000}
        with self.assertRaises(KeyError):
            tpump_analysis(self.base_dir, self.time_head, self.emgain_head,
            num_pumps_err, self.meta_path_sub, None)

    def test_meta_path(self):
        """meta_path input bad."""
        with self.assertRaises(FileNotFoundError):
            tpump_analysis(self.base_dir, self.time_head, self.emgain_head,
            self.num_pumps, 'foo', None)

    def test_thresh_factor(self):
        """thresh_factor input bad."""
        for perr in ut_check.rpslist:
            with self.assertRaises(TypeError):
                tpump_analysis(self.base_dir, self.time_head, self.emgain_head,
                self.num_pumps, self.meta_path_sub, None, thresh_factor=perr)

    def test_mean_field(self):
        """mean_field input bad."""
        #removed None from ut_check.rpslist since None is a valid input
        for perr in [1j, (1.,), [5, 5], 'txt', -1, 0]:
            with self.assertRaises(TypeError):
                tpump_analysis(self.base_dir, self.time_head, self.emgain_head,
                self.num_pumps, self.meta_path_sub, None, mean_field=perr)

    def test_k_prob(self):
        """k_prob input bad."""
        for perr in [3, 'foo', None, 0.1, -4]:
            with self.assertRaises(TypeError):
                tpump_analysis(self.base_dir, self.time_head, self.emgain_head,
                self.num_pumps, self.meta_path_sub, None, k_prob=perr)

    def test_ill_corr(self):
        """ill_corr input bad."""
        for perr in ['foo', 0, 1, -2.3]:
            with self.assertRaises(TypeError):
                tpump_analysis(self.base_dir, self.time_head, self.emgain_head,
                self.num_pumps, self.meta_path_sub, None, ill_corr = perr)

    def test_tfit_const(self):
        """tfit_const input bad."""
        for perr in ['foo', 0, 1, -2.3]:
            with self.assertRaises(TypeError):
                tpump_analysis(self.base_dir, self.time_head, self.emgain_head,
                self.num_pumps, self.meta_path_sub, None, tfit_const = perr)

    def test_tau_fit_thresh(self):
        """tau_fit_thresh input bad."""
        for perr in ut_check.rnslist:
            with self.assertRaises(TypeError):
                tpump_analysis(self.base_dir, self.time_head, self.emgain_head,
                self.num_pumps, self.meta_path_sub, None, tau_fit_thresh= perr)

    def test_tau_fit_thresh_val(self):
        """tau_fit_thresh >1 input bad."""
        for perr in [1.1, 2, 44.3]:
            with self.assertRaises(ValueError):
                tpump_analysis(self.base_dir, self.time_head, self.emgain_head,
                self.num_pumps, self.meta_path_sub, None, tau_fit_thresh= perr)

    def test_tau_min(self):
        """tau_min input bad."""
        for perr in ut_check.rnslist:
            with self.assertRaises(TypeError):
                tpump_analysis(self.base_dir, self.time_head, self.emgain_head,
                self.num_pumps, self.meta_path_sub, None, tau_min= perr)

    def test_tau_max(self):
        """tau_max input bad."""
        for perr in ut_check.rnslist:
            with self.assertRaises(TypeError):
                tpump_analysis(self.base_dir, self.time_head, self.emgain_head,
                self.num_pumps, self.meta_path_sub, None, tau_max= perr)

    def test_tau_min_val(self):
        """tau_max<=tau_min input bad."""
        tau_max = 1e-2
        with self.assertRaises(ValueError):
            tpump_analysis(self.base_dir, self.time_head, self.emgain_head,
            self.num_pumps, self.meta_path_sub, None, tau_min=tau_max,
            tau_max=tau_max)
        with self.assertRaises(ValueError):
            tpump_analysis(self.base_dir, self.time_head, self.emgain_head,
            self.num_pumps, self.meta_path_sub, None, tau_min=tau_max+1,
            tau_max=tau_max)

    def test_tauc_min(self):
        """tauc_min input bad."""
        for perr in ut_check.rnslist:
            with self.assertRaises(TypeError):
                tpump_analysis(self.base_dir, self.time_head, self.emgain_head,
                self.num_pumps, self.meta_path_sub, None, tauc_min= perr)

    def test_tauc_max(self):
        """tauc_max input bad."""
        for perr in ut_check.rnslist:
            with self.assertRaises(TypeError):
                tpump_analysis(self.base_dir, self.time_head, self.emgain_head,
                self.num_pumps, self.meta_path_sub, None, tauc_max= perr)

    def test_tauc_min_val(self):
        """tauc_max<=tauc_min input bad."""
        tauc_max = 1e-5
        with self.assertRaises(ValueError):
            tpump_analysis(self.base_dir, self.time_head, self.emgain_head,
            self.num_pumps, self.meta_path_sub, None, tauc_min=tauc_max,
            tauc_max=tauc_max)
        with self.assertRaises(ValueError):
            tpump_analysis(self.base_dir, self.time_head, self.emgain_head,
            self.num_pumps, self.meta_path_sub, None, tauc_min=tauc_max+1,
            tauc_max=tauc_max)

    def test_pc_min(self):
        """pc_min input bad."""
        for perr in ut_check.rnslist:
            with self.assertRaises(TypeError):
                tpump_analysis(self.base_dir, self.time_head, self.emgain_head,
                self.num_pumps, self.meta_path_sub, None, pc_min= perr)

    def test_pc_max(self):
        """pc_max input bad."""
        for perr in ut_check.rnslist:
            with self.assertRaises(TypeError):
                tpump_analysis(self.base_dir, self.time_head, self.emgain_head,
                self.num_pumps, self.meta_path_sub, None, pc_max= perr)

    def test_pc_min_val(self):
        """pc_max<=pc_min input bad."""
        pc_max = 2
        with self.assertRaises(ValueError):
            tpump_analysis(self.base_dir, self.time_head, self.emgain_head,
            self.num_pumps, self.meta_path_sub, None, pc_min=pc_max,
            pc_max=pc_max)
        with self.assertRaises(ValueError):
            tpump_analysis(self.base_dir, self.time_head, self.emgain_head,
            self.num_pumps, self.meta_path_sub, None, pc_min=pc_max+1,
            pc_max=pc_max)

    def test_offset_min(self):
        """offset_min input bad."""
        for perr in ut_check.rnslist:
            with self.assertRaises(TypeError):
                tpump_analysis(self.base_dir, self.time_head, self.emgain_head,
                self.num_pumps, self.meta_path_sub, None, offset_min= perr)

    def test_offset_max(self):
        """offset_max input bad."""
        for perr in ut_check.rnslist:
            with self.assertRaises(TypeError):
                tpump_analysis(self.base_dir, self.time_head, self.emgain_head,
                self.num_pumps, self.meta_path_sub, None, offset_max= perr)

    def test_cs_fit_thresh(self):
        """cs_fit_thresh input bad."""
        for perr in ut_check.rnslist:
            with self.assertRaises(TypeError):
                tpump_analysis(self.base_dir, self.time_head, self.emgain_head,
                self.num_pumps, self.meta_path_sub, None, cs_fit_thresh= perr)

    def test_cs_fit_thresh_val(self):
        """cs_fit_thresh >1 input bad."""
        for perr in [1.1, 2, 44.3]:
            with self.assertRaises(ValueError):
                tpump_analysis(self.base_dir, self.time_head, self.emgain_head,
                self.num_pumps, self.meta_path_sub, None, cs_fit_thresh= perr)

    def test_E_min(self):
        """E_min input bad."""
        for perr in ut_check.rnslist:
            with self.assertRaises(TypeError):
                tpump_analysis(self.base_dir, self.time_head, self.emgain_head,
                self.num_pumps, self.meta_path_sub, None, E_min= perr)

    def test_E_max(self):
        """E_max input bad."""
        for perr in ut_check.rnslist:
            with self.assertRaises(TypeError):
                tpump_analysis(self.base_dir, self.time_head, self.emgain_head,
                self.num_pumps, self.meta_path_sub, None, E_max= perr)

    def test_E_min_val(self):
        """E_max<=E_min input bad."""
        E_max = 1
        with self.assertRaises(ValueError):
            tpump_analysis(self.base_dir, self.time_head, self.emgain_head,
            self.num_pumps, self.meta_path_sub, None, E_min=E_max,
            E_max=E_max)
        with self.assertRaises(ValueError):
            tpump_analysis(self.base_dir, self.time_head, self.emgain_head,
            self.num_pumps, self.meta_path_sub, None, E_min=E_max+1,
            E_max=E_max)

    def test_cs_min(self):
        """cs_min input bad."""
        for perr in ut_check.rnslist:
            with self.assertRaises(TypeError):
                tpump_analysis(self.base_dir, self.time_head, self.emgain_head,
                self.num_pumps, self.meta_path_sub, None, cs_min= perr)

    def test_cs_max(self):
        """cs_max input bad."""
        for perr in ut_check.rnslist:
            with self.assertRaises(TypeError):
                tpump_analysis(self.base_dir, self.time_head, self.emgain_head,
                self.num_pumps, self.meta_path_sub, None, cs_max= perr)

    def test_cs_min_val(self):
        """cs_max<=cs_min input bad."""
        cs_max = 50
        with self.assertRaises(ValueError):
            tpump_analysis(self.base_dir, self.time_head, self.emgain_head,
            self.num_pumps, self.meta_path_sub, None, cs_min=cs_max,
            cs_max=cs_max)
        with self.assertRaises(ValueError):
            tpump_analysis(self.base_dir, self.time_head, self.emgain_head,
            self.num_pumps, self.meta_path_sub, None, cs_min=cs_max+1,
            cs_max=cs_max)

    def test_bins_E(self):
        """bins_E input bad."""
        for perr in ut_check.psilist:
            with self.assertRaises(TypeError):
                tpump_analysis(self.base_dir, self.time_head, self.emgain_head,
                self.num_pumps, self.meta_path_sub, None, bins_E= perr)

    def test_bins_cs(self):
        """bins_cs input bad."""
        for perr in ut_check.psilist:
            with self.assertRaises(TypeError):
                tpump_analysis(self.base_dir, self.time_head, self.emgain_head,
                self.num_pumps, self.meta_path_sub, None, bins_cs= perr)

    def test_input_T(self):
        """input_T input bad."""
        for perr in ut_check.rpslist:
            with self.assertRaises(TypeError):
                tpump_analysis(self.base_dir, self.time_head, self.emgain_head,
                self.num_pumps, self.meta_path_sub, None, input_T= perr)

if __name__ == '__main__':
    unittest.main(buffer=True)
    #warnings are filtered out, so if buffer=True is left out above, then
    # just the standard outputs of tpump_analysis() are printed
