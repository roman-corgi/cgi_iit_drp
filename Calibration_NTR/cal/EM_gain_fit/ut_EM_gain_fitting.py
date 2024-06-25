"""Unit tests for the EM_gain_fitting.py."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import warnings
from astropy.io import fits
import cal.util.ut_check as ut_check
from cal.EM_gain_fit.EM_gain_fitting import EMGainFit, mask_smoothing
from cal.EM_gain_fit.EM_gain_tools import _EM_gain_fit_conv

here = os.path.abspath(os.path.dirname(__file__))

config_path_locam = Path(here, '..', 'util', 'locam_config.yaml')
config_path_excam = Path(here, '..', 'util', 'excam_config.yaml')
gain_fit_path = Path(here, 'EM_gain_fit_params.yaml')
ut_yaml_dir = Path(here, 'ut_yaml')

nonlin_path_ones = Path(here, '..', 'util', 'testdata',
                        'ut_nonlin_array_ones.txt')
meta_path = str(Path(here, '..', 'util', 'metadata.yaml'))

# real data, 5 frames
excam_high_dir = Path(here, 'data', 'excam_high',
                 'n85 darks 1 s eperdn_8 cal frames')

#real data, 5 frames each
excam_low_dir = Path(here, 'data', 'excam_low',
            'G 10 HV 25_0 DC 3 V Light 051821 CCD 85')
excam_low_dirG1 = Path(here, 'data', 'excam_low',
            'G 1 HV 25_0 DC 3 V Light 051821 CCD 85')
# simulated data, 5 frames each
excam_low_dir_sim = Path(here, 'data', 'excam_low',
            'simulated','brights_G')
excam_low_dirG1_sim = Path(here, 'data', 'excam_low',
            'simulated','brights_G1')

# simulated data, 15 frames each
locam_dir = Path(here, 'data', 'locam','simulated',
            'brights_G')
locam_dir_bias = Path(here, 'data', 'locam','simulated',
            'bias_G')

#EXCAM low from TVAC, pupil images 
excam_low_TVAC_dir = Path(here, 'data', 'excam_low_pupil')

excam_high_obj = EMGainFit(cam_mode='excam_high', eperdn=8, bias_offset=0,
                com_gain=5000, exptime=1)
excam_high_obj.cosm_filter = 2
excam_low_obj = EMGainFit(cam_mode='excam_low', eperdn=6.5, bias_offset=0,
                com_gain=10, exptime=1, config_path=config_path_excam,
                gain_fit_path=gain_fit_path)
excam_low_obj.cosm_filter = 2

excam_low_sim_obj = EMGainFit(cam_mode='excam_low', eperdn=7, bias_offset=0,
                com_gain=40, exptime=1)
locam_obj = EMGainFit(cam_mode='locam', eperdn=7, bias_offset=None,
                com_gain=20, exptime=None, ampthresh_mask_thresh=10)

class TestEMGainFitInit(unittest.TestCase):
    '''Tests for EMGainFit class __init__.'''


    def test_success(self):
        '''successful class instantiation, including optional inputs'''
        emg = EMGainFit(cam_mode='excam_high', eperdn=8, bias_offset=0,
                        com_gain=5000, exptime=1)
        emg2 = EMGainFit(cam_mode='excam_low', eperdn=6.5, bias_offset=0,
                        com_gain=10, exptime=1,
                        config_path=config_path_excam,
                        gain_fit_path=gain_fit_path, binsize=5, 
                        smoothing_threshold=0.9, ampthresh_mask_thresh=50,
                        gmax_factor=2, min_freq=2)
        emg3 = EMGainFit(cam_mode='locam', eperdn=7, bias_offset=0,
                        com_gain=10, exptime=None)
        emg4 = EMGainFit(cam_mode='locam', eperdn=7, bias_offset=0,
                        com_gain=20, exptime=0.000441,
                        config_path=config_path_locam,
                        gain_fit_path=gain_fit_path)


    def test_cam_mode(self):
        '''bad input'''
        with self.assertRaises(ValueError):
            EMGainFit('foo', 7, 0, 100, 1)

    def test_eperdn(self):
        '''bad input'''
        for err in ut_check.rpslist:
            with self.assertRaises(TypeError):
                EMGainFit('locam', err, 0, 100, 1)

    # bias_offset covered by Process class in proc_cgi_frame

    def test_com_gain(self):
        '''bad input'''
        for err in ut_check.rpslist:
            with self.assertRaises(TypeError):
                EMGainFit('locam', 7, 0, err, 1)

    def test_exptime(self):
        '''bad input'''
        for err in ut_check.rpslist:
            with self.assertRaises(TypeError):
                EMGainFit('excam_high', 7, 0, 100, err)

    # config_path and gain_fit_path input checks covered by ut_loadyaml.py

    def test_binsize(self):
        '''bad input'''
        for err in ut_check.psilist:
            with self.assertRaises(TypeError):
                EMGainFit('excam_low', 7, 0, 100, 1, binsize=err)

    def test_smoothing_threshold(self):
        '''bad input'''
        for err in ut_check.rpslist:
            with self.assertRaises(TypeError):
                EMGainFit('excam_low', 7, 0, 100, 1, smoothing_threshold=err)

    def test_smoothing_threshold_value(self):
        '''bad input: can't be > 1'''
        with self.assertRaises(ValueError):
            EMGainFit('excam_low', 7, 0, 100, 1, smoothing_threshold=2)

    def test_ampthresh_mask_thresh(self):
        '''bad input'''
        for err in ut_check.psilist:
            with self.assertRaises(TypeError):
                EMGainFit('excam_low', 7, 0, 100, 1, ampthresh_mask_thresh=err)

    def test_gmax_factor(self):
        '''bad input'''
        for err in ut_check.rpslist:
            with self.assertRaises(TypeError):
                EMGainFit('excam_low', 7, 0, 100, 1, gmax_factor=err)

    def test_gmax_factor_value(self):
        '''bad input: can't be < 1'''
        with self.assertRaises(ValueError):
            EMGainFit('excam_low', 7, 0, 100, 1, gmax_factor=0.9)
    
    def test_min_freq(self):
        '''bad input'''
        for err in ut_check.nsilist:
            with self.assertRaises(TypeError):
                EMGainFit('excam_low', 7, 0, 100, 1, min_freq=err)

    def test_config_path(self):
        '''bad input'''
        with self.assertRaises(Exception):
            EMGainFit('locam', 7, 0, 100, 1, gain_fit_path='foo')
            # config_path checked by load_locam_config and load_excam_config

    def test_tol(self):
        '''bad input'''
        path = Path(ut_yaml_dir, 'bad_tol.yaml')
        with self.assertRaises(TypeError):
            EMGainFit('locam', 7, 0, 100, 1, gain_fit_path=path)

    def test_lambda_factor(self):
        '''bad input'''
        path = Path(ut_yaml_dir, 'bad_lambda_factor.yaml')
        with self.assertRaises(TypeError):
            EMGainFit('locam', 7, 0, 100, 1, gain_fit_path=path)

    def test_gain_factor(self):
        '''bad input'''
        path = Path(ut_yaml_dir, 'bad_gain_factor.yaml')
        with self.assertRaises(TypeError):
            EMGainFit('locam', 7, 0, 100, 1, gain_fit_path=path)

    def test_rn_mean(self):
        '''bad input'''
        path = Path(ut_yaml_dir, 'bad_rn_mean.yaml')
        with self.assertRaises(TypeError):
            EMGainFit('locam', 7, 0, 100, 1, gain_fit_path=path)

    def test_diff_tol(self):
        '''bad input'''
        path = Path(ut_yaml_dir, 'bad_diff_tol.yaml')
        with self.assertRaises(TypeError):
            EMGainFit('locam', 7, 0, 100, 1, gain_fit_path=path)

    def test_locam_tframe(self):
        '''bad input'''
        path = Path(ut_yaml_dir, 'bad_locam_tframe.yaml')
        with self.assertRaises(TypeError):
            EMGainFit('locam', 7, 0, 100, 1, config_path=path)

    def test_locam_g_max_comm(self):
        '''bad input'''
        path = Path(ut_yaml_dir, 'bad_locam_g_max_comm.yaml')
        with self.assertRaises(TypeError):
            EMGainFit('locam', 7, 0, 100, 1, config_path=path)

    def test_locam_g_max_comm_1(self):
        '''g_max_comm < 1 error'''
        path = Path(ut_yaml_dir, 'bad_locam_g_max_comm_1.yaml')
        with self.assertRaises(ValueError):
            EMGainFit('locam', 7, 0, 100, 1, config_path=path)

    def test_locam_darke(self):
        '''bad input'''
        path = Path(ut_yaml_dir, 'bad_locam_darke.yaml')
        with self.assertRaises(TypeError):
            EMGainFit('locam', 7, 0, 100, 1, config_path=path)

    def test_locam_cic(self):
        '''bad input'''
        path = Path(ut_yaml_dir, 'bad_locam_cic.yaml')
        with self.assertRaises(TypeError):
            EMGainFit('locam', 7, 0, 100, 1, config_path=path)

    def test_locam_fwc(self):
        '''bad input'''
        path = Path(ut_yaml_dir, 'bad_locam_fwc.yaml')
        with self.assertRaises(TypeError):
            EMGainFit('locam', 7, 0, 100, 1, config_path=path)

    def test_locam_fwc_em(self):
        '''bad input'''
        path = Path(ut_yaml_dir, 'bad_locam_fwc_em.yaml')
        with self.assertRaises(TypeError):
            EMGainFit('locam', 7, 0, 100, 1, config_path=path)

    def test_locam_e_max_age(self):
        '''bad input'''
        path = Path(ut_yaml_dir, 'bad_locam_e_max_age.yaml')
        with self.assertRaises(TypeError):
            EMGainFit('locam', 7, 0, 100, 1, config_path=path)

    def test_locam_rn(self):
        '''bad input'''
        path = Path(ut_yaml_dir, 'bad_locam_rn.yaml')
        with self.assertRaises(TypeError):
            EMGainFit('locam', 7, 0, 100, 1, gain_fit_path=path)

    def test_locam_divisor(self):
        '''bad input'''
        path = Path(ut_yaml_dir, 'bad_locam_divisor.yaml')
        with self.assertRaises(TypeError):
            EMGainFit('locam', 7, 0, 100, 1, gain_fit_path=path)

    def test_locam_locam_sat_thresh(self):
        '''bad input'''
        path = Path(ut_yaml_dir, 'bad_locam_sat_thresh.yaml')
        with self.assertRaises(TypeError):
            EMGainFit('locam', 7, 0, 100, 1, gain_fit_path=path)

    def test_locam_num_summed_frames(self):
        '''bad input'''
        path = Path(ut_yaml_dir, 'bad_locam_num_summed_frames.yaml')
        with self.assertRaises(TypeError):
            EMGainFit('locam', 7, 0, 100, 1, gain_fit_path=path)

    def test_locam_Nem(self):
        '''bad input'''
        path = Path(ut_yaml_dir, 'bad_locam_Nem.yaml')
        with self.assertRaises(TypeError):
            EMGainFit('locam', 7, 0, 100, 1, gain_fit_path=path)

    def test_excam_lthresh(self):
        '''bad input'''
        path = Path(ut_yaml_dir, 'bad_excam_lthresh.yaml')
        with self.assertRaises(TypeError):
            EMGainFit('excam_high', 7, 0, 100, 1, gain_fit_path=path)
        with self.assertRaises(TypeError):
            EMGainFit('excam_low', 7, 0, 100, 1, gain_fit_path=path)

    def test_excam_divisor(self):
        '''bad input'''
        path = Path(ut_yaml_dir, 'bad_excam_divisor.yaml')
        with self.assertRaises(TypeError):
            EMGainFit('excam_high', 7, 0, 100, 1, gain_fit_path=path)

    def test_excam_rn(self):
        '''bad input'''
        path = Path(ut_yaml_dir, 'bad_excam_rn.yaml')
        with self.assertRaises(TypeError):
            EMGainFit('excam_high', 7, 0, 100, 1, config_path=path)

    def test_excam_fwc(self):
        '''bad input'''
        path = Path(ut_yaml_dir, 'bad_excam_fwc.yaml')
        with self.assertRaises(TypeError):
            EMGainFit('excam_high', 7, 0, 100, 1, config_path=path)

    def test_excam_fwc_em(self):
        '''bad input'''
        path = Path(ut_yaml_dir, 'bad_excam_fwc_em.yaml')
        with self.assertRaises(TypeError):
            EMGainFit('excam_high', 7, 0, 100, 1, config_path=path)

    def test_excam_gmax(self):
        '''bad input'''
        path = Path(ut_yaml_dir, 'bad_excam_gmax.yaml')
        with self.assertRaises(TypeError):
            EMGainFit('excam_high', 7, 0, 100, 1, config_path=path)

    def test_excam_gmax_1(self):
        '''bad input'''
        path = Path(ut_yaml_dir, 'bad_excam_gmax_1.yaml')
        with self.assertRaises(ValueError):
            EMGainFit('excam_high', 7, 0, 100, 1, config_path=path)

     #sat_thresh, plat_thresh, cosm_filter checked by Process class
     # parameters in locam_config.yaml and excam_config.yaml checked by
     # load_locam_config() and load_excam_config().

    def test_key_error(self):
        '''If bad key, error caught in the unpacking of dictionary.'''
        path = Path(ut_yaml_dir, 'bad_key.yaml')
        with self.assertRaises(KeyError):
            EMGainFit('excam_high', 7, 0, 100, 1, gain_fit_path=path)

    def test_locam_key_error(self):
        '''If bad key, error caught in the unpacking of dictionary.'''
        path = Path(ut_yaml_dir, 'bad_locam_key.yaml')
        with self.assertRaises(KeyError):
            EMGainFit('locam', 7, 0, 100, 1, gain_fit_path=path)

    def test_excam_key_error(self):
        '''If bad key, error caught in the unpacking of dictionary.'''
        path = Path(ut_yaml_dir, 'bad_excam_key.yaml')
        with self.assertRaises(KeyError):
            EMGainFit('excam_low', 7, 0, 100, 1, gain_fit_path=path)


class TestEMGainFitReadInEXCAMFiles(unittest.TestCase):
    '''Tests for EMGainFit class method read_in_excam_files.'''

    def test_success(self):
        '''successful runs, including when optional inputs specified'''
        excam_high_obj.read_in_excam_files(excam_high_dir)
        excam_low_obj.read_in_excam_files(excam_low_dir,
                                            bad_pix=np.zeros((1200,2200)),
                                            nonlin_path=nonlin_path_ones,
                                            meta_path=meta_path,
                                            flat=np.ones((1200,2200)), 
                                            dark=np.zeros((1200,2200)),
                                            do_ampthresh=False, 
                                    mask=np.zeros((1024,1024)).astype(bool))
        excam_low_obj.read_in_excam_files(excam_low_dir,
                                            bad_pix=np.zeros((1200,2200)),
                                            nonlin_path=nonlin_path_ones,
                                            meta_path=meta_path,
                                            flat=np.ones((1200,2200)), 
                                            dark=np.zeros((1200,2200)),
                                            do_ampthresh=True, 
                                            mask=None)
        
    def test_cam_mode(self):
        '''calling function with wrong cam_mode'''
        with self.assertRaises(ValueError):
            locam_obj.read_in_excam_files(excam_low_dir)
    
    # bad_pix, nonlin_path, meta_path, flat, and dark checked by 
    # proc_cgi_frame.Process class call

    def test_do_ampthresh(self):
        '''bad input'''
        with self.assertRaises(TypeError):
            excam_low_obj.read_in_excam_files(excam_low_dir, 
                                              do_ampthresh='foo')
    
    def test_do_ampthresh_mask(self):
        '''do_ampthresh cannot be True while mask is provided as an input.'''
        with self.assertRaises(ValueError):
            excam_low_obj.read_in_excam_files(excam_low_dir, 
                                    do_ampthresh=True, 
                                    mask=np.zeros((1024,1024)).astype(bool))
            
    def test_do_ampthresh_None(self):
        '''do_ampthresh is None: results as expected.'''
        _, mask_stack = excam_low_obj.read_in_excam_files(excam_low_dir, 
                                            do_ampthresh=None)
        for m in mask_stack:
            self.assertTrue(m.max() == 1) # not a blank mask

        _, mask_stack2 = excam_high_obj.read_in_excam_files(excam_high_dir, 
                                            do_ampthresh=None)
        for m in mask_stack2:
            self.assertTrue(m.max() == 0) # no masking

        _, mask_stack3 = excam_high_obj.read_in_excam_files(excam_high_dir, 
                                        do_ampthresh=None, 
                                        mask=np.ones((1024,1024)).astype(bool))
        for m in mask_stack3:
            self.assertTrue(m.min() == 1) #input mask was used for each
        
    def test_mask(self):
        '''bad input'''
        for err in ut_check.twoDlist:
            with self.assertRaises(TypeError):
                excam_high_obj.read_in_excam_files(excam_high_dir, mask=err)
        # not a Boolean array
        with self.assertRaises(TypeError):
            excam_high_obj.read_in_excam_files(excam_high_dir, 
                                               mask=np.ones((1024,1024)))
            
    def test_frame_nonzero(self):
        '''frame must have electron counts > 0'''
        # simulate this by having subtracted dark that's too big
        with self.assertRaises(Exception):
            excam_high_obj.read_in_excam_files(excam_high_dir, 
                                               dark=1e12*np.ones((1200,2200)))
            
    def test_mask_dimensions(self):
        '''mask shape must be same as image area of frames'''
        with self.assertRaises(ValueError):
            excam_high_obj.read_in_excam_files(excam_high_dir, 
                                    mask=np.zeros((1000,1000)).astype(bool))
            
    def test_ampthresh_mask_thresh_size(self):
        '''Error when ampthresh_mask_thresh >= np.floor(frame size/2)'''
        excam_temp_obj = EMGainFit(cam_mode='excam_high', eperdn=7, 
                        bias_offset=0,
                        com_gain=10, exptime=1, config_path=config_path_excam,
                        gain_fit_path=gain_fit_path, 
                        ampthresh_mask_thresh=int(1024**2/2))
        with self.assertRaises(ValueError):
            excam_temp_obj.read_in_excam_files(excam_high_dir, 
                                               do_ampthresh=True)

    def test_temp_t(self):
        '''cannot find suitable mask for bad illuminated frame'''
        excam_temp_obj = EMGainFit(cam_mode='excam_low', eperdn=7, 
                        bias_offset=0,
                        com_gain=10, exptime=1, config_path=config_path_excam,
                        gain_fit_path=gain_fit_path, 
                        smoothing_threshold=np.finfo(float).eps)
        with self.assertRaises(Exception):
            # Smoothing needed if generating mask for this pupil image.
            # smoothing_threshold set too low so that this error is triggered
            frames, _ = excam_temp_obj.read_in_excam_files(excam_low_TVAC_dir,
                                                           desmear_flag=True)

    def test_ampthresh_mask_thresh_too_high(self):
        '''ampthresh_mask_thresh too high to get faithful mask.'''
        # pass through a frame that is not meant to have any masked pixels.
        # Default of 100 used for ampthresh_mask_thresh here.
        test_dir = Path(here, 'ut_data','excam_frame')
        with self.assertRaises(Exception):
            frames, _ = excam_low_obj.read_in_excam_files(test_dir)
    
    def test_read_in(self):
        '''result as expected'''
        # simulated frame with emccd_detect with flux=1 and gain=30 and
        # replaced 1 pixel (800,1300) with NaN and another (840,1340) with a 
        # saturated amount (500000)
        L=1
        g=30
        test_dir = Path(here, 'ut_data','excam_frame')
        frames, _ = excam_low_obj.read_in_excam_files(test_dir, 
                                                   do_ampthresh=False)

        # check that NaNs were removed
        self.assertTrue(np.isnan(frames).any() == False)
        self.assertTrue(frames.size == 1024*1024 - 2)
        self.assertTrue(np.isclose(frames.mean(), L*g, rtol=0.2))

class TestEMGainFitReadInLOCAMFiles(unittest.TestCase):
    '''Tests for EMGainFit class method read_in_locam_files.'''

    def test_success(self):
        '''successful runs, including when optional inputs specified'''
        # flat is roughly one times the bias amount times eperdn
        locam_obj.read_in_locam_files(locam_dir, locam_dir_bias,
                                      locam_obj.com_gain, 
                                      bad_pix=np.zeros((50,50)),
                                      summed_flat=np.ones((50,50)), 
                                      do_ampthresh=False, 
                                      mask=np.zeros((50,50)).astype(bool))

    def test_cam_mode(self):
        '''calling function with wrong cam_mode'''
        with self.assertRaises(ValueError):
            excam_low_obj.read_in_locam_files(locam_dir, locam_dir_bias,
                                            locam_obj.com_gain)

    def test_bad_pix(self):
        '''bad input'''
        for err in ut_check.twoDlist:
            with self.assertRaises(TypeError):
                locam_obj.read_in_locam_files(locam_dir, locam_dir_bias,
                                            locam_obj.com_gain,
                                              bad_pix=err)
                
    def test_bad_pix_shape(self):
        '''shape must match LOCAM frame shape'''
        with self.assertRaises(ValueError):
            locam_obj.read_in_locam_files(locam_dir, locam_dir_bias,
                                        locam_obj.com_gain,
                                        bad_pix=np.zeros((51,51)).astype(bool))

    def test_flat(self):
            '''bad input'''
            for err in ut_check.twoDlist:
                with self.assertRaises(TypeError):
                    locam_obj.read_in_locam_files(locam_dir, locam_dir_bias,
                                            locam_obj.com_gain,
                                                summed_lat=err)
                    
    def test_flat_shape(self):
        '''shape must match LOCAM frame shape'''
        with self.assertRaises(ValueError):
            locam_obj.read_in_locam_files(locam_dir, locam_dir_bias,
                                            locam_obj.com_gain,
                                        summed_flat=np.ones((51,51)))

    def test_do_ampthresh(self):
        '''bad input'''
        with self.assertRaises(TypeError):
            locam_obj.read_in_locam_files(locam_dir, locam_dir_bias,
                                            locam_obj.com_gain,
                                              do_ampthresh='foo')
    
    def test_do_ampthresh_mask(self):
        '''do_ampthresh cannot be True while mask is provided as an input.'''
        with self.assertRaises(ValueError):
            locam_obj.read_in_locam_files(locam_dir, locam_dir_bias,
                                    locam_obj.com_gain,
                                    do_ampthresh=True, 
                                    mask=np.zeros((50,50)).astype(bool))
            
    def test_do_ampthresh_None(self):
        '''do_ampthresh is None: results as expected.''' 
        _, mask_stack = locam_obj.read_in_locam_files(excam_low_TVAC_dir, 
                                            excam_low_dirG1,
                                            locam_obj.com_gain,
                                            do_ampthresh=None)
        for m in mask_stack:
            self.assertTrue(m.max() == 1) # not a blank mask

        _, mask_stack2 = locam_obj.read_in_locam_files(excam_low_TVAC_dir, 
                                            excam_low_dirG1,
                                            locam_obj.com_gain,
                                        mask=np.ones((1200,2200)).astype(bool))
        for m in mask_stack2:
            self.assertTrue(m.min() == 1) #input mask was used for each
        
    def test_mask(self):
        '''bad input'''
        for err in ut_check.twoDlist:
            with self.assertRaises(TypeError):
                locam_obj.read_in_locam_files(locam_dir, locam_dir_bias,
                                            locam_obj.com_gain, mask=err)
        # not a Boolean array
        with self.assertRaises(TypeError):
            locam_obj.read_in_locam_files(locam_dir, locam_dir_bias,
                                            locam_obj.com_gain,
                                               mask=np.ones((50,50)))
            
    def test_frame_nonzero(self):
        '''frame must have electron counts > 0'''
        # simulate this by having a weird flat; flats for LOCAM are before 
        # bias subtraction and e- conversion, so ones would be far too low
        with self.assertRaises(Exception):
            locam_obj.read_in_locam_files(locam_dir, locam_dir_bias,
                                            locam_obj.com_gain,
                                        summed_flat=np.ones((50,50)))
            
    def test_mask_dimensions(self):
        '''mask shape must be same as image area of frames'''
        with self.assertRaises(ValueError):
            locam_obj.read_in_locam_files(locam_dir, locam_dir_bias,
                                            locam_obj.com_gain,
                                    mask=np.zeros((70,70)).astype(bool))
            
    def test_ampthresh_mask_thresh_size(self):
        '''Error when ampthresh_mask_thresh >= np.floor(frame size/2)'''
        locam_temp_obj = EMGainFit(cam_mode='locam', eperdn=7, 
                        bias_offset=0,
                        com_gain=20, exptime=1, config_path=config_path_locam,
                        gain_fit_path=gain_fit_path, 
                        ampthresh_mask_thresh=int(50**2/2))
        with self.assertRaises(ValueError):
            locam_temp_obj.read_in_locam_files(locam_dir, 
                                            locam_dir_bias,
                                            locam_obj.com_gain)

    def test_temp_t(self):
        '''cannot find suitable mask for bad illuminated frame'''
        locam_temp_obj = EMGainFit(cam_mode='locam', eperdn=7, 
                        bias_offset=0,
                        com_gain=10, exptime=1, config_path=config_path_locam,
                        gain_fit_path=gain_fit_path, 
                        smoothing_threshold=np.finfo(float).eps)
        with self.assertRaises(Exception):
            # Smoothing needed if generating mask for this pupil image.
            # smoothing_threshold set too low so that this error is triggered
            frames, _ = locam_temp_obj.read_in_locam_files(locam_dir, 
                                                           locam_dir_bias,
                                                        locam_obj.com_gain)

    def test_ampthresh_mask_thresh_too_high(self):
        '''ampthresh_mask_thresh too high to get faithful mask.'''
        # pass through a frame that is not meant to have any masked pixels.
        # Default of 100 used for ampthresh_mask_thresh here.
        test_dir = Path(here, 'ut_data','locam_frame', 'bright')
        with self.assertRaises(Exception):
            frames, _ = locam_obj.read_in_locam_files(test_dir, locam_dir_bias,
                                                        locam_obj.com_gain)

    def test_read_in(self):
        '''result as expected'''
        # simulated frame with emccd_detect with LOCAM exposure time, summed
        # 10000 frames into one.  flux=1000 and gain=20 and
        # replaced 1 pixel with NaN and another with a saturated amount
        # (500000).  Same for bias, except no saturated pixel.
        test_dir = Path(here, 'ut_data','locam_frame', 'bright')
        bias_dir = Path(here, 'ut_data','locam_frame', 'bias')
        frames, _ = locam_obj.read_in_locam_files(test_dir, bias_dir,
                                               locam_obj.com_gain,
                                               do_ampthresh=False)

        # check that NaNs were removed
        self.assertTrue(np.isnan(frames).any() == False)
        self.assertTrue(frames.size == 50*50 - 2)

    def test_flat_bad_pix_works(self):
        '''Simple test to make sure bad_pix and summed_flat work as 
        expected for LOCAM.'''
        # bright: 3x3 frame of ones except for a 5 at [0,0]
        # bias:  3x3 frame of zeros
        # summed_flat:  3x3 frame of twos
        # bad_pix: 3x3 frame with just [0,0] as bad pixel
        # expected result: array of 8 values of 0.5; [0,0] would be 5/2,
        # but that is masked by bad_pix
        locam_temp_obj = EMGainFit(cam_mode='locam', eperdn=1, 
                        bias_offset=0,
                        com_gain=10, exptime=1, config_path=config_path_locam,
                        gain_fit_path=gain_fit_path)
        locam_temp_obj.config_dict['tframe'] = 1 # for simple division
        test_dir = Path(here, 'ut_data','locam_frame_flat_bad_pix', 'bright')
        bias_dir = Path(here, 'ut_data','locam_frame_flat_bad_pix', 'bias')
        flat_path = Path(here, 'ut_data','locam_frame_flat_bad_pix', 
                         'summed_flat.fits')
        bad_pix = Path(here, 'ut_data','locam_frame_flat_bad_pix', 
                         'bad_pix.fits')
        summed_flat = fits.getdata(flat_path)
        bad_pix = fits.getdata(bad_pix)
        fr, _ = locam_temp_obj.read_in_locam_files(test_dir, bias_dir,
                                               locam_obj.com_gain, 
                                               bad_pix=bad_pix,
                                               summed_flat=summed_flat,
                                               do_ampthresh=False)
        self.assertTrue(np.array_equal(fr, np.ones(8)*0.5))

class TestEMGainFit(unittest.TestCase):
    '''Tests for EM_gain_fit function.  Does a test for each cam_mode.'''

    def setUp(self):
        # a huge-number calculation can cause this sometimes
        warnings.filterwarnings('ignore', category=RuntimeWarning) 


    def test_excam_low(self):
        '''This is real data.  EM gain reported as 10.
        May take a while to run.'''
        frames, _ = excam_low_obj.read_in_excam_files(excam_low_dir, 
                                                      desmear_flag=True)
        framesG1, _ = excam_low_obj.read_in_excam_files(excam_low_dirG1, 
                                                        desmear_flag=True)
        exp_gain = frames.mean()/framesG1.mean() #  about 10
        # success not tested here since it could be False 
        # (see example_script.py)
        EMgain, e_mean, _, l_bg, g_bg = excam_low_obj.EM_gain_fit(frames, 
                                                                  framesG1)

        self.assertTrue(np.isclose(EMgain, exp_gain, rtol=0.1))
        self.assertTrue(np.isclose(e_mean, framesG1.mean(), rtol=0.1))

        # run with cut_option='high'; in this case, no exception raised 
        # (happens to work)
        excam_low_sim_obj.EM_gain_fit(frames, framesG1, cut_option='high')


    def test_excam_low_sim(self):
        '''Example where unity-gain directory used and not used.
        This was for simulated EM gain of 40 and fluxe of 720.02 e-.
        And this illustrates the option where the exact MLE result for
        the normal distribution is a good approximation and is used.'''
        exp_gain = 40
        exp_fluxe = 720.02
        frames, _ = excam_low_sim_obj.read_in_excam_files(excam_low_dir_sim)
        # success not tested here since it could be False 
        # (see example_script.py)
        EMgain, e_mean, _, l_bg, g_bg = excam_low_sim_obj.EM_gain_fit(frames)

        self.assertTrue(np.isclose(EMgain, exp_gain, rtol=0.01))
        self.assertTrue(np.isclose(e_mean, exp_fluxe, rtol=0.04))
        self.assertTrue(l_bg == (None, None, None))
        self.assertTrue(g_bg == (None, None, None))

        # run with unity directory
        framesG1, _ = excam_low_sim_obj.read_in_excam_files(
                                            excam_low_dirG1_sim)
        EMgain1, e_mean1, _, _, _ = excam_low_sim_obj.EM_gain_fit(frames,
                                                                framesG1)

        self.assertTrue(np.isclose(EMgain1, exp_gain, rtol=0.01))
        self.assertTrue(np.isclose(e_mean1, exp_fluxe, rtol=0.04))

        # run with cut_option='low' option; still good fit 
        EMgain2, e_mean2, _,  _, _ = excam_low_sim_obj.EM_gain_fit(frames,
                                                framesG1, cut_option='low')
        # uses non-Gaussian MLE here, so e_mean2 not necessarily reliable
        self.assertTrue(np.isclose(EMgain2, exp_gain, rtol=0.01))

    def test_excam_high(self):
        '''
        This is for real data.  EM gain reported as 5000.
        '''
        exp_gain = 5000
        frames, _ = excam_high_obj.read_in_excam_files(excam_high_dir)
        EMgain, e_mean, _,  _, _ = excam_high_obj.EM_gain_fit(frames, num_cuts=6) #decreased to 6 for sake of speed

        # gain is meant to be accurate, but not necessarily e_mean
        self.assertTrue(np.isclose(EMgain, exp_gain, rtol=0.1))

        # test with cut_option = 'no' option just to check that it works
        EMgain3, e_mean3, _,  _, _  = excam_high_obj.EM_gain_fit(frames, cut_option='no')

        # test with cut_option = 'low' and exception raised b/c 'low' 
        # method fails with high-gain frames
        with self.assertRaises(Exception):
            EMgain4, e_mean4, _,  _, _  = excam_high_obj.EM_gain_fit(frames, cut_option='low', num_cuts=3)

    def test_locam(self):
        '''Simulated data.  Simulated EM gain is 20, fluxe is
        0.4169*10000 frames = 4169 e- in each summed non-bias frame.
        15 summed frames in folders.'''
        exp_gain = 20
        # subtracting off the amount expected from bias subtraction,
        # which is the noise (dominated by CIC, .02e-) times 10000 frames
        exp_fluxe = 4169 - 10000*.02
        frames, _ = locam_obj.read_in_locam_files(locam_dir, locam_dir_bias,
                                            com_gain=20, do_ampthresh=False)
        EMgain, e_mean, _,  _, _ = locam_obj.EM_gain_fit(frames)
        self.assertTrue(np.isclose(EMgain, exp_gain, rtol=0.05))
        self.assertTrue(np.isclose(e_mean, exp_fluxe, rtol=0.05))

        # test with cut_option = 'low' option; still gets good fit
        EMgain2, e_mean2, _,  _, _ = locam_obj.EM_gain_fit(frames,
                                                            cut_option='low')
        # e_mean2 may not be accurate, so not tested 
        self.assertTrue(np.isclose(EMgain2, exp_gain, rtol=0.1))

        # test with cut_option = 'high' option just to check that it works
        EMgain3, e_mean3, _,  _, _ = locam_obj.EM_gain_fit(frames,
                                                            cut_option='high')


    def test_excam_high_framesG1(self):
        '''If framesG1 is not None in EXCAM high-gain mode,
        exception raised.'''
        frames, _ = excam_high_obj.read_in_excam_files(excam_high_dir)
        with self.assertRaises(ValueError):
            excam_high_obj.EM_gain_fit(frames, framesG1=2)


    def test_excam_high_nonnegative_hist(self):
        '''If frames.min() is not negative (and it should be for dark
        frames at high gain), exception raised.'''
        #reading in excam-low frames, which do not have negative pixels
        frames, _ = excam_low_obj.read_in_excam_files(excam_low_dir_sim)
        with self.assertRaises(Exception):
            excam_high_obj.EM_gain_fit(frames)
    
    @patch('cal.EM_gain_fit.EM_gain_tools._LogPoissonGamma')
    @patch('cal.EM_gain_fit.EM_gain_tools._PoissonGammaConvFFT')
    def test_lthresh_increment(self, mock_pgc, mock_lpg):
        '''If data has a lot of 0 counts for
        most of the histogram range, the probability distribution gets so
        low in MLE for these parts that the log of the likelihood goes to
        negative infinity.  If this happens, the code prompts the user to
        increase the lowest level of counts considered (and the relevant
        region of the histogram will in fact have non-negligible
        frequencies of counts).'''
        frames = np.ones(100)
        frames[1] = 2
        frames[2] = 3
        frames[3] = 4 # upper and lower bounds different now; and 3 bins: 1,2,3
        mock_pgc.side_effect = [np.ones(3), 2*np.ones(3), 0*np.ones(3), 
                                np.ones(3)]
        mock_lpg.side_effect = [-np.inf*np.ones(3)]
        orig_freq = excam_low_obj.min_freq 
        excam_low_obj.min_freq = 1
        with self.assertRaises(ValueError):
            excam_low_obj.EM_gain_fit(frames, cut_option='low')
        # set back
        excam_low_obj.min_freq = orig_freq

    def test_fluxe_bounds(self):
        '''If fluxe_l >= fluxe_u in _EM_gain_fit_cov(), an error is raised.'''
        frames = np.array([.1, .12, .2])*np.finfo(float).eps
        with self.assertRaises(ValueError):
            # fluxe_u below float machine precision
            _EM_gain_fit_conv(frames, 1, 10, 5000, 110, 0, .01, 0)

    def test_gain_bounds(self):
        '''If gain_l >= gain_u in _EM_gain_fit_cov(), an error is raised.'''
        # divide by huge number so that fluxe_u gets below machine precision
        frames = np.array([.1, .12, .2])
        with self.assertRaises(ValueError):
            # low gmax of 2 used to trigger error
            _EM_gain_fit_conv(frames, 1, 10, 2, 110, 0, .01, 0)

    def test_cut_option(self):
        '''bad input'''
        for err in ut_check.strlist:
            if err is None:
                continue
            with self.assertRaises(TypeError):
                excam_low_obj.EM_gain_fit(np.ones(3), cut_option=err)
        
        '''bad string value'''
        with self.assertRaises(ValueError):
            excam_low_obj.EM_gain_fit(np.ones(3), cut_option='foo')

    def test_num_cuts(self):
        '''bad input'''
        for err in ut_check.psilist:
            with self.assertRaises(TypeError):
                excam_low_obj.EM_gain_fit(np.ones(3), num_cuts=err)

        '''bad value: must be >= 3'''
        with self.assertRaises(ValueError):
            excam_low_obj.EM_gain_fit(np.ones(3), num_cuts=2)

    def test_min_freq_high(self):
        '''min_freq too high.'''
        orig_min_freq = excam_low_obj.min_freq
        frames = np.linspace(1,10,10)
        excam_low_obj.min_freq = 2
        with self.assertRaises(ValueError):
            excam_low_obj.EM_gain_fit(frames, cut_option='low')
        with self.assertRaises(ValueError):
            excam_low_obj.EM_gain_fit(frames, cut_option='high')

        # set back
        excam_low_obj.min_freq = orig_min_freq

    def test_min_freq_low(self):
        '''min_freq too low.'''
        orig_min_freq = excam_low_obj.min_freq
        frames = np.linspace(1,10,10)
        excam_low_obj.min_freq = 0 # this freq not found in frames b/c too low
        with self.assertRaises(Exception):
            excam_low_obj.EM_gain_fit(frames, cut_option='low')
        with self.assertRaises(Exception):
            excam_low_obj.EM_gain_fit(frames, cut_option='high')

        # set back
        excam_low_obj.min_freq = orig_min_freq

    def test_min_freq_zero(self):
        '''min_freq happens at zero.'''
        orig_min_freq = excam_low_obj.min_freq
        # higher values have lower frequency:
        frames = np.linspace(0,10,10)*np.linspace(10,0,10)
        excam_low_obj.min_freq = 10 # this freq found at 0
        with self.assertRaises(ValueError):
            excam_low_obj.EM_gain_fit(frames, cut_option='low')
        with self.assertRaises(ValueError):
            excam_low_obj.EM_gain_fit(frames, cut_option='high')

        # set back
        excam_low_obj.min_freq = orig_min_freq

    @patch('cal.EM_gain_fit.EM_gain_tools._EM_gain_fit_conv')
    def test_max_gain(self, mock_res):
        '''In 'high' mode, if max gain found at first cut, error.'''
        EMgain = [20, 15, 10]
        e_mean = 0.02
        success = True
        lam_b_g = (1e-5, 0.02, 1)
        gain_b_g = (1.0001, 12, 30)
        res0 = EMgain[0], e_mean, success, lam_b_g, gain_b_g
        res1 = EMgain[1], e_mean, success, lam_b_g, gain_b_g
        res2 = EMgain[2], e_mean, success, lam_b_g, gain_b_g
        mock_res.side_effect = [res0, res1, res2]

        with self.assertRaises(Exception):
            excam_low_obj.EM_gain_fit(np.ones(10), cut_option='high', 
                                      num_cuts=3)

    @patch('cal.EM_gain_fit.EM_gain_tools._EM_gain_fit_conv')
    def test_min_gain(self, mock_res):
        '''In 'low' mode, if no local min in gain found, error.'''
        EMgain = [20, 15, 10] # no local min
        e_mean = 0.02
        success = True
        lam_b_g = (1e-5, 0.02, 1)
        gain_b_g = (1.0001, 12, 30)
        res0 = EMgain[0], e_mean, success, lam_b_g, gain_b_g
        res1 = EMgain[1], e_mean, success, lam_b_g, gain_b_g
        res2 = EMgain[2], e_mean, success, lam_b_g, gain_b_g
        mock_res.side_effect = [res0, res1, res2]

        with self.assertRaises(Exception):
            excam_low_obj.EM_gain_fit(np.ones(10), cut_option='low', 
                                      num_cuts=3)

class TestMaskSmoothing(unittest.TestCase):
    '''Tests for mask_smoothing() function.'''

    def test_img(self):
        '''bad input'''
        for err in ut_check.twoDlist:
            with self.assertRaises(TypeError):
                mask_smoothing(err, binsize=1)

    def test_binsize(self):
        '''bad input'''
        for err in ut_check.psilist:
            with self.assertRaises(TypeError):
                mask_smoothing(np.ones((3,3)), binsize=err)

    def test_bin_percent(self):
        '''bad input'''
        for err in ut_check.rnslist:
            with self.assertRaises(TypeError):
                mask_smoothing(np.ones((3,3)), binsize=1, bin_percent=err)

    def test_bin_percent_value(self):
        '''bin_percent must be < 1.'''
        with self.assertRaises(ValueError):
            mask_smoothing(np.ones((3,3)), binsize=1, bin_percent=1)
        with self.assertRaises(ValueError):
            mask_smoothing(np.ones((3,3)), binsize=1, bin_percent=2)

    def test_smoothing(self):
        '''Run a simple mask through and get expected result. And the shape 
        (even or odd number of rows or columns) shouldn't matter.'''
        # bin a 10x10 mask into 2x2 bins.  If >60% is 1, then all the  
        # bin pixels become 1.  If <=60%, they all become 0.
        mask = np.zeros((11,11))
        exp_smooth = np.zeros((11,11))
        mask[0,0] = 1 # becomes 0
    
        mask[2:4, 2:3] = 1 # half a 2x2 is 1, so it all becomes 0

        # 75% of a 2x2 block is 1, so whole block becomes 1
        mask[4:6, 4:6] = 1
        mask[5,5] = 0  
        exp_smooth[4:6, 4:6] = 1

        # a 2x2 space that spans 2 bins: only 50% of each is 1, so both 
        # become all zeros
        mask[2:4, 7:9] = 1

        # since 11 not evenly divisible by 2, the last bin will include the 
        # remainder, 3x3.  2x2 pixels are 1, which is 44.4%, and so 
        # bin becomes zeros
        mask[8:10,8:10] = 1

        smooth = mask_smoothing(mask, binsize=2, bin_percent=0.6)
        self.assertTrue(np.array_equal(exp_smooth, smooth))

if __name__ == '__main__':
    unittest.main()