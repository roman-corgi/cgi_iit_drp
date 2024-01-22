#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ygouf
"""


from astropy.io import fits
import numpy as np
import os
import yaml

import time
import git

# %% User Defined Inputs

local_test = True

thisFolder = 'l1-l2b_FFT'   # Must match folder name

class_name = 'proc_cgi_frame.gsw_process.Process'

TDD_REL_PATH = 'IIT_L1_to_L2b_Processing/'  # Do not put "/" at front or os.path.join() won't work correctly.
GSW_ABS_PATH = '/Users/vbailey/Documents/code_internal/EXCAM_NTR/EXCAM_processing_NTR_scrubbed/'

VI_REL_PATH = os.path.join(TDD_REL_PATH, thisFolder)
TEMP_REL_PATH = os.path.join(VI_REL_PATH, 'temp_data/') 
TDD_ABS_PATH = os.path.join(GSW_ABS_PATH, TDD_REL_PATH)
VI_ABS_PATH = os.path.join(GSW_ABS_PATH, VI_REL_PATH)
TEMP_ABS_PATH = os.path.join(VI_ABS_PATH, 'temp_data')
INPUT_ABS_PATH = os.path.join(VI_ABS_PATH, 'input_data')
OUTPUT_ABS_PATH = os.path.join(VI_ABS_PATH, 'output')
CONFIG_ABS_PATH = os.path.join(VI_ABS_PATH, 'config_files')
CALIBRATION_ABS_PATH = os.path.join(VI_ABS_PATH, 'calibration_files')


# %% Functions

def check_input_files(dir_list,dir_path):
    """Check that all frames in the input_data folder are L1 data."""
    
    # Check that dir_list contains only L1 files    
    for filename in dir_list:
        hdul_input = fits.open(os.path.join(dir_path, filename))
        if hdul_input[1].header['HIERARCH DATA_LEVEL'] != 'L1':
            raise Exception("Please make sure all the files in the input folder are L1 data. You may check the data level in the Image HDU Header.")
  
    return      
    
def check_exptime(dir_list,dir_path):
    """Check that all frames in the input_data folder have the same exposure time."""

    i = 0  
    exptime_old = 0
    for filename in dir_list:
        print()
        
        hdul = fits.open(os.path.join(dir_path, filename))
        exptime = hdul[1].header['EXPTIME']

        # Throw error if pipeline is called on group of L1 frames that don’t have the same settings
        if i > 0 :
            if exptime != exptime_old: 
                raise ValueError("Different exposure time values detected in L1 frames from this data set. Please provide a uniform data set")
        i  = i + 1
        exptime_old = exptime
 
    return      
           
def check_gain(dir_list,dir_path):
    """Check that all frames in the input_data folder have the same gain."""
 
    i = 0
    em_gain_old = 0
    for filename in dir_list:
        
        hdul = fits.open(os.path.join(dir_path, filename))
        
        if 'EM_GAIN' in hdul[1].header:
            em_gain = hdul[1].header['EM_GAIN']
        if 'CMDGAIN' in hdul[1].header:  
            em_gain = hdul[1].header['CMDGAIN']

        # Throw error if pipeline is called on group of L1 frames that don’t have the same settings
        if i > 0 :
            if em_gain != em_gain_old: 
                raise ValueError("Different EM gain values detected in L1 frames from this data set. Please provide a uniform data set")
        i  = i + 1
        em_gain_old = em_gain
 
    return  

# Convert NumPy array to list
def convert_np_array(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError("Type not serializable")    


def gen_inputs_for_L1_to_L2a(dir_list,dir_path,config_abs_path):
    
    for filename in dir_list:
        print(filename)
        print('INPUT GENERATION: ITERATION ')  
        
        hdul = fits.open(os.path.join(dir_path, filename))
        
        # Extract frame size and other info from header/config file to use the appropriate metadata.yaml file        
        NAXIS2 = hdul[1].header['NAXIS2']
        if NAXIS2 == 1200: 
            filename_metadata = '/metadata.yaml'
        if NAXIS2 == 2200: 
            filename_metadata = '/metadata_eng.yaml'
        
        # Currently using the frame_rows paramaters to choose between SCIENCE and ENG modes but it would 
        # make more sense to use the OPSMODE parameters
        # OPSMODE Not supported for II&T
        # 1: ACQUIRE / SCIENCE 2: LOWFS / STRSH
        # 3: TRAP PUMPING
        # 4: ENGINEERING_EM
        # 5: ENGINEERING_CONV 6..14: Reserved
        # 0 or 15: None (state is DETON)
        
        if local_test == True:
            # meta_path = os.path.join(config_abs_path, filename_metadata)
            meta_path = config_abs_path+filename_metadata

        config_file_path = os.path.join(config_abs_path, 'config_file.yaml')

        with open(config_file_path, 'r') as stream:
            try:
                parsed_yaml=yaml.safe_load(stream)
                print(parsed_yaml)
            except yaml.YAMLError as exc:
                print(exc)
    
        badpix_filepath = parsed_yaml['badpix_filepath']
        dark_current_map_filepath = parsed_yaml['dark_current_map']  
        fixed_pattern_noise_filepath = parsed_yaml['fixed_pattern_noise'] 
        EXCAM_clock_induced_charge_filepath = parsed_yaml['EXCAM_clock_induced_charge']
        flatfield_filepath = parsed_yaml['flatfield_filepath']
        non_linearity_filepath = parsed_yaml['non_linearity_filepath']
    
        bad_pix = fits.getdata(os.path.join(VI_ABS_PATH, badpix_filepath))
        flat = fits.getdata(os.path.join(VI_ABS_PATH, flatfield_filepath))
        
        frames = hdul[1].data 

        if local_test == True:
            nonlin_path_ones = os.path.join(VI_ABS_PATH,non_linearity_filepath)
    
        nonlin_path = nonlin_path_ones

        with open(config_file_path, 'r') as stream:
            try:
                parsed_yaml=yaml.safe_load(stream)
                print(parsed_yaml)
            except yaml.YAMLError as exc:
                print(exc)
        
        fwc_em_e = parsed_yaml['fwc_em_e']
        fwc_pp_e = parsed_yaml['fwc_pp_e']
        bias_offset = parsed_yaml['bias_offset']
        eperdn = parsed_yaml['eperdn'] 

        if 'EM_GAIN' in hdul[1].header:
            em_gain = hdul[1].header['EM_GAIN']
        if 'CMDGAIN' in hdul[1].header:  
            em_gain = hdul[1].header['CMDGAIN']
        exptime = hdul[1].header['EXPTIME']
        meta_path = meta_path

        print('fwc_em_e ', fwc_em_e) 
        print('fwc_pp_e ', fwc_pp_e) 
        print('bias_offset ', bias_offset) 
        print('exptime ', exptime) 
        print('em_gain ', em_gain) 
        print('eperdn ', eperdn) 
        print('meta_path ', meta_path) 
     

        if local_test == True:
            D = fits.getdata(os.path.join(VI_ABS_PATH, dark_current_map_filepath))
            F = fits.getdata(os.path.join(VI_ABS_PATH, fixed_pattern_noise_filepath))
            C = fits.getdata(os.path.join(VI_ABS_PATH, EXCAM_clock_induced_charge_filepath))
            g = em_gain
            t = exptime
            
            from cal.masterdark.gsw_masterdark import build_dark
            dark = build_dark(F, D, C, g, t)
            
        # Save synthetic dark in output folder     
        fnOut = os.path.join(OUTPUT_ABS_PATH, 'synthetic_dark.fits')
        print(fnOut)
        hdu = fits.PrimaryHDU(dark)
        hdu.writeto(fnOut, overwrite=True)
    
        # Pickle the inputs and initialisation parameters
        filename_no_ext = os.path.splitext(filename)[0]
        filename_no_ext_init = 'init_'+filename_no_ext
    
        fits.PrimaryHDU(bad_pix).writeto(TEMP_ABS_PATH+'/'+filename_no_ext_init+'_'+'bad_pix.fits',overwrite=True)
        fits.PrimaryHDU([eperdn]).writeto(TEMP_ABS_PATH+'/'+filename_no_ext_init+'_'+'eperdn.fits',overwrite=True)
        fits.PrimaryHDU([fwc_em_e]).writeto(TEMP_ABS_PATH+'/'+filename_no_ext_init+'_'+'fwc_em_e.fits',overwrite=True)
        fits.PrimaryHDU([fwc_pp_e]).writeto(TEMP_ABS_PATH+'/'+filename_no_ext_init+'_'+'fwc_pp_e.fits',overwrite=True)
        fits.PrimaryHDU([bias_offset]).writeto(TEMP_ABS_PATH+'/'+filename_no_ext_init+'_'+'bias_offset.fits',overwrite=True)
        fits.PrimaryHDU(dark).writeto(TEMP_ABS_PATH+'/'+filename_no_ext_init+'_'+'dark.fits',overwrite=True)
        fits.PrimaryHDU(flat).writeto(TEMP_ABS_PATH+'/'+filename_no_ext_init+'_'+'flat.fits',overwrite=True)
        fits.PrimaryHDU([em_gain]).writeto(TEMP_ABS_PATH+'/'+filename_no_ext_init+'_'+'em_gain.fits',overwrite=True)
        fits.PrimaryHDU([exptime]).writeto(TEMP_ABS_PATH+'/'+filename_no_ext_init+'_'+'exptime.fits',overwrite=True)
        
        fits.PrimaryHDU(frames).writeto(TEMP_ABS_PATH+'/'+filename_no_ext+'_'+'frame_dn.fits',overwrite=True)
    
        with open(TEMP_ABS_PATH+'/'+filename_no_ext_init+'_'+'meta_path.txt', 'w') as file:
            file.write(meta_path)
        with open(TEMP_ABS_PATH+'/'+filename_no_ext_init+'_'+'nonlin_path.txt', 'w') as file:
            file.write(nonlin_path)
  

    return      

 
def run_gsw_L1_to_L2a(dir_list):
     
    start_time = time.time()       

    if local_test == True:   
        for filename in dir_list:
            # if testing locally
            ####################
            
            filename_no_ext = os.path.splitext(filename)[0]
            filename_no_ext_init = 'init_'+filename_no_ext

            bad_pix = fits.getdata(TEMP_ABS_PATH+'/'+filename_no_ext_init+'_'+'bad_pix.fits')
            eperdn = fits.getdata(TEMP_ABS_PATH+'/'+filename_no_ext_init+'_'+'eperdn.fits')[0] 
            fwc_em_e = fits.getdata(TEMP_ABS_PATH+'/'+filename_no_ext_init+'_'+'fwc_em_e.fits')[0]
            fwc_pp_e =  fits.getdata(TEMP_ABS_PATH+'/'+filename_no_ext_init+'_'+'fwc_pp_e.fits')[0]
            bias_offset = fits.getdata(TEMP_ABS_PATH+'/'+filename_no_ext_init+'_'+'bias_offset.fits')[0]
            dark = fits.getdata(TEMP_ABS_PATH+'/'+filename_no_ext_init+'_'+'dark.fits')
            flat = fits.getdata(TEMP_ABS_PATH+'/'+filename_no_ext_init+'_'+'flat.fits')
            em_gain = fits.getdata(TEMP_ABS_PATH+'/'+filename_no_ext_init+'_'+'em_gain.fits')[0]
            exptime = fits.getdata(TEMP_ABS_PATH+'/'+filename_no_ext_init+'_'+'exptime.fits')[0]
            
            frame_dn = fits.getdata(TEMP_ABS_PATH+'/'+filename_no_ext+'_'+'frame_dn.fits')

                
            with open(TEMP_ABS_PATH+'/'+filename_no_ext_init+'_'+'meta_path.txt', 'r') as file:
                meta_path = file.read()
            with open(TEMP_ABS_PATH+'/'+filename_no_ext_init+'_'+'nonlin_path.txt', 'r') as file:
                nonlin_path = file.read()

            
            from proc_cgi_frame.gsw_process import Process
            proc = Process(bad_pix, eperdn, fwc_em_e, fwc_pp_e,
                           bias_offset, em_gain, exptime,
                           nonlin_path,  meta_path=meta_path, dark=dark, flat=flat)
    
            (image, bpmap, image_r, bias,
        frame, bpmap_frame, frame_bias) = proc.L1_to_L2a(frame_dn)
                
            # Convert bool arrays to integer before saving to fits
            L2a_bpmap = np.multiply(bpmap, 1)
            L2a_bpmap_frame = np.multiply(bpmap_frame, 1)

            fits.PrimaryHDU(image).writeto(TEMP_ABS_PATH+'/'+filename_no_ext+'_'+'image_L2a.fits',overwrite=True)
            fits.PrimaryHDU(L2a_bpmap).writeto(TEMP_ABS_PATH+'/'+filename_no_ext+'_'+'bpmap_L2a.fits',overwrite=True)
            fits.PrimaryHDU(bias).writeto(TEMP_ABS_PATH+'/'+filename_no_ext+'_'+'bias_L2a.fits',overwrite=True)
            fits.PrimaryHDU(image_r).writeto(TEMP_ABS_PATH+'/'+filename_no_ext+'_'+'image_r_L2a.fits',overwrite=True)
            fits.PrimaryHDU(frame).writeto(TEMP_ABS_PATH+'/'+filename_no_ext+'_'+'frame_L2a.fits',overwrite=True)
            fits.PrimaryHDU(L2a_bpmap_frame).writeto(TEMP_ABS_PATH+'/'+filename_no_ext+'_'+'bpmap_frame_L2a.fits',overwrite=True)
            fits.PrimaryHDU(frame_bias).writeto(TEMP_ABS_PATH+'/'+filename_no_ext+'_'+'frame_bias_L2a.fits',overwrite=True)


    print("L1 to L2a")             
    print("--- %s seconds ---" % (time.time() - start_time))

    return      


def run_gsw_L2a_to_L2b(dir_list):
     
    start_time = time.time() 
    
    if local_test == True:   
        # if testing locally
        ####################
        for filename in dir_list:
            
            filename_no_ext = os.path.splitext(filename)[0]
            filename_no_ext_init = 'init_'+filename_no_ext
            
            bad_pix = fits.getdata(TEMP_ABS_PATH+'/'+filename_no_ext_init+'_'+'bad_pix.fits')
            eperdn = fits.getdata(TEMP_ABS_PATH+'/'+filename_no_ext_init+'_'+'eperdn.fits')[0] 
            fwc_em_e = fits.getdata(TEMP_ABS_PATH+'/'+filename_no_ext_init+'_'+'fwc_em_e.fits')[0]
            fwc_pp_e =  fits.getdata(TEMP_ABS_PATH+'/'+filename_no_ext_init+'_'+'fwc_pp_e.fits')[0]
            bias_offset = fits.getdata(TEMP_ABS_PATH+'/'+filename_no_ext_init+'_'+'bias_offset.fits')[0]
            dark = fits.getdata(TEMP_ABS_PATH+'/'+filename_no_ext_init+'_'+'dark.fits')
            flat = fits.getdata(TEMP_ABS_PATH+'/'+filename_no_ext_init+'_'+'flat.fits')
            em_gain = fits.getdata(TEMP_ABS_PATH+'/'+filename_no_ext_init+'_'+'em_gain.fits')[0]
            exptime = fits.getdata(TEMP_ABS_PATH+'/'+filename_no_ext_init+'_'+'exptime.fits')[0]
            
            # frame_dn = fits.getdata(TEMP_ABS_PATH+'/'+filename_no_ext+'/'+'frame_dn.fits')

                
            with open(TEMP_ABS_PATH+'/'+filename_no_ext_init+'_'+'meta_path.txt', 'r') as file:
                meta_path = file.read()
            with open(TEMP_ABS_PATH+'/'+filename_no_ext_init+'_'+'nonlin_path.txt', 'r') as file:
                nonlin_path = file.read()

    
            from proc_cgi_frame.gsw_process import Process
            proc = Process(bad_pix, eperdn, fwc_em_e, fwc_pp_e,
                            bias_offset, em_gain, exptime,
                            nonlin_path,  meta_path=meta_path, dark=dark, flat=flat)
                                            
            i0 = fits.getdata(TEMP_ABS_PATH+'/'+filename_no_ext+'_'+'image_L2a.fits')
            b0 = fits.getdata(TEMP_ABS_PATH+'/'+filename_no_ext+'_'+'bpmap_L2a.fits')
        
            L2b_image, L2b_bpmap, L2b_image_r = proc.L2a_to_L2b(i0, b0)
            
            fits.PrimaryHDU(L2b_image).writeto(TEMP_ABS_PATH+'/'+filename_no_ext+'_'+'image_L2b.fits',overwrite=True)
            fits.PrimaryHDU(L2b_bpmap).writeto(TEMP_ABS_PATH+'/'+filename_no_ext+'_'+'bpmap_L2b.fits',overwrite=True)
            fits.PrimaryHDU(L2b_image_r).writeto(TEMP_ABS_PATH+'/'+filename_no_ext+'_'+'image_r_L2b.fits',overwrite=True)
            

    print("L2a to L2b")             
    print("--- %s seconds ---" % (time.time() - start_time))

    return      


def gen_inputs_for_mean_and_median_combine(dir_list):
    """Generate inputs for test."""
    print("Generating inputs for median and mean combination...")

    image_list = []
    bad_mask_list = []
        
    for filename in dir_list:
        print(filename)
        print('INPUT GENERATION: ITERATION ')
            
        filename_no_ext = os.path.splitext(filename)[0]
     
        image = fits.getdata(TEMP_ABS_PATH+'/'+filename_no_ext+'_'+'image_L2b.fits')
        bad_mask = fits.getdata(TEMP_ABS_PATH+'/'+filename_no_ext+'_'+'bpmap_L2b.fits')
        
        image_list.append(image)
        bad_mask_list.append(bad_mask)
           
    fits.PrimaryHDU(image_list).writeto(TEMP_ABS_PATH+'/'+'image_L2b_list.fits',overwrite=True)
    fits.PrimaryHDU(bad_mask_list).writeto(TEMP_ABS_PATH+'/'+'bpmap_L2b_list.fits',overwrite=True)
            
    return      


def run_gsw_mean_combine():
    """Run the CTC GSW in the docker image."""
    print("Running CTC GSW in a docker image...")
      
    start_time = time.time() 
    
    if local_test == True:   
        # if testing locally
        ####################
        check_ims = fits.getdata(TEMP_ABS_PATH+'/'+'image_L2b_list.fits')
        check_masks = fits.getdata(TEMP_ABS_PATH+'/'+'bpmap_L2b_list.fits')
        
        check_masks = check_masks.astype(int)
        
        from proc_cgi_frame.gsw_process import mean_combine

        comb_image, comb_bpmap, mean_num_good_fr, enough_for_rn = mean_combine(check_ims, check_masks)
                
        comb_bpmap = np.multiply(comb_bpmap, 1)
        
        fits.PrimaryHDU(comb_image).writeto(TEMP_ABS_PATH+'/'+'mean_combine_image.fits',overwrite=True)
        fits.PrimaryHDU(comb_bpmap).writeto(TEMP_ABS_PATH+'/'+'mean_combine_bpmap.fits',overwrite=True)
                
    print("Mean Combine")             
    print("--- %s seconds ---" % (time.time() - start_time))

    return      


def run_gsw_median_combine():
    
    start_time = time.time() 
    
 
    if local_test == True:   
        # if testing locally
        ####################
                             
        from proc_cgi_frame.gsw_process import median_combine


        check_ims = fits.getdata(TEMP_ABS_PATH+'/'+'image_L2b_list.fits')
        check_masks = fits.getdata(TEMP_ABS_PATH+'/'+'bpmap_L2b_list.fits')
        
        check_masks = check_masks.astype(int)
        
        comb_image, comb_bpmap = median_combine(check_ims, check_masks)
        
        comb_bpmap = np.multiply(comb_bpmap, 1)
               
        fits.PrimaryHDU(comb_image).writeto(TEMP_ABS_PATH+'/'+'median_combine_image.fits',overwrite=True)
        fits.PrimaryHDU(comb_bpmap).writeto(TEMP_ABS_PATH+'/'+'median_combine_bpmap.fits',overwrite=True)

    print("Median Combine")             
    print("--- %s seconds ---" % (time.time() - start_time))
  
    return      
  
    
def save_outputs(dir_list,config_abs_path):
    """Load and save outputs as fits files."""      

    # git version   
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    print('Git version')
    print(sha)
    
    for filename in dir_list:
            
        filename_no_ext = os.path.splitext(filename)[0]
        filename_no_ext_init = 'init_'+filename_no_ext 
           
        L2a_image = fits.getdata(TEMP_ABS_PATH+'/'+filename_no_ext+'_'+'image_L2a.fits')
        L2a_bpmap = fits.getdata(TEMP_ABS_PATH+'/'+filename_no_ext+'_'+'bpmap_L2a.fits')
        L2b_image = fits.getdata(TEMP_ABS_PATH+'/'+filename_no_ext+'_'+'image_L2b.fits')
        L2b_bpmap = fits.getdata(TEMP_ABS_PATH+'/'+filename_no_ext+'_'+'bpmap_L2b.fits')
        fwc_em_e = fits.getdata(TEMP_ABS_PATH+'/'+filename_no_ext_init+'_'+'fwc_em_e.fits')[0]
        fwc_pp_e = fits.getdata(TEMP_ABS_PATH+'/'+filename_no_ext_init+'_'+'fwc_pp_e.fits')[0]
        bias_offset = fits.getdata(TEMP_ABS_PATH+'/'+filename_no_ext_init+'_'+'bias_offset.fits')[0]
        eperdn = fits.getdata(TEMP_ABS_PATH+'/'+filename_no_ext_init+'_'+'eperdn.fits')[0]  
    
        config_file_path = os.path.join(config_abs_path, 'config_file.yaml')

        with open(config_file_path, 'r') as stream:
            try:
                parsed_yaml=yaml.safe_load(stream)
                print(parsed_yaml)
            except yaml.YAMLError as exc:
                print(exc)
                
        badpix_filepath = parsed_yaml['badpix_filepath']
        dark_current_map_filepath = parsed_yaml['dark_current_map']  
        fixed_pattern_noise_filepath = parsed_yaml['fixed_pattern_noise'] 
        EXCAM_clock_induced_charge_filepath = parsed_yaml['EXCAM_clock_induced_charge']
        flatfield_filepath = parsed_yaml['flatfield_filepath']
        non_linearity_filepath = parsed_yaml['non_linearity_filepath']

        # Convert bool arrays to integer before saving to fits
        L2a_bpmap = np.multiply(L2a_bpmap, 1)
        L2b_bpmap = np.multiply(L2b_bpmap, 1)
               
        hdul_input = fits.open(os.path.join(INPUT_ABS_PATH, filename))
    
        fnOut = os.path.join(OUTPUT_ABS_PATH, filename_no_ext+'_L2a_image.fits')         
        hdul = fits.HDUList(hdul_input)
        hdul[1] = fits.ImageHDU(L2a_image)
        hdul[1].header = hdul_input[1].header
        hdul[1].header['DATA_LEVEL'] = 'L2a'
        hdul[1].header['MEDIAN_COMBINED'] = False
        hdul[1].header['MEAN_COMBINED'] = False
        hdul[1].header['DESMEARED'] = False
        hdul[1].header['HIERARCH FWC_EM_E'] = fwc_em_e
        hdul[1].header['HIERARCH FWC_PP_E'] = fwc_pp_e
        hdul[1].header['HIERARCH EPERDN'] = eperdn
        hdul[1].header['BIAS_OFFSET'] = bias_offset
        hdul[1].header['BADPIX_PATH'] = badpix_filepath
        hdul[1].header['NONLIN_PATH'] = non_linearity_filepath       
        hdul[1].header['DARK_C_PATH'] = dark_current_map_filepath
        hdul[1].header['F_NOISE_PATH'] = fixed_pattern_noise_filepath
        hdul[1].header['HIERARCH CIC_PATH'] = EXCAM_clock_induced_charge_filepath
        hdul[1].header['FLAT_PATH'] = flatfield_filepath
        hdul[1].header['GIT_COMMIT'] = sha
        if 'BZERO' in hdul[1].header:
            hdul[1].header['BZERO'] = 0
        hdul.writeto(fnOut, overwrite=True)
        hdul.close()

        fnOut = os.path.join(OUTPUT_ABS_PATH, filename_no_ext+'_L2a_bpmap.fits')
        hdul = fits.HDUList(hdul_input)
        hdul[1] = fits.ImageHDU(L2a_bpmap)
        hdul[1].header = hdul_input[1].header
        hdul[1].header['DATA_LEVEL'] = 'L2a'
        hdul[1].header['MEDIAN_COMBINED'] = False
        hdul[1].header['MEAN_COMBINED'] = False
        hdul[1].header['DESMEARED'] = False
        hdul[1].header['HIERARCH FWC_EM_E'] = fwc_em_e
        hdul[1].header['HIERARCH FWC_PP_E'] = fwc_pp_e
        hdul[1].header['HIERARCH EPERDN'] = eperdn
        hdul[1].header['BIAS_OFFSET'] = bias_offset
        hdul[1].header['BADPIX_PATH'] = badpix_filepath
        hdul[1].header['NONLIN_PATH'] = non_linearity_filepath 
        hdul[1].header['DARK_C_PATH'] = dark_current_map_filepath
        hdul[1].header['F_NOISE_PATH'] = fixed_pattern_noise_filepath
        hdul[1].header['HIERARCH CIC_PATH'] = EXCAM_clock_induced_charge_filepath
        hdul[1].header['FLAT_PATH'] = flatfield_filepath
        hdul[1].header['GIT_COMMIT'] = sha
        if 'BZERO' in hdul[1].header:
            hdul[1].header['BZERO'] = 0
        hdul.writeto(fnOut, overwrite=True)
        hdul.close()
        
        fnOut = os.path.join(OUTPUT_ABS_PATH, filename_no_ext+'_L2b_image.fits')
        hdul = fits.HDUList(hdul_input)
        hdul[1] = fits.ImageHDU(L2b_image)
        hdul[1].header = hdul_input[1].header
        hdul[1].header['DATA_LEVEL'] = 'L2b'
        hdul[1].header['MEDIAN_COMBINED'] = False
        hdul[1].header['MEAN_COMBINED'] = False
        hdul[1].header['DESMEARED'] = False
        hdul[1].header['HIERARCH FWC_EM_E'] = fwc_em_e
        hdul[1].header['HIERARCH FWC_PP_E'] = fwc_pp_e
        hdul[1].header['HIERARCH EPERDN'] = eperdn
        hdul[1].header['BIAS_OFFSET'] = bias_offset
        hdul[1].header['BADPIX_PATH'] = badpix_filepath
        hdul[1].header['NONLIN_PATH'] = non_linearity_filepath 
        hdul[1].header['DARK_C_PATH'] = dark_current_map_filepath
        hdul[1].header['F_NOISE_PATH'] = fixed_pattern_noise_filepath
        hdul[1].header['HIERARCH CIC_PATH'] = EXCAM_clock_induced_charge_filepath
        hdul[1].header['FLAT_PATH'] = flatfield_filepath
        hdul[1].header['GIT_COMMIT'] = sha
        if 'BZERO' in hdul[1].header:
            hdul[1].header['BZERO'] = 0
        hdul.writeto(fnOut, overwrite=True)
        hdul.close()

        fnOut = os.path.join(OUTPUT_ABS_PATH, filename_no_ext+'_L2b_bpmap.fits')
        hdul = fits.HDUList(hdul_input)
        hdul[1] = fits.ImageHDU(L2b_bpmap)
        hdul[1].header = hdul_input[1].header
        hdul[1].header['DATA_LEVEL'] = 'L2b'
        hdul[1].header['MEDIAN_COMBINED'] = False
        hdul[1].header['MEAN_COMBINED'] = False
        hdul[1].header['DESMEARED'] = False
        hdul[1].header['HIERARCH FWC_EM_E'] = fwc_em_e
        hdul[1].header['HIERARCH FWC_PP_E'] = fwc_pp_e
        hdul[1].header['HIERARCH EPERDN'] = eperdn
        hdul[1].header['BIAS_OFFSET'] = bias_offset
        hdul[1].header['BADPIX_PATH'] = badpix_filepath
        hdul[1].header['NONLIN_PATH'] = non_linearity_filepath 
        hdul[1].header['DARK_C_PATH'] = dark_current_map_filepath
        hdul[1].header['F_NOISE_PATH'] = fixed_pattern_noise_filepath
        hdul[1].header['HIERARCH CIC_PATH'] = EXCAM_clock_induced_charge_filepath
        hdul[1].header['FLAT_PATH'] = flatfield_filepath
        hdul[1].header['GIT_COMMIT'] = sha
        if 'BZERO' in hdul[1].header:
            hdul[1].header['BZERO'] = 0
        hdul.writeto(fnOut, overwrite=True)
        hdul.close()

    if local_test == True:        
        comb_image3 = fits.getdata(TEMP_ABS_PATH+'/'+'mean_combine_image.fits')
        comb_bpmap3 = fits.getdata(TEMP_ABS_PATH+'/'+'mean_combine_bpmap.fits')
        comb_image4 = fits.getdata(TEMP_ABS_PATH+'/'+'median_combine_image.fits')
        comb_bpmap4 = fits.getdata(TEMP_ABS_PATH+'/'+'median_combine_bpmap.fits')
                

    fnOut = os.path.join(OUTPUT_ABS_PATH, 'mean_comb_image.fits')       
    hdul = fits.HDUList(hdul_input)
    hdul[1] = fits.ImageHDU(comb_image3)
    hdul[1].header = hdul_input[1].header
    hdul[1].header['DATA_LEVEL'] = 'L2b'
    hdul[1].header['MEDIAN_COMBINED'] = False
    hdul[1].header['MEAN_COMBINED'] = True
    hdul[1].header['DESMEARED'] = False
    hdul[1].header['HIERARCH FWC_EM_E'] = fwc_em_e
    hdul[1].header['HIERARCH FWC_PP_E'] = fwc_pp_e
    hdul[1].header['HIERARCH EPERDN'] = eperdn
    hdul[1].header['BIAS_OFFSET'] = bias_offset
    hdul[1].header['BADPIX_PATH'] = badpix_filepath
    hdul[1].header['NONLIN_PATH'] = non_linearity_filepath 
    hdul[1].header['DARK_C_PATH'] = dark_current_map_filepath
    hdul[1].header['F_NOISE_PATH'] = fixed_pattern_noise_filepath
    hdul[1].header['HIERARCH CIC_PATH'] = EXCAM_clock_induced_charge_filepath
    hdul[1].header['FLAT_PATH'] = flatfield_filepath
    hdul[1].header['GIT_COMMIT'] = sha
    hdul.writeto(fnOut, overwrite=True)
    hdul.close()
    
    fnOut = os.path.join(OUTPUT_ABS_PATH, 'mean_comb_bpmap.fits')
    hdul = fits.HDUList(hdul_input)
    hdul[1] = fits.ImageHDU(comb_bpmap3.astype(float))
    hdul[1].header = hdul_input[1].header
    hdul[1].header['DATA_LEVEL'] = 'L2b'
    hdul[1].header['MEDIAN_COMBINED'] = False
    hdul[1].header['MEAN_COMBINED'] = True
    hdul[1].header['DESMEARED'] = False
    hdul[1].header['HIERARCH FWC_EM_E'] = fwc_em_e
    hdul[1].header['HIERARCH FWC_PP_E'] = fwc_pp_e
    hdul[1].header['HIERARCH EPERDN'] = eperdn
    hdul[1].header['BIAS_OFFSET'] = bias_offset
    hdul[1].header['BADPIX_PATH'] = badpix_filepath
    hdul[1].header['NONLIN_PATH'] = non_linearity_filepath       
    hdul[1].header['DARK_C_PATH'] = dark_current_map_filepath
    hdul[1].header['F_NOISE_PATH'] = fixed_pattern_noise_filepath
    hdul[1].header['HIERARCH CIC_PATH'] = EXCAM_clock_induced_charge_filepath
    hdul[1].header['FLAT_PATH'] = flatfield_filepath
    hdul[1].header['GIT_COMMIT'] = sha
    hdul.writeto(fnOut, overwrite=True)  
    hdul.close()       
    
    fnOut = os.path.join(OUTPUT_ABS_PATH, 'median_comb_image.fits')       
    hdul = fits.HDUList(hdul_input)
    hdul[1] = fits.ImageHDU(comb_image4)
    hdul[1].header = hdul_input[1].header
    hdul[1].header['DATA_LEVEL'] = 'L2b'
    hdul[1].header['MEDIAN_COMBINED'] = True
    hdul[1].header['MEAN_COMBINED'] = False
    hdul[1].header['DESMEARED'] = False
    hdul[1].header['HIERARCH FWC_EM_E'] = fwc_em_e
    hdul[1].header['HIERARCH FWC_PP_E'] = fwc_pp_e
    hdul[1].header['HIERARCH EPERDN'] = eperdn
    hdul[1].header['BIAS_OFFSET'] = bias_offset
    hdul[1].header['BADPIX_PATH'] = badpix_filepath
    hdul[1].header['NONLIN_PATH'] = non_linearity_filepath     
    hdul[1].header['DARK_C_PATH'] = dark_current_map_filepath
    hdul[1].header['F_NOISE_PATH'] = fixed_pattern_noise_filepath
    hdul[1].header['HIERARCH CIC_PATH'] = EXCAM_clock_induced_charge_filepath
    hdul[1].header['FLAT_PATH'] = flatfield_filepath
    hdul[1].header['GIT_COMMIT'] = sha
    hdul.writeto(fnOut, overwrite=True)
    hdul.close()
    
    fnOut = os.path.join(OUTPUT_ABS_PATH, 'median_comb_bpmap.fits')
    hdul = fits.HDUList(hdul_input)
    hdul[1] = fits.ImageHDU(comb_bpmap4.astype(float))
    hdul[1].header = hdul_input[1].header
    hdul[1].header['DATA_LEVEL'] = 'L2b'
    hdul[1].header['MEDIAN_COMBINED'] = True
    hdul[1].header['MEAN_COMBINED'] = False
    hdul[1].header['DESMEARED'] = False
    hdul[1].header['FWC_EM_E'] = fwc_em_e
    hdul[1].header['HIERARCH FWC_EM_E'] = fwc_em_e
    hdul[1].header['HIERARCH FWC_PP_E'] = fwc_pp_e
    hdul[1].header['HIERARCH EPERDN'] = eperdn
    hdul[1].header['BIAS_OFFSET'] = bias_offset
    hdul[1].header['BADPIX_PATH'] = badpix_filepath
    hdul[1].header['NONLIN_PATH'] = non_linearity_filepath       
    hdul[1].header['DARK_C_PATH'] = dark_current_map_filepath
    hdul[1].header['F_NOISE_PATH'] = fixed_pattern_noise_filepath
    hdul[1].header['HIERARCH CIC_PATH'] = EXCAM_clock_induced_charge_filepath
    hdul[1].header['FLAT_PATH'] = flatfield_filepath
    hdul[1].header['GIT_COMMIT'] = sha
    hdul.writeto(fnOut, overwrite=True)
    hdul.close()
    
        
    # Convert all fits in TEMP_DATA into png files
    ##############################################    
    config_file_path = os.path.join(CONFIG_ABS_PATH, 'config_file.yaml')
    with open(config_file_path, 'r') as stream:
            try:
                parsed_yaml=yaml.safe_load(stream)
                print(parsed_yaml)
            except yaml.YAMLError as exc:
                print(exc)
    

    return      


if __name__ == '__main__':

    start_time = time.time()    
    # Get the list of all files in the INPUT_ABS_PATH directory
    dir_list = os.listdir(INPUT_ABS_PATH)
    if '.DS_Store' in dir_list:
        dir_list.remove('.DS_Store')
    print("Files and directories in '", INPUT_ABS_PATH, "' :")
    print(dir_list)
    DIR_LIST = dir_list

    # Check that dir_list contains only L1 files            
    check_input_files(dir_list,INPUT_ABS_PATH)
    # Throw error if pipeline is called on group of frames that 
    # don’t have the same settings
    check_exptime(dir_list,INPUT_ABS_PATH)
    check_gain(dir_list,INPUT_ABS_PATH)
    
    gen_inputs_for_L1_to_L2a(dir_list,INPUT_ABS_PATH,CONFIG_ABS_PATH)  
    run_gsw_L1_to_L2a(dir_list)
    run_gsw_L2a_to_L2b(dir_list)
    gen_inputs_for_mean_and_median_combine(dir_list)
    run_gsw_mean_combine()
    run_gsw_median_combine()
    save_outputs(dir_list,CONFIG_ABS_PATH)
    print("--- %s seconds ---" % (time.time() - start_time))












