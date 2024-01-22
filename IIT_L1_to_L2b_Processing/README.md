# IIT_L1_to_L2b_Processing

L1 to L2 Data Processing for II&T - 

## l1-l2b_FFT Version 

This version processes all files of a data set that are located in the input_data folder. 

Outputs are saved in the output folder. For each processed L1 frame CGI_EXCAM_L1_XXXXXXi.fits of format 2200x1200 4 output cleaned files of format 1024x1024 will be saved:

- CGI_EXCAM_L1_XXXXXXi_L2a_image.fits

- CGI_EXCAM_L1_XXXXXXi_L2a_bpmap.fits                                

- CGI_EXCAM_L1_XXXXXXi_L2b_image.fits

- CGI_EXCAM_L1_XXXXXXi_L2b_bpmap.fits   

For each dataset, 4 output combined files will be saved: 

- CGI_EXCAM_L1_XXXXXX_mean_comb_image.fits

- CGI_EXCAM_L1_XXXXXX_mean_comb_bmap.fits

- CGI_EXCAM_L1_XXXXXX_median_comb_image.fits

- CGI_EXCAM_L1_XXXXXX_median_comb_bmap.fits


The output files are fits files with Primary HDU and Image HDU types headers. 

Input data for the pipeline must be placed in the `input_data` folder. All FITS files in this folder will be processed by the pipeline, so it is the user's responsibility to ensure that the input folder has only the files that are desired to be processed and combined. 


# Run the Pipeline

* ### Configuration files

Defaults detector parameters and calibration files paths are set up in config_file.yaml. Those defaults parameters and paths should be updated as needed. 

There's an option to generate png files of the outputs that is set to False by default. To set it to True, open the config_file.yaml file in the folder 'config files\' and replace 'False' by 'True'. png files will then be generated for each fits file present in the output folder.

fixed_pattern_noise: 1024x1024 array of floats. This is a per-pixel map of fixed-pattern noise in electrons. There are no constraints on value (may be positive or negative, similar to read noise). Default is a map of zeros.

dark_current_map: 1024x1024 array of floats. This is a per-pixel map of dark current noise in electrons per second. Each array element should be >= 0.

EXCAM_clock_induced_charge: 1024x1024 array of floats. This is a per-pixel map of EXCAM clock-induced charge in electrons. Each array element should be >= 0. Default is a map of zeros.

flatfield: 1024x1024 array of floats. Flat field array for pixel scaling of image section. Default is a map of ones.

badpix: 1024x1024 array of booleans. Bad pixel mask. Bad pixels are True. Default is a map of zeros.

non_linearity: txt file. See example file for format. Default is an array of ones.

eperdn : float. Electrons per dn conversion factor (detector k gain).

fwc_em_e : int. Full well capacity of detector EM gain register (electrons).

fwc_pp_e : int. Full well capacity of detector image area pixels (electrons).

bias_offset: float. Median number of counts in the bias region due to fixed non-bias noise not in common with the image region. Basically we compute the bias for the image region based on the prescan from each frame, and the bias_offset is how many additional counts the prescan had from e.g. fixed-pattern noise. This value is subtracted from each measured bias. Units of DNs.

There's an option to generate png files that is set to false per default. The PNG generation results in an overall slowdown of the processing pipeline.


* ### Input Data

Ensure that the `input_data` folder has been populated with the desired L1 files to be processed for this run of the pipeline. There should be no additional files in the `input_data` folder besides the input L1 EXCAM FITS files. 

An example L1 FITS file from CGI Full Functional Test in November 2023 is provided in the `input_data` directory. Note that an idiosyncrasy of FFT data is that the readout sequence is reported as `TRAP_PUMPING` even though the data themselves are not trap pumping calibration data. This will not be the case in flight.

* ### Run the pipeline

  `time python -m IIT_L1_to_L2b_Data_Processing`
  
  or simply 
  
  `python IIT_L1_to_L2b_Data_Processing`

* ### Visualize Data

  Outputs are saved in the output folder. 
  
  Examine using ds9 or your perferred .fits viewer, or open the .png versions for a quick glance.

* ### Post-test clean-up

  > :warning: **Warning**: This command will remove all data in the input folder as well as all saved files in the temp_data and output folders

  From the l1-l2b_FFT directory:

  `python ../run_post_test_cleanup.py`
  
  
  
## Authors

* Marie Ygouf (JPL)
* Nick Bowman (JPL)
