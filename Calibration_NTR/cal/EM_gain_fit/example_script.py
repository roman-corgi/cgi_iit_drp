'''Example script showing how to use the EM_gain_fit module.'''
import os
from pathlib import Path
from EM_gain_fitting import EMGainFit

here = os.path.abspath(os.path.dirname(__file__))

# NOTE See doc strings for the class EM_gain_fitting.EMGainFit for details.
# See the doc strings of EM_gain_fitting.read_in_excam_files and
# EM_gain_fitting.read_in_locam_files for details on how
# the frame files should be structured in their directories.
# EM_gain_fitting.EM_gain_fit uses EM_gain_tools._EM_gain_fit_conv to do the
# actual fitting algorithm.  See that function for more details on the inputs
# and methodology.  EM_gain_fit_params.yaml also has details on the input
# parameters.

############ This is for EXCAM high-gain.  Dark frames needed.
# Real data used below.
directory = Path(here, 'data', 'excam_high',
                 'n85 darks 1 s eperdn_8 cal frames')
# this data has k gain of 8 e-/DN, commanded gain of 5000, exposure time of 1s
# no knowledge of bias offset, so we leave it at 0
emg = EMGainFit(cam_mode='excam_high', eperdn=8, bias_offset=0, com_gain=5000,
                exptime=1)

# process the 5 darks for analysis; for 'excam_high', darks are used instead of 
# illuminated images that have to be masked to isolate the relevant illuminated 
# region, so mask_stack (which includes the mask used for each frame in the 
# directory) is a stack of frames filled with zeros (i.e., no masking).
frames, mask_stack = emg.read_in_excam_files(directory)

# perform the fit for EM gain
EMgain, e_mean, success, lam_b_g, gain_b_g = emg.EM_gain_fit(frames)
print(EMgain, e_mean, success)

############ This is for EXCAM low-gain.  Dark and illuminated frames needed.
# Simulated data below.

#non-unity gain
directory = Path(here, 'data', 'excam_low',
            'simulated','brights_G')
# unity gain
directoryG1 = Path(here, 'data', 'excam_low',
            'simulated','brights_G1')

# this data has k gain of 7 e-/DN, commanded gain of 40, exposure time of 1s,
# bias offset of 0
emg = EMGainFit(cam_mode='excam_low', eperdn=7, bias_offset=0, com_gain=40,
                exptime=1)

# process the 5 non-unity gain frames for analysis.  mask_stack is generated 
# for 'excam_low' with amplitude thresholding unless the input 
# do_ampthresh=False.  These are not pupil images or images illuminated on 
# only part of the image area, so we choose do_ampthresh=False. 
# These are illuminated frames (high above
# background), so we choose to apply desmearing.
frames, mask_stack = emg.read_in_excam_files(directory, desmear_flag=True,
                                             do_ampthresh=False)
# process the 5 unity gain frames for analysis.  You can use your own input 
# mask as well.  If these were partially illuminated frames, it would be a 
# good idea to use the same mask for both non-unity gain frames and 
# unity gain frames (in this case, though, the mask is simply a 
# frame of zeros):
framesG1, mask_stack1 = emg.read_in_excam_files(directoryG1, 
                                            mask=mask_stack[0].astype(bool))

# perform the fit for EM gain
EMgain, e_mean, success, lam_b_g, gain_b_g = emg.EM_gain_fit(frames, framesG1)
print(EMgain, e_mean, success)

# Even if success = False, one can get a simpler estimate of the EM gain:
EM_gain_est = frames.mean()/framesG1.mean()
# If EMgain is close to EM_gain_est, then the optimization result EMgain may
# be trustworthy.

# Here's an example of a pupil image that requires amplitude thresholding.
directory_pupil = Path(here, 'data', 'excam_low_pupil')
# We will demonstrate the masking without making a new class instance for this 
# different data.  # A good strategy is to get a mask from a good-contrast 
# pupil image to get 
# the best-quality mask and apply that mask to other images via the 'mask' 
# input parameter.
# For the pupil image, the relevant illuminated area has a 
# radius of about 150 pixels, and specifying a bigger ampthresh_mask_thresh 
# results in a better mask.  See doc string forEMGainFit class for 
# more options.
emg.ampthresh_mask_thresh = 3*150**2

frames, mask_stack = emg.read_in_excam_files(directory_pupil, 
                                             desmear_flag=True)

############ This is for LOCAM.  Illuminated and bias frames needed at
# non-unity gain.  If illuminated and bias frames available for unity gain,
# those can be read in, too.
# Simulated data used here.
directory = Path(here, 'data', 'locam', 'simulated', 'brights_G')
bias_directory = Path(here, 'data', 'locam', 'simulated', 'bias_G')

# this data has a k gain of 7 e-/DN, EM gain of 20, mean electron counts
# (fluxe) of about 1000, each frame consisting of 10000 summed frames.  Bias
# offset is not meaningful for LOCAM, and when cam_mode='locam', it is not
# used.  And the exposure time is fixed and comes from the config_dict, so we
# can use None for these two inputs (although it doesn't matter what we put for
# them).  
# These simulated images are not pupil images, so no need to do 
# amplitude thresholding. 
emg = EMGainFit(cam_mode='locam', eperdn=7, bias_offset=None, com_gain=20,
                exptime=None)

# process the non-unity gain summed frames for analysis
frames, mask_stack = emg.read_in_locam_files(directory, bias_directory,
                                 com_gain=emg.com_gain,  do_ampthresh=False)

# perform the fit for EM gain
EMgain, e_mean, success, lam_b_g, gain_b_g = emg.EM_gain_fit(frames)
print(EMgain, e_mean, success)

# If unity gain summed frames were available, one could process them with
# read_in_locam_files() using the input com_gain=1. And one could do a
# simple estimate for the EM gain as was shown above in the EXCAM low-gain
# case to check the trustworthiness of a result with success = False.

# In addition, the class allows for an input master dark and flat if desired.

# The class object emg has many attributes.  dir(emg) will show them all.

# If you needed to change, for example, the read noise and maximum EM gain
# without having to change a .yaml file:
emg.rn = 100
emg.config_dict['gmax'] = 6000