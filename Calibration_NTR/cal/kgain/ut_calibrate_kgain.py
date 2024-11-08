"""
Unit test suite for the calibrate_kgain module.

Frames for unit tests are simulated SCI-size frames made with Gaussian and Poisson 
noises included. The assumed flux map is a realistic pupil image made from 
TVAC frames.
"""

import os
import pandas as pd
from pathlib import Path
import unittest
import warnings
import numpy as np
from astropy.io import fits

from cal.util.loadyaml import loadyaml
import cal.util.ut_check as ut_check
from .calibrate_kgain import (calibrate_kgain, CalKgainException)

######################## function definitions ###############################

def nonlin_coefs(filename,EMgain,order):
    # filename is the name of the csv text file containing the TVAC nonlin table
    # EM gain selects the closest column in the table
    # Load the specified file
    bigArray = pd.read_csv(filename, header=None).values
    EMgains = bigArray[0, 1:]
    DNs = bigArray[1:, 0]
    
    # Find the closest EM gain available to what was requested
    iG = (np.abs(EMgains - EMgain)).argmin()
    
    # Fit the nonlinearity numbers to a polynomial
    vals = bigArray[1:, iG + 1]
    coeffs = np.polyfit(DNs, vals, order)
    
    # shift so that function passes through unity at 3000 DN for these tests
    fitVals0 = np.polyval(coeffs, DNs)
    ind = np.where(DNs == 3000)
    unity_val = fitVals0[ind][0]
    coeffs[3] = coeffs[3] - (unity_val-1.0)
    fitVals = np.polyval(coeffs,DNs)
    
    return coeffs, DNs, fitVals

def nonlin_factor(coeffs,DN):
    # input ceoffs from nonlin_ceofs and a DN value and return the 
    # nonlinearity factor
    min_value = 800.0
    max_value = 10000.0
    f_nonlin = np.polyval(coeffs, DN)
    # Control values outside the min/max range
    f_nonlin = np.where(DN < min_value, np.polyval(coeffs, min_value), f_nonlin)
    f_nonlin = np.where(DN > max_value, np.polyval(coeffs, max_value), f_nonlin)
    
    return f_nonlin

def make_frame(f_map, bias, kgain, rn, emgain, time, coeffs, nonlin_flag):
    # makes a SCI-sized frame with simulated noise and a fluxmap
    # f_map is the fluxmap in e/s/px and is 1024x1024 pixels in size
    # rn is read noise in electrons
    # bias is in electrons
    # time is exposure time in sec
    # coeffs is the array of cubic polynomial coefficients from nonlin_coefs
    # if nonlin_flag is True, then nonlinearity is applied
    
    # Generate random values of rn in elecrons from a Gaussian distribution
    random_array = np.random.normal(0, rn, (1200, 2200)) # e-
    # Generate random values from fluxmap from a Poisson distribution
    Poiss_noise_arr = emgain*np.random.poisson(time*f_map) # e-
    signal_arr = np.zeros((1200,2200))
    start_row = 10
    start_col = 1100
    signal_arr[start_row:start_row + Poiss_noise_arr.shape[0], 
                start_col:start_col + Poiss_noise_arr.shape[1]] = Poiss_noise_arr
    temp = random_array + signal_arr # e-
    if nonlin_flag:
        temp2 = nonlin_factor(coeffs, signal_arr/kgain)
        frame = np.round((bias + random_array + signal_arr/temp2)/kgain) # DN
    else:    
        frame = np.round((bias+temp)/kgain) # DN
        
    return frame

def count_contiguous_repeats(arr):
    if isinstance(arr, (np.ndarray, list)) and len(arr) == 0:
        return []
    counts = []
    current_count = 1
    for i in range(1, len(arr)):
        if arr[i] == arr[i - 1]:
            current_count += 1
        else:
            counts.append(current_count)
            current_count = 1
    counts.append(current_count)  # append the count of the last element
    return counts

############### make stacks with simulated frames ##################

# path to nonlin table made from running calibrate_nonlin.py on TVAC frames
# table used only to choose parameters to make analytic nonlin functions
here = os.path.abspath(os.path.dirname(__file__))
nonlin_table_path = Path(here,'ut_data','nonlin_table_TVAC.txt')
nonlin_flag = False # True adds nonlinearity to simulated frames

# Load the arrays needed for calibrate_nonlin function from the .npz file
loaded = np.load(Path(here,'ut_data','nonlin_arrays_ut.npz'))
# Access the arrays needed for calibrate_nonlin function
exp_time_stack_arr0 = loaded['array1']
len_list0 = loaded['array3']

# Load the flux map
hdul =  fits.open(Path(here,'ut_data','FluxMap1024.fits'))
fluxmap_init = hdul[0].data
hdul.close()
fluxmap_init[fluxmap_init < 50] = 0 # cleanup flux map a bit
fluxMap = 0.8*fluxmap_init # e/s/px, for G = 1

# assumed detector parameters
kgain_in = 8.7 # e-/DN
rn_in = 130 # read noise in e-
bias = 2000 # e-
emgain = 1.0

# cubic function nonlinearity for emgain of 1
if nonlin_flag:
    coeffs_1, DNs, _ = nonlin_coefs(nonlin_table_path,1.0,3)
else:
    coeffs_1 = [0.0, 0.0, 0.0, 1.0]
    _, DNs, _ = nonlin_coefs(nonlin_table_path,1.0,3)

frame_list2 = []
# make 30 uniform frames with emgain = 1
for j in range(30):
    frame2 = make_frame(fluxMap,bias,kgain_in,rn_in,emgain,7.0,coeffs_1,nonlin_flag)
    frame_list2.append(frame2)
stack_arr2 = np.stack(frame_list2)

index = 0
iG = 0 # doing only the em gain = 1 case
g = emgain
exp_time_loop = exp_time_stack_arr0[index:index+len_list0[iG]]
index = index + len_list0[iG]
frame_list = [] # initialize frame stack
if nonlin_flag:
    coeffs, _, vals = nonlin_coefs(nonlin_table_path,g,3)
else:
    coeffs = [0.0, 0.0, 0.0, 1.0]
    vals = np.ones(len(DNs))

stack_list = []
exp_repeat_counts = count_contiguous_repeats(exp_time_loop)
for j in range(len(exp_repeat_counts)):
    frame_list = []
    for t in range(exp_repeat_counts[j]):
        # Simulate full frame
        exp_time = exp_time_loop[t+j*exp_repeat_counts[j]]
        frame_sim = make_frame(fluxMap,bias,kgain_in,rn_in,g,
                               exp_time,coeffs,nonlin_flag)
        frame_list.append(frame_sim)
    frame_stack = np.stack(frame_list)
    stack_list.append(frame_stack)
stack_arr = np.stack(stack_list)

# set input parameters for calibrate_kgain function
min_val = 800
max_val = 3000
binwidth = 68

################### define class for tests ######################

class TestCalibrateKgain(unittest.TestCase):
    """Unit tests for calibrate_kgain method."""
    
    # sort out paths
    local_path = os.path.dirname(os.path.realpath(__file__))

    # example config file
    config_file = os.path.join(os.path.join(local_path, 'config_files'),
                               'kgain_parms.yaml')
    
    def setUp(self):

        self.emgain = emgain
        self.min_val = min_val
        self.max_val = max_val
        self.binwidth = binwidth

        # filter out expected warnings
        warnings.filterwarnings('ignore', category=UserWarning,
            module='kgain.calibrate_kgain')

    def test_expected_results_sub(self):
        """Outputs are as expected, for imported frames."""
        (kgain, read_noise_gauss, read_noise_stdev, ptc) = \
        calibrate_kgain(stack_arr, stack_arr2, self.emgain,
            self.min_val, self.max_val, self.binwidth, self.config_file)
        
        constants_config = loadyaml(self.config_file)
        signal_bins_N = constants_config['signal_bins_N']
        # kgain - should be close to the assumed value
        self.assertTrue(np.isclose(round(kgain,1), kgain_in, atol=0.5))
        # read noises. these are not requirements, but nice to check
        self.assertTrue(np.isclose(round(read_noise_gauss,1), rn_in, atol=8))
        self.assertTrue(np.isclose(round(read_noise_stdev,1), rn_in, atol=8))
        # check that the ptc output is the correct size
        self.assertTrue(np.all(np.equal(ptc.shape, (signal_bins_N,2))))

    def test_ndarray(self):
        """stack_arr and stack_arr2 must be ndarrays."""
        array1 = np.array([1, 2, 3])  # Shape (3,)
        array2 = np.array([[4, 5, 6], [7, 8, 9]])  # Shape (2, 3)
        array3 = np.array([[10], [11]])  # Shape (2, 1)
        object_arr = np.array([array1, array2, array3], dtype=object)
        # stack_arr
        with self.assertRaises(CalKgainException):
            calibrate_kgain(object_arr, stack_arr2, self.emgain, 
                self.min_val, self.max_val, self.binwidth, self.config_file)
        # stack_arr2
        with self.assertRaises(CalKgainException):
            calibrate_kgain(stack_arr, object_arr, self.emgain, 
                self.min_val, self.max_val, self.binwidth, self.config_file)
    
    def test_4D(self):
        """stack_arr should be 4-D."""
        with self.assertRaises(CalKgainException):
            calibrate_kgain(stack_arr[0], stack_arr2, self.emgain, 
                self.min_val, self.max_val, self.binwidth, self.config_file)

    def test_sub_stack_len(self):
        """stack_arr must have at least 10 sub-stacks."""
        with self.assertRaises(CalKgainException):
            calibrate_kgain(stack_arr[0:8], stack_arr2, self.emgain, 
                self.min_val, self.max_val, self.binwidth, self.config_file)

    def test_sub_sub_stack_len(self):
        """Each sub-stack of stack_arr must have 5 sub-stacks."""
        with self.assertRaises(CalKgainException):
            calibrate_kgain(stack_arr[:,0:3,:,:], stack_arr2, self.emgain, 
                self.min_val, self.max_val, self.binwidth, self.config_file)
    
    def test_3D(self):
        """stack_arr2 must be 3-D."""
        with self.assertRaises(CalKgainException):
            calibrate_kgain(stack_arr, stack_arr2[0], self.emgain, 
                self.min_val, self.max_val, self.binwidth, self.config_file)
    
    def test_sub_stack2_len(self):
        """stack_arr2 must have at least 30 sub-stacks."""
        with self.assertRaises(CalKgainException):
            calibrate_kgain(stack_arr, stack_arr2[0:28], self.emgain, 
                self.min_val, self.max_val, self.binwidth, self.config_file)

    def test_psi(self):
        """These three below must be positive scalar integers."""
        check_list = ut_check.psilist
        # min_val
        for perr in check_list:
            with self.assertRaises(TypeError):
                calibrate_kgain(stack_arr, stack_arr2, self.emgain, 
                    perr, self.max_val, self.binwidth, self.config_file)

        # max_val
        for perr in check_list:
            with self.assertRaises(TypeError):
                calibrate_kgain(stack_arr, stack_arr2, self.emgain, 
                    self.min_val, perr, self.binwidth, self.config_file)

        # binwidth
        for perr in check_list:
            with self.assertRaises(TypeError):
                calibrate_kgain(stack_arr, stack_arr2, self.emgain, 
                    self.min_val, self.max_val, perr, self.config_file)
        
    def test_binwidth(self):
        """binwidth must be >= 10."""
        with self.assertRaises(CalKgainException):
            calibrate_kgain(stack_arr, stack_arr2, self.emgain, 
                self.min_val, self.max_val, 9, self.config_file)
    
    def test_rps(self):
        """emgain must be a real positive scalar."""
        check_list = ut_check.rpslist
        # min_write
        for rerr in check_list:
            with self.assertRaises(TypeError):
                calibrate_kgain(stack_arr, stack_arr2, rerr, 
                    self.min_val, self.max_val, self.binwidth, self.config_file)
    
    def test_emgain(self):
        """emgain must be >= 1."""
        with self.assertRaises(CalKgainException):
            calibrate_kgain(stack_arr, stack_arr2, 0.5, 
                self.min_val, self.max_val, self.binwidth, self.config_file)

if __name__ == '__main__':
    unittest.main()

