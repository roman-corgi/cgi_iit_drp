# calibrate_darks
Calibrating the Master Dark

Given an array of frame stacks of the same number of dark frames (in DN
units), where the stacks are for various EM gain values and exposure times,
this module subtracts the bias from each frame in each stack, masks for
cosmic rays, optionally corrects for nonlinearity, converts DN to e-,
and averages each stack (which minimizes read noise since it has a mean of 0)
while accounting for masks.  It then computes a per-pixel map of fixed-pattern
noise (due to electromagnetic pick-up before going through the amplifier),
dark current, and the clock-induced charge (CIC), and it also returns the
bias offset value.  The function assumes the stacks have the same
noise profile (at least the same CIC, fixed-pattern noise, and dark
current).

This module satisfies these CTC requirements:

Given one or more raw dark frames collected during a dark frame collection
calibration activity, the CTC GSW shall compute a per-pixel map of
fixed-pattern noise in electrons.

Given one or more raw dark frames collected during a dark frame collection
calibration activity, the CTC GSW shall compute a per-pixel map of dark current
in electrons per second.

Given one or more raw dark frames collected during a dark frame collection
calibration activity, the CTC GSW shall compute a per-pixel map of EXCAM
clock-induced charge in electrons.

Given one or more raw dark frames collected during a dark frame collection
calibration activity, the CTC GSW shall compute a value of the mean
fixed-pattern and CIC residual in the raw frame prescan region.

# Usage

The main function call is given below:
`(F_map, C_map, D_map, bias_offset, F_image_map, C_image_map, D_image_map,
    Fvar, Cvar, Dvar, read_noise, R_map, F_image_mean, C_image_mean,
    D_image_mean) = calibrate_darks_lsq(stack_arr, g_arr, t_arr, k_arr,
    fwc_em_e, fwc_pp_e, meta_path, nonlin_path, Nem)`

If you would like to run simulated data through the function, see
the bottom of `calibrate_darks_lsq.py` for an example.  See the doc string for
calibrate_darks_lsq for more details about its inputs and returns.

# Author
Kevin Ludwick