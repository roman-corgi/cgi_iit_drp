"""File used to generate FITS files for testing in
ut_tpump_final.py.  All traps produced from the probability functions that
assume the time-dependent capture probability.

The test data that includes noise has been generated and tested with the unit
test a few times and passed each time.  It may be that a large statistical
fluctuation occurs in a fresh data generation with this script and then fails
a test in ut_tpump_final.py.  In that case, running this script again to see if
the tests pass on the fresh data.  But on average, the data generated here
should pass all the tests in ut_tpump_final.py.

Data in test_data_sub_frame_no_noise folder:
    metadata_test.yaml, gain of 1, CIC of 0, dark current of 0 for each
    temperature, read noise of 0, bias of 0, k gain of 1 e-/DN, 1 bit for ADU,
    and 0 injected charge.
    add_defect() was run, and the trap it created is only detectable when
    ill_corr is True.  Uses three distinct trap types:
    tauc = 1e-8s, E = 0.32 eV, cs = 2e-15 cm^2
    tauc2 = 1.2e-8s, E2 = 0.28 eV, cs2 = 12e-15 cm^2
    tauc3 = 1e-8s, E3 = 0.4 eV, cs3 = 2e-15 cm^2
    A trap that doesn't meet
    the length limit is at (71,84), and thus it doesn't show up as a trap
    (intentionally; if the threshold is low enough, noise could randomly
    allow it to be found by trap_id()).  Other traps:
    ((26, 28), 'CENel1', 0), ((50, 50), 'RHSel1', 0), ((60, 80), 'LHSel2', 0),
    ((68, 67), 'CENel2', 0), ((98, 33), 'LHSel3', 0), ((98, 33), 'RHSel2', 0),
    ((41, 15), 'CENel3', 0), ((89, 2), 'RHSel3', 0), ((89, 2), 'LHSel4', 0),
    ((10, 10), 'LHSel4', 0), ((10, 10), 'LHSel4', 1), ((56, 56), 'CENel4', 0),
    ((77, 90), 'RHSel4', 0), ((77, 90), 'CENel2', 0), (13, 21, 'LHSel1', 0)
Data in test_data_sub_frame_noise folder:
    Same as above except noise and gain are present:
    metadata_test.yaml, gain of 10, CIC of 200e-, dark current in e-:
    {180K: 0.163, 190K: 0.243, 200K: 0.323, 210K: 0.403, 220K: 0.483},
    read noise of 100e-, bias of 1000, k gain of 6 e-/DN, 14 bits for ADU,
    and exposure time of 1s.  The amount of injected charge per pixel is 500e-.
    add_defect() was run, and the trap it created is only detectable when
    ill_corr is True.  Uses the same distinct trap types and traps as above.
Data in test_data_sub_frame_noise_one_temp folder:
    Contains just 180K, scheme 1 from test_data_sub_frame_noise folder.
Data in test_data_sub_frame_noise_no_prob1 folder:
    Same as test_data_sub_frame_noise folder except
    all P1-type traps (including P1 in 'both' type) removed from scheme 1, and
    no injected charge.  And just 180K included.
    And the trap in the defect region (image-area pixel (13,21)) was changed
    to a 'LHSel2' trap because the trap type in other data folders included a
    P1 trap in scheme 1.  Trap list for this folder:
    ((13, 21), 'LHSel2', 0), ((26, 28), 'CENel1', 0), ((60, 80), 'LHSel2', 0),
    ((41, 15), 'CENel3', 0)
Data in test_data_sub_frame_noise_no_sch1_traps folder:
    Same as test_data_sub_frame_noise_no_prob1 folder except all detectable
    scheme-1 traps have been removed, and only scheme 1 from one
    temperature (180K) is present in the folder (and no injected charge).
    Trap list for this folder:
    ((26, 28), 'CENel1', 0), ((41, 15), 'CENel3', 0).  A trap that doesn't meet
    the length limit is at (71,84), and thus it doesn't show up as a trap
    (intentionally; if the threshold is low enough, noise could randomly
    allow it to be found by trap_id()).  It
    is also included in the other data sets, but the standard deviation is
    higher in those such that it isn't even recognized as a dipole in those.
    (Complete testing of length_lim is done in ut_trap_id.py.)
Data in test_data_sub_frame_no_noise_mean_field folder:
    Same as test_data_sub_frame_noise data except tauc is 3e-3s,
    only 180K is used, and only the following traps are included:
    ((68, 67), 'CENel2', 0), ((41, 15), 'CENel3', 0).
"""

import os
from pathlib import Path
import numpy as np

from astropy.io import fits
from emccd_detect.emccd_detect import EMCCDDetect
from emccd_detect.util.read_metadata_wrapper import MetadataWrapper
from cal.tpumpanalysis.ut_trap_fitting import (P1, P1_P1, P1_P2, P2, P2_P2,
    P2_P3, P3, P3_P3, tau_temp)

if __name__ == '__main__':
    here = os.path.abspath(os.path.dirname(__file__))
    meta_path = Path(here, '..', 'util', 'metadata_test.yaml')
    #meta_path = Path(here, '..', 'util', 'metadata.yaml')
    meta = MetadataWrapper(meta_path)
    num_pumps = 10000
    #nrows, ncols, _ = meta._imaging_area_geom()
    # the way emccd_detect works is that it takes an input for the selected
    # image area within the viable CCD pixels, so my input here must be that
    # smaller size (below) as opposed to the full useable CCD pixel size
    # (commented out above)
    nrows, ncols, _ = meta._unpack_geom('image')
    #EM gain
    g = 10 #1
    cic = 200 # 0
    rn = 100 #0
    dc = {180: 0.163, 190: 0.243, 200: 0.323, 210: 0.403,
          220: 0.483}
    # dc = {180: 0, 190: 0, 195: 0, 200: 0, 210: 0, 220: 0}
    bias = 1000 #0
    eperdn = 6 #1
    bias_dn = bias/eperdn
    nbits = 14 #1
    inj_charge = 500 # 0
    def _ENF(g, Nem):
        """
        Returns the ENF.
        """
        return np.sqrt(2*(g-1)*g**(-(Nem+1)/Nem) + 1/g)
    # std dev in e-, before gain division
    std_dev = np.sqrt(100**2 + _ENF(g,604)**2*g**2*(cic+ 1*dc[220]))
    fit_thresh = 3 #standard deviations above mean for trap detection
    #Offset ensures detection.  Physically, shouldn't have to add offset to
    #frame to meet threshold for detection, but in a small-sized frame, the
    # addition of traps increases the std dev a lot
    # (gain divided, w/o offset: from 22 e- before traps to 73 e- after adding)
    # If I run code with lower threshold, though, I can do an offset of 0.
    # For regular full-sized, definitely shouldn't have to add in offset.
    # Also, a trap can't capture more than mean per pixel e-, which is 200e-
    # in this case.  So max amp P1 trap will not be 2500e- but rather the
    #mean e- per pixel!  But this discrepancy doesn't affect validity of tests.

    offset_u = 0
    # offset_u = (bias_dn + ((cic+1*dc[220])*g + fit_thresh*std_dev/g)/eperdn+\
    #    inj_charge/eperdn)
    # #offset_l = bias_dn + ((cic+1*dc[220])*g - fit_thresh*std_dev/g)/eperdn
    # gives these 0 offset in the function (which gives e-), then add it in
    # by hand and convert to DN
    # and I increase dark current with temp linearly (even though it should
    # be exponential, but dc really doesn't affect anything here)
    emccd = {}
    # leaving out 170K
    #170K: gain of 10-20; gives g*CIC ~ 2000 e-
    # emccd[170] = EMCCDDetect(
    #         em_gain=1,#10,
    #         full_well_image=50000.,  # e-
    #         full_well_serial=50000.,  # e-
    #         dark_current=0.083,  # e-/pix/s
    #         cic=200, # e-/pix/frame; lots of CIC from all the prep clocking
    #         read_noise=100.,  # e-/pix/frame
    #         bias=bias,  # e-
    #         qe=0.9,
    #         cr_rate=0.,  # hits/cm^2/s
    #         pixel_pitch=13e-6,  # m
    #         eperdn=7.,
    #         nbits=14,
    #         numel_gain_register=604,
    #         meta_path=meta_path
    #    )
    #180K: gain of 10-20
    emccd[180] = EMCCDDetect(
            em_gain=g,#10,
            full_well_image=50000.,  # e-
            full_well_serial=50000.,  # e-
            dark_current=dc[180], #0.163,  # e-/pix/s
            cic=cic, # e-/pix/frame; lots of CIC from all the prep clocking
            read_noise=rn,  # e-/pix/frame
            bias=bias,  # e-
            qe=0.9,
            cr_rate=0.,  # hits/cm^2/s
            pixel_pitch=13e-6,  # m
            eperdn=eperdn,
            nbits=nbits,
            numel_gain_register=604,
            meta_path=meta_path
        )
    #190K: gain of 10-20
    emccd[190] = EMCCDDetect(
            em_gain=g,#10,
            full_well_image=50000.,  # e-
            full_well_serial=50000.,  # e-
            dark_current= dc[190],#0.243,  # e-/pix/s
            cic=cic, # e-/pix/frame
            read_noise=rn,  # e-/pix/frame
            bias=bias,  # e-
            qe=0.9,
            cr_rate=0.,  # hits/cm^2/s
            pixel_pitch=13e-6,  # m
            eperdn=eperdn,
            nbits=nbits,
            numel_gain_register=604,
            meta_path=meta_path
        )
    #195K: gain of 10-20
    # emccd[195] = EMCCDDetect(
    #         em_gain=g,#10,
    #         full_well_image=50000.,  # e-
    #         full_well_serial=50000.,  # e-
    #         dark_current= dc[195],#0.263,  # e-/pix/s
    #         cic=cic, # e-/pix/frame
    #         read_noise=rn,  # e-/pix/frame
    #         bias=bias,  # e-
    #         qe=0.9,
    #         cr_rate=0.,  # hits/cm^2/s
    #         pixel_pitch=13e-6,  # m
    #         eperdn=eperdn,
    #         nbits=nbits,
    #         numel_gain_register=604,
    #         meta_path=meta_path
    #     )
    #200K: gain of 10-20
    emccd[200] = EMCCDDetect(
            em_gain=g,#10,
            full_well_image=50000.,  # e-
            full_well_serial=50000.,  # e-
            dark_current=dc[200], #0.323,  # e-/pix/s
            cic=cic, # e-/pix/frame
            read_noise=rn,  # e-/pix/frame
            bias=bias,  # e-
            qe=0.9,
            cr_rate=0.,  # hits/cm^2/s
            pixel_pitch=13e-6,  # m
            eperdn=eperdn,
            nbits=nbits,
            numel_gain_register=604,
            meta_path=meta_path
        )
    #210K: gain of 10-20
    emccd[210] = EMCCDDetect(
            em_gain=g, #10,
            full_well_image=50000.,  # e-
            full_well_serial=50000.,  # e-
            dark_current=dc[210], #0.403,  # e-/pix/s
            cic=cic, # e-/pix/frame
            read_noise=rn,  # e-/pix/frame
            bias=bias,  # e-
            qe=0.9,
            cr_rate=0.,  # hits/cm^2/s
            pixel_pitch=13e-6,  # m
            eperdn=eperdn,
            nbits=nbits,
            numel_gain_register=604,
            meta_path=meta_path
        )
    #220K: gain of 10-20
    emccd[220] = EMCCDDetect(
            em_gain=g, #10,
            full_well_image=50000.,  # e-
            full_well_serial=50000.,  # e-
            dark_current=dc[220], #0.483,  # e-/pix/s
            cic=cic, # e-/pix/frame; divide by 15 to get the same 1000
            read_noise=rn,  # e-/pix/frame
            bias=bias,  # e-
            qe=0.9,
            cr_rate=0.,  # hits/cm^2/s
            pixel_pitch=13e-6,  # m
            eperdn=eperdn,
            nbits=nbits,
            numel_gain_register=604,
            meta_path=meta_path
        )

    # trap-pumping done when CGI is secondary instrument (i.e., dark):
    fluxmap = np.zeros((nrows, ncols))
    # frametime for pumped frames: 1000ms, or 1 s
    frametime = 1

    #when tauc is 3e-3, that gives a mean e- field of 2090 e-
    tauc = 1e-8 #3e-3
    tauc2 = 1.2e-8 # 3e-3
    tauc3 = 1e-8 # 3e-3
    # tried for mean field test, but gave low amps that got lost in noise
    tauc4 = 1e-3 #constant Pc over time not a great approximation in theory
    #In order of amplitudes overall (given comparable tau and tau2):
    # P1 biggest, then P3, then P2
    # E,E3 and cs,cs3 params below chosen to ensure a P1 trap found at its
    # peak amp for good eperdn determination
    # E3,cs3: will give tau outside of 1e-6,1e-2
    # for all temps except 220K; we'll just make sure it's present in all
    # scheme 1 stacks for all temps to ensure good eperdn for all temps;
    # E, cs: will give tau outside of 1e-6, 1e-2
    # for just 170K, which I took out of temp_data
    # E2, cs2: fine for all temps
    E = 0.32 #eV
    E2 = 0.28 #0.24 # eV
    E3 = 0.4 #eV
    # tried mean field test (gets tau = 1e-4 for 180K)
    E4 = 0.266 #eV
    cs = 2 #in 1e-19 m^2
    cs2 = 12 #3 #8 # in 1e-19 m^2
    cs3 = 2 # in 1e-19 m^2
    # for mean field test
    cs4 = 4 # in 1e-19 m^2
    #temp_data = np.array([170, 180, 190, 200, 210, 220])
    temp_data = np.array([180, 190, 195, 200, 210, 220])
    #temp_data = np.array([180])
    taus = {}
    taus2 = {}
    taus3 = {}
    taus4 = {}
    for i in temp_data:
        taus[i] = tau_temp(i, E, cs)
        taus2[i] = tau_temp(i, E2, cs2)
        taus3[i] = tau_temp(i, E3, cs3)
        taus4[i] = tau_temp(i, E4, cs4)
    #tau = 7.5e-3
    #tau2 = 8.8e-3
    time_data = (np.logspace(-6, -2, 100))*10**6 # in us
    #time_data = (np.linspace(1e-6, 1e-2, 50))*10**6 # in us
    time_data = time_data.astype(float)
    # make one phase time a repitition
    time_data[-1] = time_data[-2]
    time_data_s = time_data/10**6 # in s
    # half the # of frames for length limit
    length_limit = 5 #int(np.ceil((len(time_data)/2)))
    # mean of these frames will be a bit more than 2000e-, which is gain*CIC
    # std dev: sqrt(rn^2 + ENF^2 * g^2(e- signal))

    # with offset_u non-zero in below, I expect to get eperdn 4.7 w/ the code
    amps1 = {}; amps2 = {}; amps3 = {}
    amps1_k = {}; amps1_tau2 = {}; amps3_tau2 = {}; amps1_mean_field = {}
    amps2_mean_field = {}
    amps11 = {}; amps12 = {}; amps22 = {}; amps23 = {}; amps33 = {}; amps21 ={}
    for i in temp_data:
        amps1[i] = offset_u + g*P1(time_data_s, 0, tauc, taus[i])/eperdn
        amps11[i] = offset_u + g*P1_P1(time_data_s, 0, tauc, taus[i],
            tauc2, taus2[i])/eperdn
        amps2[i] = offset_u + g*P2(time_data_s, 0, tauc, taus[i])/eperdn
        amps12[i] = offset_u + g*P1_P2(time_data_s, 0, tauc, taus[i],
            tauc2, taus2[i])/eperdn
        amps22[i] = offset_u + g*P2_P2(time_data_s, 0, tauc, taus[i],
            tauc2, taus2[i])/eperdn
        amps3[i] = offset_u + g*P3(time_data_s, 0, tauc, taus[i])/eperdn
        amps33[i] = offset_u + g*P3_P3(time_data_s, 0, tauc, taus[i],
            tauc2, taus2[i])/eperdn
        amps23[i] = offset_u + g*P2_P3(time_data_s, 0, tauc, taus[i],
            tauc2, taus2[i])/eperdn
        # just for (98,33)
        amps21[i] =  offset_u + g*P1_P2(time_data_s, 0, tauc2, taus2[i],
            tauc, taus[i])/eperdn
        # now a special amps just for ensuring good eperdn determination
        # actually, doesn't usually meet trap_id thresh, but no harm
        # including it
        amps1_k[i] = offset_u + g*P1(time_data_s, 0, tauc3, taus3[i])/eperdn
        # for the case of (89,2) with a single trap with tau2
        amps1_tau2[i] = offset_u + g*P1(time_data_s, 0, tauc2, taus2[i])/eperdn
        # for the case of (77,90) with a single trap with tau2
        amps3_tau2[i] = offset_u + g*P3(time_data_s, 0, tauc2, taus2[i])/eperdn
        #amps1_k[i] = g*2500/eperdn
        # make a trap for the mean_field test (when mean field=400e- < 2500e-)
        #this trap peaks at 250 e-
        amps1_mean_field[i] = offset_u + \
            g*P1(time_data_s,0,tauc4,taus4[i])/eperdn
        amps2_mean_field[i] = offset_u + \
            g*P2(time_data_s,0,tauc4,taus4[i])/eperdn
    amps_1_trap = {1: amps1, 2: amps2, 3: amps3, 'sp': amps1_k,
            '1b': amps1_tau2, '3b': amps3_tau2, 'mf1': amps1_mean_field,
            'mf2': amps2_mean_field}
    amps_2_trap = {11: amps11, 12: amps12, 21: amps21, 22: amps22, 23: amps23,
        33: amps33}

    #r0c0[0]: starting row for imaging area (physical CCD pixels)
    #r0c0[1]: starting col for imaging area (physical CCD pixels)
    _, _, r0c0 = meta._imaging_area_geom()

    def add_1_dipole(img_stack, row, col, ori, prob, start, end, temp):
        """Adds a dipole to an image stack img_stack at the location of the
        bright pixel given by row and col (relative to image area coordinates)
        that is of orientation 'above' or
        'below' (specified by ori) for a number of unique phase times
        going from start to end (inclusive; don't use -1 for end; 0 for start
        means first frame, length of time array means last frame), and the
        dipole is of the probability function prob (which can be 1, 2, 3,
        'sp', '1b', '3b', 'mf1', or 'mf2').
        The temperature is specified by temp (in K)."""
        img_stack[:,r0c0[0]+row,r0c0[1]+col] += amps_1_trap[prob][temp][:]
        # length limit controlled by how 'long' deficit pixel is since
        #threshold should be met for all frames for bright pixel
        # The subtractions below may get the e- value negative for a sub frame,
        #  but this
        # is simply to ensure that the trap_id() threshold is met since
        # the std dev for a small sub will be big with the presence of traps
        if ori == 'above':
            #img_stack[start:end,r0c0[0]+row+1,r0c0[1]+col] = offset_l
            img_stack[start:end,r0c0[0]+row+1,r0c0[1]+col] -= \
                amps_1_trap[prob][temp][start:end]
        if ori == 'below':
            #img_stack[start:end,r0c0[0]+row-1,r0c0[1]+col] = offset_l
            img_stack[start:end,r0c0[0]+row-1,r0c0[1]+col] -= \
                amps_1_trap[prob][temp][start:end]
        return img_stack

    def add_2_dipole(img_stack, row, col, ori1, ori2, prob, start1, end1,
        start2, end2, temp):
        """Adds a 2-dipole to an image stack img_stack at the location of the
        bright pixel given by row and col (relative to image area coordinates)
        that is of orientation 'above' or
        'below' (specified by ori1 and ori2).  The 1st dipole is for a number
        of unique phase times going from start1 to end1, and
        the 2nd dipole starts from start2 and ends at end2 (inclusive; don't
        use -1 for end; 0 for start means first frame, length of time array
        means last frame). The 2-dipole is of probability function
        prob.  Valid values for prob are 11, 12, 22, 23, and 33.
        The temperature is specified by temp (in K)."""
        img_stack[:,r0c0[0]+row,r0c0[1]+col] += \
            amps_2_trap[prob][temp][:]
        # length limit controlled by how 'long' deficit pixel is since
        #threshold should be met for all frames for bright pixel
        # The subtractions below may get the e- value negative for a sub frame,
        #  but this
        # is simply to ensure that the trap_id() threshold is met since
        # the std dev for a small sub will be big with the presence of traps
        if ori1 == 'above':
            #img_stack[start1:end1,r0c0[0]+row+1,r0c0[1]+col] = offset_l
            img_stack[start1:end1,r0c0[0]+row+1,r0c0[1]+col] -= \
                amps_2_trap[prob][temp][start1:end1]
        if ori1 == 'below':
            #img_stack[start1:end1,r0c0[0]+row-1,r0c0[1]+col] = offset_l
            img_stack[start1:end1,r0c0[0]+row-1,r0c0[1]+col] -= \
                amps_2_trap[prob][temp][start1:end1]
        if ori2 == 'above':
            #img_stack[start2:end2,r0c0[0]+row+1,r0c0[1]+col] = offset_l
            img_stack[start2:end2,r0c0[0]+row+1,r0c0[1]+col] -= \
                amps_2_trap[prob][temp][start2:end2]
        if ori2 == 'below':
            #img_stack[start2:end2,r0c0[0]+row-1,r0c0[1]+col] = offset_l
            img_stack[start2:end2,r0c0[0]+row-1,r0c0[1]+col] -= \
                amps_2_trap[prob][temp][start2:end2]
        # technically, if there is overlap b/w start1:end1 and start2:end2,
        # then you are physically causing too big of a deficit since you're
        # saying more emitted than the amount captured in bright pixel, so
        # avoid this
        return img_stack

    def make_scheme_frames(emccd_inst, phase_times = time_data,
        inj_charge = inj_charge ):
        """Makes a series of frames according to the emccd_detect instance
        emccd_inst, one for each element in the array phase_times (assumed to
        be in s).
        """
        full_frames = []
        for i in range(len(phase_times)):
            full = (emccd_inst.sim_full_frame(fluxmap,frametime)).astype(float)
            full_frames.append(full)
        # inj charge is before gain, but since it has no variance,
        # g*0 = no noise from this
        full_frames = np.stack(full_frames)
        # lazy and not putting in the last image row and col, but doesn't
        #matter since I only use prescan and image areas
        # add to just image area so that it isn't wiped with bias subtraction
        full_frames[:,r0c0[0]:,r0c0[1]:] += inj_charge
        return full_frames

    def add_defect(sch_imgs, prob, ori, temp):
        """Adds to all frames of an image stack sch_imgs a defect area with
        local mean above image-area mean such that a
        dipole in that area that isn't detectable unless ill_corr is True.
        The dipole is a single trap with orientation
        ori ('above' or 'below') and is of probability function prob
        (can be 1, 2, or 3).  The temperature is specified by temp (in K).

        Note: If a defect region is arbitrarily small (e.g., a 2x2 region of
        very bright pixels hiding a trap dipole), that trap simply will not
        be found since the illumination correction bin size is not allowed to
        be less than 5.  In v2.0, a moving median subtraction can be
        implemented that would be more likely to catch cases similar to that.
        However, physically, a defect region of such a small number of rows is
        improbable; even a cosmic ray hit, which could have this signature for
        perhaps 1 phase time, is very unlikely to hit the same region while
        data for each phase time is being taken."""
        # area with defect (high above mean),
        # but no dipole that stands out enough without ill_corr = True
        sch_imgs[:,r0c0[0]+12:r0c0[0]+22,r0c0[1]+17:r0c0[1]+27]=g*9000/eperdn
        # now a dipole that meets threshold around local mean doesn't meet
        # threshold around frame mean; would be detected only after
        # illumination correction
        sch_imgs[:,r0c0[0]+13, r0c0[1]+21] += \
            amps_1_trap[prob][temp][:]
            #offset_u + amps_1_trap[prob][temp][:]
            #2*offset_u + fit_thresh*std_dev/eperdn
        if ori == 'above':
            sch_imgs[:,r0c0[0]+13+1, r0c0[1]+21] -= \
                amps_1_trap[prob][temp][:]
        if ori == 'below':
            sch_imgs[:,r0c0[0]+13-1, r0c0[1]+21] -= \
                amps_1_trap[prob][temp][:]
                # 2*offset_u - fit_thresh*std_dev/eperdn

        return sch_imgs
    #initializing
    sch = {1: None, 2: None, 3: None, 4: None}
    #temps = {170: sch, 180: sch, 190: sch, 200: sch, 210: sch, 220: sch}
    temps = {180: sch, 190: sch, 200: sch, 210: sch, 220: sch}
    #temps = {180: sch}

    # first, get rid of files already existing in the folders where I'll put
    # the simulated data
    for temp in temps.keys():
        for sch in [1,2,3,4]:
            curr_sch_dir = Path(here, 'test_data_sub_frame_noise', str(temp)+'K',
                'Scheme_'+str(sch))
            for file in os.listdir(curr_sch_dir):
                os.remove(Path(curr_sch_dir, file))

    for temp in temps.keys():
        for sch in [1,2,3,4]:
            temps[temp][sch] = make_scheme_frames(emccd[temp])
        # 14 total traps (15 with the (13,19) defect trap); at least 1 in every
        # possible sub-electrode location
        # careful not to add traps in defect region; do that with add_defect()
        # careful not to add, e.g., bright pixel of one trap in the deficit
        # pixel of another trap since that would negate the original trap

        # add in 'LHSel1' trap in midst of defect for all phase times
        # (only detectable with ill_corr)
        add_defect(temps[temp][1], 1, 'below', temp)
        add_defect(temps[temp][3], 3, 'below', temp)
        #this defect was used for k_prob=2 case instead of the 2 lines above
        # 'LHSel2':
    #    add_defect(temps[temp][1], 2, 'above', temp)
    #    add_defect(temps[temp][2], 1, 'below', temp)
    #    add_defect(temps[temp][4], 3, 'above', temp)
        # add in 'special' max amp trap for good eperdn determination
        # has tau value outside of 1e-6 to 1e-2, but provides a peak trap
        # actually, doesn't meet threshold usually to count as trap, but
        #no harm leaving it in
        add_1_dipole(temps[temp][1], 33, 77, 'below', 'sp', 0, 100, temp)
        # add in 'CENel1' trap for all phase times
    #    add_1_dipole(temps[temp][3], 26, 28, 'below', 'mf2', 0, 100, temp)
    #    add_1_dipole(temps[temp][4], 26, 28, 'above', 'mf2', 0, 100, temp)
        add_1_dipole(temps[temp][3], 26, 28, 'below', 2, 0, 100, temp)
        add_1_dipole(temps[temp][4], 26, 28, 'above', 2, 0, 100, temp)
        # add in 'RHSel1' trap for more than length limit (but diff lengths)
        #unused sch2 in this same pixel that is compatible with another trap
        add_1_dipole(temps[temp][1], 50, 50, 'above', 1, 0, 100, temp)
        add_1_dipole(temps[temp][4], 50, 50, 'above', 3, 3, 98, temp)
        add_1_dipole(temps[temp][2], 50, 50, 'below', 1, 2, 99, temp)
        # FALSE TRAPS: 'LHSel2' trap that doesn't meet length limit of unique
        # phase times even though the actual length is met for first 2
        # (and/or doesn't pass trap_id(), but I've already tested this case in
        # its unit test file)
        # (3rd will be 'unused')
        add_1_dipole(temps[temp][1], 71, 84, 'above', 2, 95, 100, temp)
        add_1_dipole(temps[temp][2], 71, 84, 'below', 1, 95, 100, temp)
        add_1_dipole(temps[temp][4], 71, 84, 'above', 3, 9, 20, temp)
        # 'LHSel2' trap
        add_1_dipole(temps[temp][1], 60, 80, 'above', 2, 1, 100, temp)
        add_1_dipole(temps[temp][2], 60, 80, 'below', 1, 1, 100, temp)
        add_1_dipole(temps[temp][4], 60, 80, 'above', 3, 1, 100, temp)
        # 'CENel2' trap
        add_1_dipole(temps[temp][1], 68, 67, 'above', 1, 0, 100, temp)
        add_1_dipole(temps[temp][2], 68, 67, 'below', 1, 0, 100, temp)
    #    add_1_dipole(temps[temp][1], 68, 67, 'above', 'mf1', 0, 100, temp)
    #    add_1_dipole(temps[temp][2], 68, 67, 'below', 'mf1', 0, 100, temp)
        # 'RHSel2' and 'LHSel3' traps in same pixel (could overlap phase time),
        # but good detectability means separation of peaks
        add_1_dipole(temps[temp][1], 98, 33, 'above', 1, 0, 100, temp)
        add_2_dipole(temps[temp][2], 98, 33, 'below', 'below', 21,
            60, 100, 0, 40, temp) #80, 100, 0, 20, temp)
        add_2_dipole(temps[temp][4], 98, 33, 'below', 'below', 33,
            60, 100, 0, 40, temp)
        # old:
        # add_2_dipole(temps[temp][2], 98, 33, 'below', 'below', 21,
        #     50, 100, 0, 50, temp) #80, 100, 0, 20, temp)
        # add_2_dipole(temps[temp][4], 98, 33, 'below', 'below', 33,
        #     50, 100, 0, 50, temp)
        # 'CENel3' trap (where sch3 has a 2-trap where one goes unused)
        add_2_dipole(temps[temp][3], 41, 15, 'above', 'above', 23,
           30, 100, 0, 30, temp)
        add_1_dipole(temps[temp][4], 41, 15, 'below', 2, 30, 100, temp)
        # 'RHSel3' and 'LHSel4'
        add_1_dipole(temps[temp][1], 89, 2, 'below', '1b', 0, 100, temp)
        add_2_dipole(temps[temp][2], 89, 2, 'above', 'above', 12,
            60, 100, 0, 30, temp) #30 was 40 in the past
        add_2_dipole(temps[temp][3], 89, 2, 'above', 'above', 33,
            60, 100, 0, 40, temp)
        # 2 'LHSel4' traps; whether the '0' or '1' trap gets assigned tau2 is
        # somewhat random; if one has an earlier starting temp than the other,
        # it would get assigned tau
        add_2_dipole(temps[temp][1], 10, 10, 'below', 'below', 11,
            0, 40, 63, 100, temp)
        add_2_dipole(temps[temp][2], 10, 10, 'above', 'above', 22,
            0, 40, 63, 100, temp)
        add_2_dipole(temps[temp][3], 10, 10, 'above', 'above', 33,
            0, 40, 63, 100, temp) #30, 60, 100
        # old:
        # add_2_dipole(temps[temp][1], 10, 10, 'below', 'below', 11,
        #     0, 40, 50, 100, temp)
        # add_2_dipole(temps[temp][2], 10, 10, 'above', 'above', 22,
        #     0, 40, 50, 100, temp)
        # add_2_dipole(temps[temp][3], 10, 10, 'above', 'above', 33,
        #     0, 40, 50, 100, temp)
        # 'CENel4' trap
        add_1_dipole(temps[temp][1], 56, 56, 'below', 1, 1, 100, temp)
        add_1_dipole(temps[temp][2], 56, 56, 'above', 1, 3, 99, temp)
        #'RHSel4' and 'CENel2' trap (tests 'a' and 'b' splitting in trap_fit_*)
        add_2_dipole(temps[temp][1], 77, 90, 'above', 'below', 12,
            60, 100, 0, 40, temp)
        add_2_dipole(temps[temp][2], 77, 90, 'below', 'above', 11,
            60, 100, 0, 40, temp)
        add_1_dipole(temps[temp][3], 77, 90, 'below', '3b', 0, 40, temp)
        # old:
        # add_2_dipole(temps[temp][1], 77, 90, 'above', 'below', 12,
        #     30, 100, 0, 30, temp)
        # add_2_dipole(temps[temp][2], 77, 90, 'below', 'above', 11,
        #     53, 100, 0, 53, temp)
        # add_1_dipole(temps[temp][3], 77, 90, 'below', '3b', 0, 30, temp)
        pass
        # save to FITS files
        for sch in [1,2,3,4]:
            for i in range(len(temps[temp][sch])):
                hdr = fits.Header()
                hdr['EM_GAIN'] = g
                hdr['PHASE_T'] = time_data[i]
                prim = fits.PrimaryHDU(header = hdr)
                hdr_img = fits.ImageHDU(temps[temp][sch][i])
                hdul = fits.HDUList([prim, hdr_img])
                t = time_data[i]
                curr_sch_dir = Path(here, 'test_data_sub_frame_noise', str(temp)+'K',
                'Scheme_'+str(sch))
                if os.path.isfile(Path(curr_sch_dir,
                'TPUMP_Npumps_10000_gain'+str(g)+'_phasetime'+str(t)+'.fits')):
                    hdul.writeto(Path(curr_sch_dir,
                    'TPUMP_Npumps_10000_gain'+str(g)+'_phasetime'+
                    str(t)+'_2.fits'), overwrite = True)
                else:
                    hdul.writeto(Path(curr_sch_dir,
                    'TPUMP_Npumps_10000_gain'+str(g)+'_phasetime'+
                    str(t)+'.fits'), overwrite = True)
