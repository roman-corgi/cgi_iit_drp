import os
from pathlib import Path
import numpy as np
from astropy.io import fits

from cal.util import check
from cal.util.loadyaml import loadyaml
from cal.util.abs_rel_path import abs_or_rel_path
from cal.util.gsw_process import Process
from cal.util.read_metadata import Metadata as MetadataWrapper

from cal.ampthresh.ampthresh import ampthresh

from cal.EM_gain_fit.EM_gain_tools import _EM_gain_fit_conv

here = os.path.abspath(os.path.dirname(__file__))

class EMGainFit:
    '''Class for fitting EM gain from a frame or a set of frames.

    Parameters
    ----------
    cam_mode : string
        Indicates what type of frames to analyze.  For LOCAM frames, use
        'locam'.  For EXCAM high-gain frames, use 'excam_high'.  For EXCAM
        low-gain frames, use 'excam_high'.

    eperdn : float
        Electrons per dn conversion factor (detector k gain), >= 0.

    bias_offset : float
        Median number of counts in the bias region due to fixed non-bias
        noise not in common with the image region.  Only meaningful for EXCAM
        frames; if cam_mode is 'locam', this number is not used.
        The bias is computed for the image region based on the prescan from
        each frame, and the bias_offset is how many additional counts the
        prescan had from e.g. fixed-pattern noise.  This value is
        subtracted from each measured bias.  Units of DNs, >= 0.

    com_gain : float
        Electron multiplying gain that was commanded for the
        frames. >= 1.

    exptime : float
        Exposure time for each frame, in seconds.
        For LOCAM, the time is fixed and is
        drawn from the config file at config_path, so this input is not used
        for the LOCAM case.

    config_path : string, optional
        Absolute or relative path of .yaml config file to use.
        If None, the config file in the util folder is used
        (excam_config.yaml for EXCAM, locam_config.yaml for LOCAM).  Structure
        of dictionary in .yaml file must conform to format found in those
        files.  Defaults to None.

    gain_fit_path : string, optional
        Absolute or relative path of .yaml file to use for inputs for EM gain
        fitting. If None, EM_gain_fit_params.yaml in the EMgain_fit module is
        used.  Structure of dictionary in .yaml file must conform to format
        found in that file, and descriptions of the parameters are also in that
        file.  Defaults to None.

    binsize : int, optional
        When selecting the region of interest on an illuminated frame, a frame
        is broken up into squares of side length binsize for smoothing out
        the mask via the function 
        mask_smoothing.  Defaults to 4.  
    
    smoothing_threshold : float, optional
        Number between 0 and 1.  If this fraction of mask pixels within a 
        square of side length binsize is masked, the whole bin square is 
        masked, and the whole bin square is not masked if below this fraction.
        Defaults to 0.95.

    ampthresh_mask_thresh : int, optional
        Minimum number of pixels that should be False in the mask produced from
        ampthresh (used for finding region of interest in illuminated frame).  
        It also serves as the minimum number that should be True.  Therefore, 
        this number should be less than half the number of pixels in the image 
        area.  Defaults to 100.

    gmax_factor : float, optional
        In theory, it is possible for the fitted EM gain to be larger than 
        gmax coming from config_path for EXCAM (g_max_comm for LOCAM), 
        and the upper bound for gain in the fit is gmax*gmax_factor for EXCAM
        (g_max_comm*gmax_factor for LOCAM).  Should be >= 1.
        Defaults to 1.5.
    
    min_freq : int, optional
        When a series of cuts is made, this value is the minimum frequency of 
        the histogram which must be present in all cuts of histograms 
        (to try to ensure good statistics).  Can be as low as 0.  
        Defaults to 3.
    '''

    def __init__(self, cam_mode, eperdn, bias_offset, com_gain, exptime,
                 config_path=None, gain_fit_path=None, binsize=4, 
                 smoothing_threshold=0.95, ampthresh_mask_thresh=100, 
                 gmax_factor=1.5, min_freq=3):

        if gain_fit_path is None:
            gain_fit_path = Path(here, 'EM_gain_fit_params.yaml')

        gain_fit_path = abs_or_rel_path(gain_fit_path)
        # assume gain_fit_dict YAML set up as intended; if not, the user
        # will get key error, which is sufficient
        gain_fit_dict = loadyaml(gain_fit_path)
        check.positive_scalar_integer(binsize, 'binsize', TypeError)
        self.binsize = binsize
        check.real_positive_scalar(smoothing_threshold, 'smoothing_threshold',
                                   TypeError)
        if smoothing_threshold > 1:
            raise ValueError("smoothing_threshold must be <= 1.")
        self.smoothing_threshold = smoothing_threshold
        check.positive_scalar_integer(ampthresh_mask_thresh, 
                                      'ampthresh_mask_thresh', TypeError)
        self.ampthresh_mask_thresh = ampthresh_mask_thresh
        check.real_positive_scalar(gmax_factor, 'gmax_factor', TypeError)
        if gmax_factor < 1:
            raise ValueError('gmax_factor must be >= 1.')
        self.gmax_factor = gmax_factor
        check.nonnegative_scalar_integer(min_freq, 'min_freq', TypeError)
        self.min_freq = min_freq

        if cam_mode == 'locam':
            if config_path is None:
                config_path = str(Path(here, '..', 'util',
                                    'locam_config.yaml'))
            config_path = str(abs_or_rel_path(config_path))
            config_dict = loadyaml(config_path)
            #checking things from config_dict unique to locam
            g_max_comm = config_dict['g_max_comm']
            check.real_positive_scalar(g_max_comm, 'g_max_comm', TypeError)
            if g_max_comm < 1:
                raise ValueError('g_max_comm must be >= 1')
            tframe = config_dict['tframe']
            check.real_positive_scalar(tframe, 'tframe', TypeError)
            e_max_age = config_dict['e_max_age']
            check.positive_scalar_integer(e_max_age, 'e_max_age', TypeError)
            locam_dict = gain_fit_dict['locam']
            divisor = locam_dict['divisor']
            locam_rn = locam_dict['rn']
            check.real_positive_scalar(locam_rn, 'LOCAM rn', TypeError)
            self.rn = locam_rn
            Nem = locam_dict['Nem']
            check.positive_scalar_integer(Nem, 'Nem', TypeError)
            self.Nem = Nem
            locam_sat_thresh = locam_dict['locam_sat_thresh']
            check.real_positive_scalar(locam_sat_thresh, 'locam_sat_thresh',
                                   TypeError)
            # don't restrict LOCAM or EXCAM sat_thresh values to be 1 or less
            # since it's nice to have that parameter to tweak in case, for
            # example, commanded gain is an underestimate for the true gain
            self.locam_sat_thresh = locam_sat_thresh
            num_summed_frames = locam_dict['num_summed_frames']
            check.positive_scalar_integer(num_summed_frames,
                                          'num_summed_frames', TypeError)
            self.num_summed_frames = num_summed_frames

        elif cam_mode == 'excam_high' or cam_mode == 'excam_low':
            if config_path is None:
                config_path = str(Path(here, '..', 'util',
                                       'excam_config.yaml'))
                config_dict = loadyaml(config_path)
            config_path = str(abs_or_rel_path(config_path))
            config_dict = loadyaml(config_path)
            self.rn = config_dict['rn']
            # additional check above what load_excam_config() does:
            # we want rn > 0 so that read noise distribution doesn't blow up
            check.real_positive_scalar(self.rn, 'EXCAM rn', TypeError)
            if cam_mode == 'excam_high':
                excam_dict = gain_fit_dict['excam_high']
            if cam_mode == 'excam_low':
                excam_dict = gain_fit_dict['excam_low']
            lthresh = excam_dict['lthresh']
            divisor = excam_dict['divisor']
            sat_thresh = excam_dict['sat_thresh']
            plat_thresh = excam_dict['plat_thresh']
            cosm_filter = excam_dict['cosm_filter']
            cosm_box = excam_dict['cosm_box']
            cosm_tail = excam_dict['cosm_tail']
            rowreadtime = excam_dict['rowreadtime']
            # non-LOCAM params (bias_offset, sat_thresh, plat_thresh, and
            # cosm_filter, cosm_box, cosm_tail, rowreadtime) 
            # checked by Process class later
            self.sat_thresh = sat_thresh
            self.plat_thresh = plat_thresh
            self.cosm_filter = cosm_filter
            self.cosm_box = cosm_box
            self.cosm_tail = cosm_tail
            self.rowreadtime = rowreadtime
            # bias_offset only valid for EXCAM, and checked by Process class
            self.bias_offset = bias_offset
            # exptime input only used for EXCAM (comes from config file for
            # LOCAM)
            check.real_positive_scalar(exptime, 'exptime', TypeError)
            self.exptime = exptime
            # lthresh only used for EXCAM
            check.real_nonnegative_scalar(lthresh, 'lthresh', TypeError)
            self.lthresh = lthresh
            gmax = config_dict['gmax']
            check.real_nonnegative_scalar(gmax, 'gmax', TypeError)
            if gmax < 1:
                raise ValueError('gmax must be >= 1')
        else:
            raise ValueError('invalid cam_mode value')

        self.cam_mode = cam_mode
        self.config_path = config_path
        self.config_dict = config_dict
        # eperdn for EXCAM checked by Process class, but check it in case LOCAM
        check.real_positive_scalar(eperdn, 'eperdn', TypeError)
        self.eperdn = eperdn


        #unpack values common to all modes
        darke = config_dict['darke']
        cic = config_dict['cic']
        fwc_em = config_dict['fwc_em']
        fwc = config_dict['fwc']
        check.real_nonnegative_scalar(darke, 'darke', TypeError)
        check.real_nonnegative_scalar(cic, 'cic', TypeError)
        check.positive_scalar_integer(fwc_em, 'fwc_em', TypeError)
        check.positive_scalar_integer(fwc, 'fwc', TypeError)
        tol = float(gain_fit_dict['tol'])
        lambda_factor = gain_fit_dict['lambda_factor']
        gain_factor = gain_fit_dict['gain_factor']
        rn_mean = gain_fit_dict['rn_mean']
        diff_tol = float(gain_fit_dict['diff_tol'])

        check.real_positive_scalar(com_gain, 'com_gain', TypeError)
        if com_gain <= 1:
            raise ValueError('com_gain must be > 1.')
        self.com_gain = com_gain
        check.real_positive_scalar(divisor, 'divisor', TypeError)
        if divisor < 1:
            raise ValueError('divisor must be >= 1.')
        self.divisor = divisor
        check.real_nonnegative_scalar(tol, 'tol', TypeError)
        self.tol = tol
        check.real_positive_scalar(lambda_factor, 'lambda_factor', TypeError)
        self.lambda_factor = lambda_factor
        check.real_positive_scalar(gain_factor, 'gain_factor', TypeError)
        self.gain_factor = gain_factor
        check.real_scalar(rn_mean, 'rn_mean', TypeError)
        self.rn_mean = rn_mean
        check.real_nonnegative_scalar(diff_tol, 'diff_tol', TypeError)
        self.diff_tol = diff_tol

    def read_in_locam_files(self, directory, bias_directory, com_gain, 
                            bad_pix=None, 
                            summed_flat=None, do_ampthresh=None, mask=None):
        '''Reads in a directory containing .fits files of illuminated
        LOCAM summed frames at some commanded gain
        (1 or multiple summed frames) and a directory containing .fits files of
        bias LOCAM summed frames at the same commanded gain (the same number
        of summed frames as in the other directory).  The function uses
        the built-in Python function "sorted" to sort the filenames in each
        directory, which sorts the filename strings alpha-numerically, giving
        precedence to numerical ordering over alphabetic ordering.  For
        example, for two files called '4a' and '3z', '3z' would come first.
        This function assumes the filenames in the directories,
        when ordered in this way, correspond to each other so that
        the bias frame is subtracted from the corresponding illuminated frame
        (assuming this is important to the user).
        After the bias has been subtracted, a rough check for saturated pixels
        is done (utilizing the commanded gain, assuming it is roughly accurate,
        and locam_sat_thresh from the
        .yaml file from gain_fit_path). Any saturated pixels are marked as
        NaNs and then ignored.  The resulting frames are multiplied
        by k gain to convert from DN to electrons and flattened into one array
        for analysis.

        This can be used for frames at non-unity or unity gain.

        If illuminated summed frames that are illuminated only in a region of 
        interest within the image area (such as a pupil image) are input via 
        the directory argument, then 
        cal.ampthresh is used to select the relevant illuminated pixels when 
        do_ampthresh is set to True.

        Parameters
        ----------
        directory : string
            Absolute or relative path of directory containing the .fits files
            of LOCAM illuminated summed frames at com_gain.
            Files that are not .fits files will be unaffected.  15 or more
            summed frames are recommended for a successful result using
            maximum likelihood estimation (MLE), but fewer may yield
            a trustworthy result. A simple estimate even for 1 summed frame
            is possible after processing:  frames.mean()/framesG1.mean(), where
            framesG1 is the result of processing unity gain summed frames
            with this function.

        bias_directory : string
            Absolute or relative path of directory containing the .fits files
            of LOCAM bias frames at com_gain.
            Files that are not .fits files will be unaffected.  Number of files
            and ordering of filenames should correspond to what is in
            directory.

        com_gain : string
            Commanded EM gain of the frames in the two directories.  (Can be
            unity or non-unity.)

        bad_pix : array_like, optional
            Bad pixel mask. Bad pixels are True.  If None, no bad pixels are
            assumed.  Must match shape of LOCAM frame.  Defaults to None.

        summed_flat : array-like, optional
            Summed flat field array (in e-) for pixel scaling of image section. 
            The frame should already be processed.  
            Must match shape of LOCAM frame.
            If None, no flat correction is applied.  Defaults to None.

        do_ampthresh : bool, optional
            True means the frames will be masked (mask generated by amplitude 
            thresholding) so that only the relevant 
            illuminated part of the image is used for fitting.  If None (which 
            is the default), this is set to  
            True if no input mask is provided.
        
        mask : array-like, optional
            Mask array for selecting the pixels to use for analysis for fitting
            for the gain.  False array values for accepting pixels, True for 
            masking them.  Must match shape of LOCAM frame.  
            Defaults to None, which means no input mask is used.

        Returns
        -------
        frames : array
            Flattened array of all bias-subtracted good pixels (in e-)
            from all the frames in directory.
        
        mask : array stack
            A stack of the mask arrays that were used, in the order according 
            to the alpha-numeric ordering of filenames in directory (where 
            numbers take precedence over characters, i.e., '0' would be come 
            before 'a').  These arrays include masking due to the input 
            bad_pix as well.  This may be useful for examining by eye how 
            close a mask created by this function is to the desired mask. 
        '''
        if self.cam_mode != 'locam':
            raise ValueError('read_in_locam_files() should only be called if '
                             'cam_mode=\'locam\'.')
        if do_ampthresh is not None:
            if not isinstance(do_ampthresh, bool):
                raise TypeError('If ampthresh is specified, it must be '
                                'Boolean.')
            if do_ampthresh and mask is not None:
                raise ValueError('do_ampthresh cannot be True while mask is '
                                 'provided as an input.')
        elif do_ampthresh is None:
            if mask is None:
                do_ampthresh = True
            elif mask is not None:
                do_ampthresh = False
        if mask is not None:
            check.twoD_array(mask, 'mask', TypeError)
            if mask.dtype != bool:
                raise TypeError('mask must be an array of bool type.')
            
        directory = abs_or_rel_path(directory)
        bias_directory = abs_or_rel_path(bias_directory)
        dir_list = []
        bias_dir_list = []
        for i in  os.listdir(directory):
            if i[-5:] == '.fits':
                dir_list.append(i)
        for i in  os.listdir(bias_directory):
            if i[-5:] == '.fits':
                bias_dir_list.append(i)

        sorted_dir_list = sorted(dir_list)
        sorted_bias_dir_list = sorted(bias_dir_list)

        framelist = []
        mask_stack = []
        for ind in range(len(sorted_dir_list)):
            ill_file = sorted_dir_list[ind]
            if ill_file.endswith('fits'):
                ill_f = os.path.join(directory, ill_file)
                ill_d = fits.getdata(ill_f).astype(float)
                bias_file = sorted_bias_dir_list[ind]
                bias_f = os.path.join(bias_directory, bias_file)
                bias_d = fits.getdata(bias_f).astype(float)

                if bad_pix is None:
                    bad_pix = np.zeros_like(ill_d, dtype=bool)
                else:
                   check.twoD_array(bad_pix, 'bad_pix', TypeError)
                   if bad_pix.shape != ill_d.shape:
                       raise ValueError('Shape of bad_pix array must match '
                                        'shape of LOCAM summed frames.') 
                   
                if summed_flat is None:
                    s_flat = np.ones_like(ill_d, dtype=float)
                else:
                   check.twoD_array(summed_flat, 'summed_flat', TypeError)
                   if summed_flat.shape != ill_d.shape:
                       raise ValueError('Shape of summed_flat array must '
                                        'match shape of LOCAM summed frames.')
                   s_flat = summed_flat
                # subtract bias; gets rid of FPN
                data = ill_d - bias_d
                

                # convert from DN to e-
                frame = data * self.eperdn

                # Divide by flat
                # Divide image by flat only where flat is not equal to 0.
                # Where flat is equal to 0, set image to zero
                frame = np.divide(frame,
                                    s_flat,
                                    out=np.zeros_like(frame),
                                    where=s_flat != 0)
                
                if frame[frame > 0].size == 0:
                    raise Exception('Frame has no electron counts greater '
                                    'than 0.')
                # get relevant illuminated pixels
                if mask is not None:
                    if mask.shape != frame.shape:
                        raise ValueError('mask must have same shape as image '
                                         'area of frames.')
                    mask = np.logical_or(bad_pix, mask)
                elif do_ampthresh: #mask would be None here
                    nBin = 2 # starting point
                    mask = np.ones_like(frame).astype(bool)
                    if self.ampthresh_mask_thresh >= np.floor(mask.size/2):
                        raise ValueError('ampthresh_mask_thresh bigger than '
                                         'half the frame size.') 
                    # mask out the negative-valued pixels to extract signal 
                    ma_frame = np.ma.masked_array(frame, (frame < 0))
                    counter = 0
                    while (mask[mask==True].size < self.ampthresh_mask_thresh 
                      or mask[mask==False].size < self.ampthresh_mask_thresh):
                        try:
                            mask = ampthresh(ma_frame, nBin=nBin)
                            mask = mask.filled(0)
                        except:
                            pass
                        nBin += 1
                        counter += 1
                        if counter > 1000:
                            raise Exception('Mask could not be found that '
                                            'passes ampthresh_mask_thresh. '
                                            'Reduce ampthresh_mask_thresh.')

                    smooth_m = mask_smoothing(mask, self.binsize, 
                                              self.smoothing_threshold)
                    temp_t = self.smoothing_threshold
                    while (smooth_m[smooth_m==True].size < 
                                    self.ampthresh_mask_thresh) or \
                        (smooth_m[smooth_m==False].size < 
                                    self.ampthresh_mask_thresh) :
                        temp_t *= 0.9 #decrease it some
                        smooth_m = mask_smoothing(mask, self.binsize, 
                                              temp_t)
                        if temp_t <= np.finfo(float).eps:
                            raise Exception("Cannot find a suitable mask for "
                                            "illuminated frame.")

                    mask = (smooth_m==False)
                    mask = np.logical_or(bad_pix, mask)
                elif mask is None and not do_ampthresh:
                    mask = bad_pix

                mask_stack.append(mask)
                frame_m = np.ma.masked_array(frame, mask)
                frame = frame_m.filled(np.nan)

                # check for saturation despite the potential inaccuracy of
                # commanded gain; the confidence in it can be tuned via
                # locam_sat_thresh
                fwc = self.config_dict['fwc']
                fwc_em = self.config_dict['fwc_em']
                e_max_age = self.config_dict['e_max_age']
                sat_level = (self.locam_sat_thresh *
                    min(fwc_em, fwc * com_gain, e_max_age) *
                            self.num_summed_frames)

                frame[ill_d >= sat_level] = np.nan
                
                framelist.append(frame.ravel())

        if len(framelist) > 1:
            framelist = np.stack(framelist)
        s = np.ravel(framelist)
        s_out_ind = np.where(~np.isnan(s))
        frames = s[s_out_ind]

        return frames, mask_stack


    def read_in_excam_files(self, directory, bad_pix=None,
                      nonlin_path=None, meta_path=None, flat=None, dark=None, 
                      do_ampthresh=None, mask=None, desmear_flag=False):
        '''Reads in a directory containing .fits files of EXCAM frames (1 or
        multiple).  It processes them from L1 to L2b (flat-divided,
        bias-subtracted, gain-divided frames in electrons),
        and then it multiplies by the gain
        to undo the division. (No dark subtraction is performed to prevent any
        false subtraction, and it also is not needed. Then all image-area pixel
        data is collected, bad pixels are marked as NaNs and ignored, and
        the data is combined into one flattened array for analysis.

        If illuminated frames that are illuminated only in a region of 
        interest within the image area (such as a pupil image) are input via 
        the directory argument, then 
        cal.ampthresh is used to select the relevant illuminated pixels when 
        do_ampthresh is set to True.  If instead the user wants to input a 
        mask valid for all frames in directory, the input mask is used instead.

        Parameters
        ----------
        directory : string
            Absolute or relative path of directory containing the .fits files
            of EXCAM frames (non-unity or unity gain).
            Files that are not .fits files will be unaffected. 5 or more
            frames are recommended for a successful result using
            maximum likelihood estimation (MLE), but fewer may yield
            a trustworthy result. A simple estimate even for 1 frame
            is possible after processing:  frames.mean()/framesG1.mean(), where
            framesG1 is the result of processing unity gain summed frames
            with this function.

        bad_pix : array_like, optional
            Bad pixel mask. Bad pixels are True.  Must match shape of full 
            frame.  If None, no bad pixels are
            assumed.  Defaults to None.

        nonlin_path : string, optional
            Full or relative path to nonlinearity relative gain file.
            If None, no nonlinearity correction is used. Defaults to None.

        meta_path : string, optional
            Full or relative path of desired metadata .yaml file.
            If None, it uses metadata.yaml included in proc_cgi_frame. Defaults
            to None.

        flat : array-like, optional
            Flat field array for pixel scaling of image section. 
            This frame is expected to have been pre-processed 
            (bias subtracted, flat field corrected, nonlinearity corrected, 
            and gain divided).  Must match shape of full frame.  If None,
            flat will be all ones and have no effect on the image.
            Defaults to None.

        dark : array-like, optional
            Dark array for dark subtraction if desired (but is not necessary). 
            This is mainly useful for getting
            rid of the fixed-pattern noise before fitting for gain, but things 
            that multiply through the gain register 
            (like clock-induced charge and dark current) do not need to be 
            removed.  A short-time exposure at unity gain (no illumination) 
            should be sufficient for input frames of whatever gain and exposure
            time.  This frame is expected to have been pre-processed 
            (bias subtracted, flat field corrected, nonlinearity corrected, 
            and gain divided). Must match shape of full frame.  
            If None, dark will be all zeros and have no 
            effect on the image. Defaults to None.

        do_ampthresh : bool, optional
            True means the frames will be masked (mask generated by amplitude 
            thresholding) so that only the relevant 
            illuminated part of the image is used for fitting.  If None (which 
            is the default), this is set to  
            True if 'excam_low' and False if 'excam_high' (if no input mask is 
            provided).

        mask : array-like, optional
            Mask array for selecting the pixels to use for analysis for fitting
            for the gain.  False array values for accepting pixels, True for 
            masking them.  Must match shape of image area.  
            Defaults to None, which means no input mask is used.

        desmear_flag : bool
            If True, frame will be desmeared. Useful if frames are illuminated
            enough above the background noise which is mostly not smeared 
            (e.g., perhaps if the mean of an illuminated frame > 5 times the 
            mean of a dark frame).  Defaults to False. 

        Returns
        -------
        frames : array
            Flattened array of all good pixels from all the frames in the
            directory.

        mask : array stack
            A stack of the mask arrays that were used, in the order according 
            to the alpha-numeric ordering of filenames in directory (where 
            numbers take precedence over characters, i.e., '0' would be come 
            before 'a').  These arrays include masking due to the input 
            bad_pix as well.  This may be useful for examining by eye how 
            close a mask created by this function is to the desired mask.
        '''
        if self.cam_mode == 'locam':
            raise ValueError('read_in_excam_files() should only be called if '
                             'cam_mode=0 (EXCAM).')
        if do_ampthresh is not None:
            if not isinstance(do_ampthresh, bool):
                raise TypeError('If ampthresh is specified, it must be '
                                'Boolean.')
            if do_ampthresh and mask is not None:
                raise ValueError('do_ampthresh cannot be True while mask is '
                                 'provided as an input.')
        elif do_ampthresh is None:
            if mask is None and self.cam_mode == 'excam_low':
                do_ampthresh = True
            elif mask is None and self.cam_mode == 'excam_high':
                do_ampthresh = False
            elif mask is not None:
                do_ampthresh = False
        if mask is not None:
            check.twoD_array(mask, 'mask', TypeError)
            if mask.dtype != bool:
                raise TypeError('mask must be an array of bool type.')

        if nonlin_path is not None:
            nonlin_path = abs_or_rel_path(nonlin_path)
        else:
            nonlin_path = Path(here, '..', 'util', 'testdata',
                'ut_nonlin_array_ones.txt') # does no corrections
        if meta_path is not None:
            meta_path = abs_or_rel_path(meta_path)
        else:
            # identical to the one in proc_cgi_frame
            meta_path = str(Path(here, '..', 'util', 'metadata.yaml'))
        if bad_pix is None:
            meta = MetadataWrapper(meta_path)
            bad_pix = np.zeros((meta.frame_rows,meta.frame_cols))

        # frametime below doesn't matter or affect any of the processing, so
        # we just set it to 1
        # input checks are done by Process class
        fwc_em_e = self.config_dict['fwc_em']
        fwc_pp_e = self.config_dict['fwc']
        proc = Process(bad_pix=bad_pix,
                        eperdn=self.eperdn,
                        fwc_em_e=fwc_em_e, fwc_pp_e=fwc_pp_e,
                        bias_offset=self.bias_offset,
                        em_gain=self.com_gain, exptime=1,
                        nonlin_path=nonlin_path,
                        meta_path=meta_path, dark=dark, flat=flat, 
                        sat_thresh=self.sat_thresh, 
                        plat_thresh=self.plat_thresh,
                        cosm_filter=self.cosm_filter, cosm_box=self.cosm_box,
                        cosm_tail=self.cosm_tail, desmear_flag=desmear_flag,
                        rowreadtime=self.rowreadtime)
        framelist = []
        mask_stack = []
        directory = abs_or_rel_path(directory)

        for file in sorted(os.listdir(directory)):
            if file.endswith('fits'):
                f = os.path.join(directory, file)
                d = fits.getdata(f)
                _, _, _, _, f0, b0, _ = proc.L1_to_L2a(d)
                f1, b1, _ = proc.L2a_to_L2b(f0, b0)
                f1 = proc.meta.slice_section(f1, 'image')
                b1 = proc.meta.slice_section(b1, 'image')
        
                # to undo the division by gain in L2a_to_L2b()
                frame = f1*self.com_gain
                
                if frame[frame > 0].size == 0:
                    raise Exception('Frame has no electron counts greater '
                                    'than 0.')

                # get relevant illuminated pixels
                if mask is not None:
                    if mask.shape != f1.shape:
                        raise ValueError('mask must have same shape as image '
                                         'area of frames.')
                    b1 = np.logical_or(b1, mask)
                elif do_ampthresh: #mask would be None here
                    nBin = 2 # starting point
                    mask = np.ones_like(frame).astype(bool)
                    if self.ampthresh_mask_thresh >= np.floor(mask.size/2):
                        raise ValueError('ampthresh_mask_thresh bigger than '
                                         'half the frame size.') 
                    # mask out the negative-valued pixels to extract signal 
                    ma_frame = np.ma.masked_array(frame, (frame < 0))
                    counter = 0
                    while (mask[mask==True].size < self.ampthresh_mask_thresh 
                      or mask[mask==False].size < self.ampthresh_mask_thresh):
                        try:
                            mask = ampthresh(ma_frame, nBin=nBin)
                            mask = mask.filled(0)
                        except:
                            pass
                        nBin += 1
                        counter += 1
                        if counter > 1000:
                            raise Exception('Mask could not be found that '
                                            'passes ampthresh_mask_thresh. '
                                            'Reduce ampthresh_mask_thresh.')
                    
                    smooth_m = mask_smoothing(mask, self.binsize, 
                                              self.smoothing_threshold)
                    temp_t = self.smoothing_threshold
                    while (smooth_m[smooth_m==True].size < 
                                    self.ampthresh_mask_thresh) or \
                        (smooth_m[smooth_m==False].size < 
                                    self.ampthresh_mask_thresh):
                        temp_t *= 0.9 #decrease it some
                        smooth_m = mask_smoothing(mask, self.binsize, 
                                              temp_t)
                        if temp_t <= np.finfo(float).eps:
                            raise Exception("Cannot find a suitable mask for "
                                            "illuminated frame.")
                  
                    mask = (smooth_m==False)
                    b1 = np.logical_or(b1, mask)
                
                mask_stack.append(b1)
                ff = np.ma.masked_array(frame, mask=b1.astype(bool))
                f1 = ff.astype(float).filled(np.nan)
                framelist.append(f1.ravel())
        if len(framelist) > 1:
            framelist = np.stack(framelist)
        s = np.ravel(framelist)
        s_out_ind = np.where(~np.isnan(s))
        frames = s[s_out_ind]
        mask_stack = np.stack(mask_stack)
        return frames, mask_stack


    def EM_gain_fit(self, frames, framesG1=None, cut_option=None, 
                    num_cuts=9):
        '''
        Given the input called frames 
        (an array of the processed frame or frames to fit), this
        function performs maximum likelihood estimation (MLE) to determine
        which values of ungained mean electron counts (mean of Poisson
        distribution) and EM gain (for gamma distribution)
        maximize the likelihood of the expected probability distribution
        for the frame or frames.  The distribution assumed is the composition
        of the gamma distribution with the Poisson distribution convolved with
        a normal distribution for the read noise.  See the doc string of
        EM_gain_tools._EM_gain_fit_conv() for more details.

        For high-gain frames, a series of fits at various cuts is made by 
        default, and the most stable gain value is assumed to be the best-fit 
        value.  This avoids the biasing effect of CIC in the gain register 
        on the fitted gain.  
        
        The fitted value for the return 
        e_mean may not be as accurate because of the region of the histogram 
        that is sampled, but the gain will be reliable. 

        Parameters
        ----------
        frames : array-like
            Array containing data from a frame or frames with non-unity EM
            gain.

        framesG1 : array-like, optional
            Array containing data from a frame or frames with unity EM
            gain.  If not applicable (as in EXCAM high-gain), use None.
            Defaults to None.

        cut_option : string or None, optional
            If 'high', a series of cuts is made to try to find the first 
            instance of a local minimum of change in gain over the cuts, 
            using MLE. The fitted gain can change 
            substantially depending on where the histogram is cut 
            if the applied gain is high (see description 
            below in num_cuts).  'high' is intended for a series of cuts 
            that expects a high gain, regardless of self.cam_mode. 

            If 'low', a series of cuts is made to try to find the first local 
            minimum in the fitted gain, using MLE.  Cuts can be useful to avoid 
            any effects not accounted for 
            by the PDF (fixed-pattern noise or non-uniform illumination, 
            for example).  'low' is intended for a series of cuts that expects 
            a low gain, regardless of self.cam_mode.

            If 'no', no cuts will be performed.  Instead, the gain will be 
            obtained by dividing the mean of the input frames by the mean 
            electron count level (which is the mean of the input framesG1 if 
            available, or the expected electron count level from the expected 
            clock-induced charge (CIC) and dark current from excam_config.yaml 
            if self.cam_mode is 'excam_high'). 
            This method is often sufficient for low and mid gain
            values. 
            
            If None, this parameter is set to 'high' if self.cam_mode is 
            'excam_high' and 'no' if self.cam_mode is 'excam_low' or 'locam'.
            'low' is intended as a backup method and often takes longer, but it
            can be useful if a non-ideal mask 
            selecting the illuminated pixels is used.  Defaults to None. 

        num_cuts : int, optional
            Relevant only if cut_option is not 'no'.  This is the number of 
            cuts made.  The more that are made, the more precisely the gain
            can be determined.  Must be >= 3.  Defaults to 9.    

            Notes:
            If clock-induced charge (CIC) is created in one 
            or more gain stages, the PDF used for MLE is not 
            longer applicable over the usual domain since the PDF does not 
            account for this effect, and this effect can result in very 
            different behavior detector to detector.  This "partial CIC" is 
            more likely to be present at high voltage in the gain register, 
            which is the case when EM gain is high.  However, the electron 
            counts for charges that entered the beginning of the gain register
            are still higher than the counts for a CIC spontaneously created in
            a given gain stage since it undergoes only a subset of the gain 
            register stages.  In order to make the best cut 
            in the histogram for doing MLE, this function will examine the 
            fit results of the number of evenly-spaced cuts specified by 
            num_cuts, ranging between 0 and the first entry on the 
            horizontal axis at which the histogram frequency reaches 
            self.min_freq.  The two cuts between which the first instance of 
            a local minimum in change of fitted gain 
            is selected, and the cut of these two which has the smaller fitted 
            lambda is selected for return values 
            (where lambda is the fitted mean electron count for the frames).
            A local minimum in lambda over the cuts is expected since the 
            cuts begin at 0, where the read noise dominates, and a higher cut 
            favors a smaller fit lambda.  As the cut value increases, higher 
            counts are favored and thus the fitted lambda should increase. The
            fitted gain also should increase with higher cuts and may reach the 
            upper bound of the gain range used for MLE.  Only the fitted gains 
            that are less than the first local maximum in fitted gain are used,
            and the global maximum is used for this boundary instead if there 
            is no local maximum.  

        Returns
        -------
        EMgain : float
            EM gain value found via MLE.

        e_mean : float
            Ungained mean electron counts found via MLE.

        success : bool
            If success is True, the optimization process for MLE was
            successful.  If success is False, the optimization process for MLE
            was unsuccessful, and more frames are probably needed.
            If cut_option is 'no', this is irrelevant and 
            will be None.

        lam_b_g : tuple
            Tuple of 3 values: 
            (lambda lower bound, lambda initial guess, lambda upper bound). 
            This is just for information.  For example, suppose the output 
            e_mean = 300e- and lam_b_g = (1, 256, 300).  The output e_mean 
            is suspiciously close to one of the bounds, and this function 
            may need to be run again with a larger 
            value for lambda_factor in the file for gain_fit_path (same as 
            self.lambda_factor).  If cut_option is 'no', this is irrelevant and 
            will be the tuple (None, None, None).

        gain_b_g : tuple
            Tuple of 3 values: 
            (gain lower bound, gain initial guess, gain upper bound). 
            This is just for information.  For example, suppose the 
            output EMgain = 40 and gain_b_g = (1, 5, 40).  
            EMgain is suspiciously close to one of the bounds, and 
            this function may need to be run again with a larger 
            value for gain_factor in the file for gain_fit_path (same as 
            self.gain_factor).  If cut_option is 'no', this is irrelevant and 
            will be the tuple (None, None, None).
        '''
        if cut_option is None:
            if self.cam_mode == 'excam_high':
                cut_option = 'high'
            elif self.cam_mode == 'excam_low' or self.cam_mode == 'locam':
                cut_option = 'no'
        check.string(cut_option, 'cut_option', TypeError)
        if cut_option != 'high' and cut_option != 'low' and cut_option != 'no':
            raise ValueError('cut_option must be \'high\', \'low\', \'no\', '
                             'or None.')
        check.positive_scalar_integer(num_cuts, 'num_cuts', 
                                      TypeError)
        if num_cuts < 3:
            raise ValueError('num_cuts should be at least 3.')
        
        if self.cam_mode == 'locam':
            if cut_option is None:
                cut_option = False
            if framesG1 is not None:
                fluxe = framesG1.mean()
            else:
                fluxe = frames.mean()/self.com_gain
            gain = frames.mean()/fluxe
            # multiply by factor in case post facto gain was higher than max
            # commandable gain
            gmax = self.config_dict['g_max_comm']*self.gmax_factor
            rn = self.rn
            mu = self.rn_mean
            darke = self.config_dict['darke']
            tfr = self.config_dict['tframe']
            cic = self.config_dict['cic']
            fluxe_dark = (darke*tfr + cic)*self.num_summed_frames
            lthresh = 0 # not relevant for locam
            Nem = self.Nem
            num_summed_frames=self.num_summed_frames

        elif self.cam_mode == 'excam_low':
            if cut_option is None:
                cut_option = False
            if framesG1 is not None:
                fluxe = framesG1.mean()
            else:
                fluxe = frames.mean()/self.com_gain
            gain = frames.mean()/fluxe
            # multiply by factor in case post facto gain was higher than max
            # commandable gain
            gmax = self.config_dict['gmax']*self.gmax_factor
            rn = self.rn
            mu = self.rn_mean
            fluxe_dark = 0 # not actually used for EXCAM case
            lthresh = self.lthresh
            Nem = self.config_dict['Nem']
            num_summed_frames=None

        elif self.cam_mode == 'excam_high':
            if cut_option is None:
                cut_option = True
            if framesG1 is not None:
                raise ValueError('framesG1 must be None in \'excam_high\' '
                                 'mode.')
            if frames.min() > 0:
                raise Exception('Frames should have negative values in e- '
                                'due to read noise for high-gain frames. '
                                'Perhaps more frames are needed for better '
                                'statistics.')
            cic = self.config_dict['cic']
            # these are dark frames; would include dark current if we had 
            # exposure time; cic alone gets us ballpark initial guess, though
            darke = self.config_dict['darke']
            fluxe = cic + darke*self.exptime
            gain = frames.mean()/fluxe #self.com_gain
            # multiply by factor in case post facto gain was higher than max
            # commandable gain
            gmax = self.config_dict['gmax']*self.gmax_factor
            rn = self.rn
            mu = self.rn_mean
            fluxe_dark = 0 # not actually used for EXCAM case
            lthresh = self.lthresh
            Nem = self.config_dict['Nem']
            num_summed_frames=None

        if cut_option == 'high':
            cut = 1 #lowest cut at 1, near read-noise peak
            y_vals, bin_edges = np.histogram(frames, 
                                bins=int((frames.max() - frames.min())))
            x_vals = bin_edges[:-1]
            if self.min_freq >= y_vals.max():
                raise ValueError('min_freq must be less than maximum '
                                    'frequency in histogram.')
            # index of last x_vals value with min_freq frequency
            elements = np.argwhere(y_vals == self.min_freq)
            if elements.size != 0:
                cut_cutoff = elements[-1][0]
            else:
                raise Exception('min_freq is not found in histogram. '
                                'Select a higher value for min_freq.')
            if cut == x_vals[cut_cutoff]:
                raise ValueError('min_freq occured at 0 on histogram, so '
                                    'a series of cuts starting at 0 cannot '
                                    'be made. min_freq should be changed.')
            cuts = np.linspace(cut, x_vals[cut_cutoff], num_cuts)   
            if x_vals[cut_cutoff] == x_vals.max():
                cuts = cuts[:-1] 
            gains = np.array([])
            e_means = np.array([])
            successes = np.array([])
            for c in cuts:
                res = _EM_gain_fit_conv(frames, fluxe, gain, gmax, rn, mu,
                                        fluxe_dark, lthresh, 
                                        self.divisor, self.gain_factor,
                                        self.lambda_factor, self.tol, c,
                                        num_summed_frames=num_summed_frames,
                                        Nem=Nem,
                                        diff_tol=self.diff_tol)
    
                EMgain = res[0]
                e_mean = res[1]
                success = res[2]
                lam_b_g = res[3] #doesn't change with varying cut
                gain_b_g = res[4] #doesn't change with varying cut
                gains = np.append(gains, EMgain)
                e_means = np.append(e_means, e_mean)
                successes = np.append(successes, success)
            # max below favors "left" one of the two
            local_max_bool = np.logical_and(gains[1:-1] > gains[:-2],
                    gains[1:-1] >= gains[2:])
            if local_max_bool[local_max_bool==True].size > 0:
                gain_lmax = np.max(gains[1:-1][local_max_bool])
                gmaxind = 1 + np.argwhere(gains[1:-1]==gain_lmax)[0][0]
            else: #go with global max 
                gmaxind = gains.size
            ga = gains[:gmaxind]
            e_mea = e_means[:gmaxind]
            if ga.size == 0:
                raise Exception('Max in fitted gain is at first cut, '
                                'unexpected for a high gain. Try a '
                                'different value for cut_option.')
            g_diff = np.roll(ga, -1) - ga
            if g_diff.size <= 1:
                ind = 0
            else:
                #leave off last index b/c that wraps around
                ab_diff = np.abs(g_diff[:-1])

                local_diff_bool = np.logical_and(ab_diff[1:-1] <= ab_diff[:-2],
                        ab_diff[1:-1] < ab_diff[2:])
                if local_diff_bool[local_diff_bool==True].size > 0:
                    # first local min in the difference in gain over the cuts
                    diffmin = np.argmax(local_diff_bool) # gets first "True"
                    ind = 1 + diffmin
                else: # if no local min, use global min
                    ind = np.argmin(ab_diff)
                # e_mea is always 1 bigger in size than ab_diff, e_mea[ind+1] 
                # exists
                if e_mea[ind+1] <= e_mea[ind]:
                    ind += 1
                    
            EMgain = ga[ind]
            e_mean = e_mea[ind]
            success = bool(successes[ind])
            
        elif cut_option == 'low':
            cut = 1 #lowest cut at 1, near read-noise peak
            y_vals, bin_edges = np.histogram(frames, 
                                bins=int((frames.max() - frames.min())))
            x_vals = bin_edges[:-1]
            if self.min_freq >= y_vals.max():
                raise ValueError('min_freq must be less than maximum '
                                    'value in histogram.')
            # index of last x_vals value with min_freq frequency
            elements = np.argwhere(y_vals == self.min_freq)
            if elements.size != 0:
                cut_cutoff = elements[-1][0]
            else:
                raise Exception('min_freq is not found in histogram. '
                                'Select a higher value for min_freq.')
            if cut == x_vals[cut_cutoff]:
                raise ValueError('min_freq occured at 0 on histogram, so '
                                    'a series of cuts starting at 0 cannot '
                                    'be made. min_freq should be changed.')
            cuts = np.linspace(cut, x_vals[cut_cutoff], num_cuts)   
            if x_vals[cut_cutoff] == x_vals.max():
                cuts = cuts[:-1] 
            gains = np.array([])
            e_means = np.array([])
            successes = np.array([])
            for c in cuts:
                res = _EM_gain_fit_conv(frames, fluxe, gain, gmax, rn, mu,
                                        fluxe_dark, lthresh, 
                                        self.divisor, self.gain_factor,
                                        self.lambda_factor, self.tol, c,
                                        num_summed_frames=num_summed_frames,
                                        Nem=Nem,
                                        diff_tol=self.diff_tol)
    
                EMgain = res[0]
                e_mean = res[1]
                success = res[2]
                lam_b_g = res[3] #doesn't change with varying cut
                gain_b_g = res[4] #doesn't change with varying cut
                gains = np.append(gains, EMgain)
                e_means = np.append(e_means, e_mean)
                successes = np.append(successes, success)

            ga = gains
            e_mea = e_means
            local_min_bool = np.logical_and(ga[1:-1] < ga[:-2],
                    ga[1:-1] <= ga[2:])
            if local_min_bool[local_min_bool==True].size > 0:
                # smallest-cut local min
                gmin = np.min(ga[1:-1][local_min_bool])
                gminind = 1 + np.argwhere(ga[1:-1]==gmin)[0][0]
            else: # no local min
                raise Exception('No local minimum in gain.  Try increasing '
                                'num_cuts (or try another cut method).')
            EMgain = ga[gminind]
            e_mean = e_mea[gminind]
            success = bool(successes[gminind])

        elif cut_option == 'no':
            # just use values that came from the gain estimate
            EMgain = gain
            e_mean = fluxe
            # the other outputs are irrelevant in this case
            success = None
            lam_b_g = (None, None, None)
            gain_b_g = (None, None, None)

        return EMgain, e_mean, success, lam_b_g, gain_b_g
    

def mask_smoothing(img, binsize, bin_percent=0.8):
    """The mask provided by cal.ampthresh.ampthresh.ampthresh() may include 
    pixels outside of the region of interest due to the effect of fixed-pattern
    noise and/or relatively low exposure times.  This function corrects the 
    mask by masking all pixels that do not have a high enough local spatial 
    density of unmasked pixels.  

    Parameters
    ----------
    img : 2-D array
        Mask to be corrected.
    binsize : int > 0 or None
        The mask is broken up into square bins of side length binsize for 
        the purpose of examining the local density of unmasked pixels. Should 
        be > 0.  
    bin_percent : float, optional
        The fraction of unmasked pixels within each square bin must be 
        bigger than bin_percent.  Defaults to 0.8.

    Returns
    -------
    corrected_img : 2-D array
        Corrected mask.
    """

    check.twoD_array(img, 'img', TypeError)
    check.positive_scalar_integer(binsize, 'binsize', TypeError)
    check.real_nonnegative_scalar(bin_percent, 'bin_percent', TypeError)
    if bin_percent >= 1:
        raise ValueError('bin_percent must be < 1.')

    # ensures there is a bin that runs all the way to the end
    if np.mod(len(img), binsize) == 0:
        row_bins = np.arange(0, len(img)+1, binsize)
    else:
        row_bins = np.arange(0, len(img), binsize)
    # If len(img) not divisible by binsize, this makes last bin of size
    # binsize + the remainder
    row_bins[-1] = len(img)

    # same thing for columns now
    if np.mod(len(img[0]), binsize) == 0:
        col_bins = np.arange(0, len(img[0])+1, binsize)
    else:
        col_bins = np.arange(0, len(img[0]), binsize)
    col_bins[-1] = len(img[0])

    # initializing
    loc_ill = (np.zeros([len(row_bins)-1, len(col_bins)-1])).astype(float)

    corrected_img = (np.zeros([len(img), len(img[0])])).astype(float)

    for i in range(len(row_bins)-1):
        for j in range(len(col_bins)-1):
            bin_region = img[int(row_bins[i]):int(row_bins[i+1]),
                                int(col_bins[j]):int(col_bins[j+1])]
            loc_ill[i,j]=np.sum(bin_region)/bin_region.size
            if loc_ill[i,j] > bin_percent:
                corrected_img[int(row_bins[i]):int(row_bins[i+1]),
                int(col_bins[j]):int(col_bins[j+1])] = 1
            else:
                corrected_img[int(row_bins[i]):int(row_bins[i+1]),
                int(col_bins[j]):int(col_bins[j+1])] = 0

    return corrected_img