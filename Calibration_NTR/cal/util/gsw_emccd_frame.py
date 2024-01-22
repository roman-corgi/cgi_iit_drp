# -*- coding: utf-8 -*-
"""Process EMCCD images."""

import numpy as np

from .gsw_remove_cosmics import remove_cosmics


class EMCCDFrameException(Exception):
    """Exception class for emccd_frame module."""

class EMCCDFrame:
    """Get data from EMCCD frame and subtract the bias and bias offset.

    Parameters
    ----------
    frame_dn : array_like
        Raw EMCCD full frame (DN).
    meta : instance
        Instance of Metadata class containing detector metadata.
    fwc_em : float
        Detector EM gain register full well capacity (DN).
    fwc_pp : float
        Detector image area per-pixel full well capacity (DN).
    em_gain : float
        Gain from EM gain register, >= 1 (unitless).
    bias_offset : float
        Median number of counts in the bias region due to fixed non-bias noise
        not in common with the image region.  Basically we compute the bias
        for the image region based on the prescan from each frame, and the
        bias_offset is how many additional counts the prescan had from extra
        noise not captured in the master dark fit.  This value is subtracted
        from each measured bias.  Units of DN.


    Attributes
    ----------
    image : array_like
        Image section of frame (DN).
    prescan : array_like
        Prescan section of frame (DN).
    al_prescan : array_like
        Prescan with row numbers relative to the first image row (DN).
    frame_bias : array_like
        Column vector with each entry the median of the prescan row minus the
        bias offset (DN).
    bias : array_like
        Column vector with each entry the median of the prescan row relative
        to the first image row minus the bias offset (DN).
    frame_bias0 : array_like
        Total frame minus the bias (row by row) minus the bias offset (DN).
    image_bias0 : array_like
        Image area minus the bias (row by row) minus the bias offset (DN).

    S Miller - UAH - 16-April-2019

    """

    def __init__(self, frame_dn, meta, fwc_em, fwc_pp, em_gain, bias_offset):
        self.frame_dn = frame_dn
        self.meta = meta
        self.fwc_em = fwc_em
        self.fwc_pp = fwc_pp
        self.em_gain = em_gain
        self.bias_offset = bias_offset

        # Divide frame into sections
        try:
            self.image = self.meta.slice_section(self.frame_dn, 'image')
            self.prescan = self.meta.slice_section(self.frame_dn, 'prescan')
        except Exception:
            raise EMCCDFrameException('Frame size inconsistent with metadata')

        # Get the part of the prescan that lines up with the image, and do a
        # row-by-row bias subtraction on it
        i_r0 = self.meta.geom['image']['r0c0'][0]
        p_r0 = self.meta.geom['prescan']['r0c0'][0]
        i_nrow = self.meta.geom['image']['rows']
        # select the good cols for getting row-by-row bias
        st = self.meta.geom['prescan']['col_start']
        end = self.meta.geom['prescan']['col_end']
        # over all prescan rows
        medbyrow_tot = np.median(self.prescan[:,st:end], axis=1)[:, np.newaxis]
        # prescan relative to image rows
        self.al_prescan = self.prescan[(i_r0-p_r0):(i_r0-p_r0+i_nrow), :]
        medbyrow = np.median(self.al_prescan[:,st:end], axis=1)[:, np.newaxis]

        # Get data from prescan (image area)
        self.bias = medbyrow - self.bias_offset
        self.image_bias0 = self.image - self.bias

        # over total frame
        self.frame_bias = medbyrow_tot - self.bias_offset
        self.frame_bias0 = self.frame_dn[p_r0:, :] -  self.frame_bias


    def remove_cosmics(self, sat_thresh, plat_thresh, cosm_filter):
        """Fix cosmic tails, get cosmic and tail masks.

        Parameters
        ----------
        sat_thresh : float
            Multiplication factor for fwc that determines saturated cosmic
            pixels.
        plat_thresh : float
            Multiplication factor for fwc that determines edges of cosmic
            plateau.
        cosm_filter : int
            Minimum length in pixels of cosmic plateus to be identified.

        Returns
        -------
        mask : array_like, int
            Image-area mask for pixels that have been set to zero.
        mask_full : array_like, int
            Full-frame mask for pixels that have been set to zero.

        """
        # pick the FWC that will get saturated first, depending on gain
        sat_fwc = sat_thresh*min(self.fwc_em, self.fwc_pp*self.em_gain)

        # threshold the frame to catch any values above sat_fwc --> this is
        # mask 1
        m1 = (self.image_bias0 >= sat_fwc)
        # run remove_cosmics() with fwc=fwc_em since tails only come from
        # saturation in the gain register --> this is mask 2
        m2 = remove_cosmics(image=self.image_bias0,
                            fwc=self.fwc_em,
                            sat_thresh=sat_thresh,
                            plat_thresh=plat_thresh,
                            cosm_filter=cosm_filter,
                            )
        # same thing, but now making masks for full frame (for calibrate_darks)

        # threshold the frame to catch any values above sat_fwc --> this is
        # mask 1
        m1_full = (self.frame_bias0 >= sat_fwc)
        # run remove_cosmics() with fwc=fwc_em since tails only come from
        # saturation in the gain register --> this is mask 2
        m2_full = remove_cosmics(image=self.frame_bias0,
                            fwc=self.fwc_em,
                            sat_thresh=sat_thresh,
                            plat_thresh=plat_thresh,
                            cosm_filter=cosm_filter,
                            )

        # OR the two masks together and return
        return np.logical_or(m1, m2), np.logical_or(m1_full, m2_full)
