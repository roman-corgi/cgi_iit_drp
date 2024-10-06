# -*- coding: utf-8 -*-
"""Remove saturated cosmics from image area of frame."""

import numpy as np
from scipy.ndimage import median_filter
from .read_metadata import Metadata

def remove_cosmics(image, fwc, sat_thresh, plat_thresh, cosm_filter, cosm_box,
                   cosm_tail, meta=None, mode='image'):
    """Identify and remove saturated cosmic ray hits and tails.

    Use sat_thresh (interval 0 to 1) to set the threshold above which cosmics
    will be detected. For example, sat_thresh=0.99 will detect cosmics above
    0.99*fwc.

    Use plat_thresh (interval 0 to 1) to set the threshold under which cosmic
    plateaus will end. For example, if plat_thresh=0.85, once a cosmic is
    detected the beginning and end of its plateau will be determined where the
    pixel values drop below 0.85*fwc.

    Use cosm_filter to determine the smallest plateaus (in pixels) that will
    be identified. A reasonable value is 2.

    Parameters
    ----------
    image : array_like, float
        Image area of frame (bias of zero).
    fwc : float
        Full well capacity of detector *in DNs*.  Note that this may require a
        conversion as FWCs are usually specified in electrons, but the image
        is in DNs at this point.
    sat_thresh : float
        Multiplication factor for fwc that determines saturated cosmic pixels.
    plat_thresh : float
        Multiplication factor for fwc that determines edges of cosmic plateau.
    cosm_filter : int
        Minimum length in pixels of cosmic plateaus to be identified.
    cosm_box : int
        Number of pixels out from an identified cosmic head (i.e., beginning of
        the plateau) to mask out.
        For example, if cosm_box is 3, a 7x7 box is masked,
        with the cosmic head as the center pixel of the box.
    cosm_tail : int
        Number of pixels in the row downstream of the end of a cosmic plateau
        to mask.  If cosm_tail is greater than the number of
        columns left to the end of the row from the cosmic
        plateau, the cosmic masking ends at the end of the row. Defaults to 10.
    meta : Metadata class instance
        Metadata class instance, which is used to determine whether the
        beginning of a plateau is not in the image area, in which case no 
        cosmic ray masking should occur.  Only relevant when mode is 'full'.
        Defaults to None, in which case masking is allowed anywhere on the 
        input frame.
    mode : string
        If 'image', an image-area input is assumed, and if the input
        tail length is longer than the length to the end of the image-area row,
        the mask is truncated at the end of the row.
        If 'full', a full-frame input is assumed, and if the input tail length
        is longer than the length to the end of the full-frame row, the masking
        continues onto the next row.  Defaults to 'image'.

    Returns
    -------
    mask : array_like, int
        Mask for pixels that have been set to zero.

    Notes
    -----
    This algorithm uses a row by row method for cosmic removal. It first finds
    streak rows, which are rows that potentially contain cosmics. It then
    filters each of these rows in order to differentiate cosmic hits (plateaus)
    from any outlier saturated pixels. For each cosmic hit it finds the leading
    ledge of the plateau and kills the plateau (specified by cosm_filter) and
    the tail (specified by cosm_tail).

    |<-------- streak row is the whole row ----------------------->|
     ......|<-plateau->|<------------------tail---------->|.........

    B Nemati and S Miller - UAH - 02-Oct-2018
    Kevin Ludwick - UAH - 2024

    """
    mask = np.zeros(image.shape, dtype=int)
    if meta is not None:
        if not isinstance(meta, Metadata):
            raise Exception('meta must be an instance of the Metadata class.')
    if meta is not None and mode=='full':
        im_num_rows = meta.geom['image']['rows']
        im_num_cols = meta.geom['image']['cols']
        im_starting_row = meta.geom['image']['r0c0'][0]
        im_ending_row = im_starting_row + im_num_rows
        im_starting_col = meta.geom['image']['r0c0'][1]
        im_ending_col = im_starting_col + im_num_cols
    else:
        im_starting_row = 0
        im_ending_row = mask.shape[0] - 1 # - 1 to get the index, not size
        im_starting_col = 0
        im_ending_col = mask.shape[1] - 1 # - 1 to get the index, not size

    # Do a cheap prefilter for rows that don't have anything bright
    max_rows = np.max(image, axis=1)
    i_streak_rows = (max_rows >= sat_thresh*fwc).nonzero()[0]

    for i in i_streak_rows:
        row = image[i]
        if i < im_starting_row or i > im_ending_row:
            continue
        # Find if and where saturated plateaus start in streak row
        i_begs = find_plateaus(row, fwc, sat_thresh, plat_thresh, cosm_filter)

        # If plateaus exist, kill the hit and the tail
        cutoffs = np.array([])
        ex_l = np.array([])
        if i_begs is not None:
            for i_beg in i_begs:
                if i_beg < im_starting_col or i_beg > im_ending_col:
                    continue
                # implement cosm_tail
                if i_beg+cosm_filter+cosm_tail+1 > mask.shape[1]:
                    ex_l = np.append(ex_l,
                            i_beg+cosm_filter+cosm_tail+1-mask.shape[1])
                    cutoffs = np.append(cutoffs, i+1)
                streak_end = int(min(i_beg+cosm_filter+cosm_tail+1,
                                mask.shape[1]))
                mask[i, i_beg:streak_end] = 1
                # implement cosm_box
                # can't have cosm_box appear in non-image pixels
                st_row = max(i-cosm_box, im_starting_row)
                end_row = min(i+cosm_box+1, im_ending_row+1)
                st_col = max(i_beg-cosm_box, im_starting_col)
                end_col = min(i_beg+cosm_box+1, im_ending_col+1)
                mask[st_row:end_row, st_col:end_col] = 1
                pass

        if mode == 'full' and len(ex_l) > 0:
            mask_rav = mask.ravel()
            for j in range(len(ex_l)):
                row = cutoffs[j]
                rav_ind = int(row * mask.shape[1] - 1)
                mask_rav[rav_ind:rav_ind + int(ex_l[j])] = 1

    return mask


def find_plateaus(streak_row, fwc, sat_thresh, plat_thresh, cosm_filter):
    """Find the beginning index of each cosmic plateau in a row.

    Parameters
    ----------
    streak_row : array_like, float
        Row with possible cosmics.
    fwc : float
        Full well capacity of detector *in DNs*.  Note that this may require a
        conversion as FWCs are usually specified in electrons, but the image
        is in DNs at this point.
    sat_thresh : float
        Multiplication factor for fwc that determines saturated cosmic pixels.
    plat_thresh : float
        Multiplication factor for fwc that determines edges of cosmic plateu.
    cosm_filter : int
        Minimum length in pixels of cosmic plateus to be identified.

    Returns
    -------
    i_begs : array_like, int
        Index of plateau beginnings, or None if there is no plateau.

    """
    # Lowpass filter row to differentiate plateaus from standalone pixels
    # The way median_filter works, it will find cosmics that are cosm_filter-1
    # wide. Add 1 to cosm_filter to correct for this
    filtered = median_filter(streak_row, cosm_filter+1, mode='nearest')
    saturated = (filtered >= sat_thresh*fwc).nonzero()[0]

    if len(saturated) > 0:
        i_begs = np.array([])
        for i in range(len(saturated)):
            i_beg = saturated[i]
            while i_beg > 0 and streak_row[i_beg] >= plat_thresh*fwc:
                i_beg -= 1
            # unless saturated at col 0, shifts forward 1 to plateau start
            if streak_row[i_beg] < plat_thresh*fwc:
                i_beg += 1
            i_begs = np.append(i_begs, i_beg)

        return np.unique(i_begs).astype(int)
    else:
        return None
