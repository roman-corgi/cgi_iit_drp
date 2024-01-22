# -*- coding: utf-8 -*-
"""Basic detector calibration and processing script.

B Nemati and S Miller - UAH - 16-Apr-2020
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from proc_cgi_frame.gsw_process import Process, median_combine, mean_combine
from proc_cgi_frame.read_metadata import Metadata

class ProcessFitsException(Exception):
    """Exception class for process_fits module."""


class FitsDir(object):
    """Read directory containing fits files and sort names into list.

    Parameters
    ----------
    path : str
        Full path of directory containing fits files.

    Attributes
    ----------
    path : :class:`pathlib.PosixPath`
        Full path of directory containing fits files.
    fits_list : list
        Sorted list of fits filenames in directory.

    S Miller - UAH - 13-Feb-2019

    """

    def __init__(self, path):
        self._path = path

    @property
    def path(self):
        """Check for existance of path."""
        try:
            path = Path(self._path)
        except Exception:
            raise

        if not path.is_dir():
            raise ProcessFitsException('No such directory: '
                                       '{:}'.format(self._path))

        return path

    @property
    def fits_list(self):
        """Organize filenames in directory into list of only fits filenames."""
        dir_list = os.listdir(str(self.path))
        dir_range = range(len(dir_list))
        out = [dir_list[i] for i in dir_range if dir_list[i].endswith('.fits')]

        if not out:
            raise ProcessFitsException('No fits files in directory')
        out.sort()

        return out



def imagesc(data, title=None, vmin=None, vmax=None, cmap='viridis',
            aspect='equal', colorbar=True):
    """Plot a scaled colormap."""
    fig, ax = plt.subplots()
    im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, aspect=aspect)

    if title:
        ax.set_title(title)
    if colorbar:
        fig.colorbar(im, ax=ax)

    return fig, ax


if __name__ == '__main__':
    here = os.path.abspath(os.path.dirname(__file__))

    # Get filepaths of frames
    use_sim = False
    if use_sim:
        brights_path = Path(here, 'data', 'sim', 'brights')
        darks_path = Path(here, 'data', 'sim', 'darks')
    else:
        brights_path = Path(here, 'data', 'brights')
        darks_path = Path(here, 'data', 'darks')

    # For images of the geometry of the different frame formats, see the docs
    # folder.
    # --Metadata.yaml is valid for SCI format frames and is the default value
    # for the Process class.  It also includes the ENG format frame geometry
    # for the SCIENCE/ACQUIRE and TRAP PUMPING sequences as broken off from the
    # SCI part.
    # --For the full ENG format frame for the SCIENCE/ACQUIRE and TRAP PUMPING
    # sequences, see metadata_eng.yaml.
    # --Metadata_eng_em.yaml is valid for the ENG format frame under the
    # ENGINEERING_EM sequence.
    # --Metadata_eng_conv.yaml is valid for the ENG format frame under the
    # ENGINEERING_CONV sequence.

    # Metadata file path
    meta_path = Path(here, 'proc_cgi_frame', 'metadata.yaml')
    # Nonlin path
    nonlin_path = Path(here, 'proc_cgi_frame', 'nonlin_sample.csv')

    # Read frames
    brights = np.stack([fits.getdata(Path(brights_path, f))
                        for f in FitsDir(brights_path).fits_list])
    darks = np.stack([fits.getdata(Path(darks_path, f))
                      for f in FitsDir(darks_path).fits_list])

    # Set up inputs
    meta = Metadata(meta_path)
    im_rows = meta.geom['image']['rows']
    im_cols = meta.geom['image']['cols']
    fr_rows = meta.frame_rows
    fr_cols = meta.frame_cols
    flat = np.ones([im_rows, im_cols])
    flat[-100:, :100] = 0.1  # Test lower left corner
    bad_pix = np.zeros([im_rows, im_cols])
    bad_pix[-100:, -100:] = 1  # Test lower right corner
    bias_offset = 0
    eperdn = 6
    fwc_em_e = 90000
    fwc_pp_e = 50000
    em_gain = 4000
    exptime = 1
    # Specify parameters for removal of cosmic rays if ones different from the
    # default values are desired. Below are the default values used:
    sat_thresh = 0.99
    plat_thresh = 0.85
    cosm_filter = 2

    # Using this as a proxy; actual master dark calibrated with
    # cal.calibrate_darks and assembled by cal.master_dark (details
    # further below)
    master_dark = np.zeros((im_rows, im_cols))

    # Should do processing of a real flat taken by a detector
    # (and for simulated flat, run it through
    # emccd_detect before processing), but image gets very washed out.
    # To do this right, would need to do mean_combine() of several
    # individually-processed flats
    # frames to get a good flat to use to correct the final image.
    # So in lieu of that, we just use here an unprocessed array of
    # ones for flat.  Below are the commands to use for processing
    # a (single) flat, though.  fl_flux is assumed to have no non-uniformity:
    # proc_flat_f = Process(fixedbp, eperdn, fwc_em_e, fwc_pp_e, bias_offset,
    #             em_gain, exptime, nonlin_path, meta_path, master_dark)
    # fl_flux = np.ones_like(frame_full)
    # flat_im, fl_b, _, _, _, _, _ = proc_flat_f.L1_to_L2a(fl)
    # flat, fl_bp, _ = proc_flat_f.L2a_to_L2b(flat_im, fl_b)

    # process observation frames
    proc = Process(bad_pix, eperdn, fwc_em_e, fwc_pp_e, bias_offset,
                   em_gain, exptime, nonlin_path, meta_path, master_dark, flat)
    # for a single image, brights[0]:
    i0, b0, _, _, _, _, _ = proc.L1_to_L2a(brights[0])
    image, bad_mask, _ = proc.L2a_to_L2b(i0, b0)

    # for a collection of images:
    i_br = []
    b_br = []
    for br in brights:
        i0, b0, _, _, _, _, _ = proc.L1_to_L2a(br)
        i1, b1, _ = proc.L2a_to_L2b(i0, b0)
        i_br.append(i1)
        b_br.append(b1)
        pass

    med_image, med_bad_mask = median_combine(i_br, b_br)
    mean_image, mean_bad_mask, _, _ = mean_combine(i_br, b_br)

    ########################################
    # For calibrating a master dark, the latter outputs are useful (as used in
    # calibrate_darks.calibrate_darks_lsq() in the calibration repository).
    # In lieu of real dark frames, we will use simulated proxy ones
    full_dark = np.zeros((fr_rows, fr_cols))
    full_flat = np.ones((fr_rows, fr_cols))
    # Could put these through emccd_detect first to make them more realistic
    # No bad pixels before master dark calibrated
    full_bad_pix = np.zeros_like(full_dark)
    proc_dark = Process(full_bad_pix, eperdn, fwc_em_e, fwc_pp_e, bias_offset,
                        em_gain, exptime, nonlin_path,
                        meta_path, full_dark, full_flat,
                        sat_thresh=sat_thresh, plat_thresh=plat_thresh,
                        cosm_filter=cosm_filter)
    dark_frames = []
    bp_frames = []
    for i in range(10):
        _, _, _, _, d0, bp0, _ = proc_dark.L1_to_L2a(full_dark)
        d1, bp1, _ = proc_dark.L2a_to_L2b(d0, bp0)
        d1 *= em_gain # undo the gain division for master dark creation process
        dark_frames.append(d1)
        bp_frames.append(bp1)

    # The last output of mean_combine() are useful for calibrate_darks
    # module in the calibration repository:
    mean_frame, _, mean_num_good_fr, _ = mean_combine(dark_frames, bp_frames)
    #######################################

    plot_images = True
    if plot_images:
        # Plot single image
        m = np.ma.masked_array(image, bad_mask)
        imagesc(m.filled(0), 'Masked Image')
        imagesc(bad_mask, 'Bad Mask', vmin=0, vmax=1)

        # Plot median combined
        m_med = np.ma.masked_array(med_image, med_bad_mask)
        imagesc(m_med.filled(0), 'Masked Median Combined Image')
        imagesc(med_bad_mask, 'Median Combined Bad Mask', vmin=0, vmax=1)

        # Plot mean combined
        m_mean = np.ma.masked_array(mean_image, mean_bad_mask)
        imagesc(m_mean.filled(0), 'Masked Mean Combined Image')
        imagesc(mean_bad_mask, 'Mean Combined Bad Mask', vmin=0, vmax=1)

        plt.show()
