# -*- coding: utf-8 -*-
"""Detector geometry metadata."""

import os
from pathlib import Path

import yaml

class ReadMetadataException(Exception):
    """Exception class for read_metadata module."""

# Set up to allow the metadata.yaml in the repo be the default
here = Path(os.path.dirname(os.path.abspath(__file__)))
meta_path = Path(here, 'metadata.yaml')

class Metadata(object):
    """Create masks for different regions of the Roman CGI EXCAM detector.

    Parameters
    ----------
    meta_path : str
        Full path of metadta yaml.

    Attributes
    ----------
    data : dict
        Data from metadata file.
    geom : SimpleNamespace
        Geometry specific data.

    B Nemati and S Miller - UAH - 03-Aug-2018

    """

    def __init__(self, meta_path=meta_path):
        self.meta_path = meta_path

        self.data = self.get_data()
        self.frame_rows = self.data['frame_rows']
        self.frame_cols = self.data['frame_cols']
        self.geom = self.data['geom']

    def get_data(self):
        """Read yaml data into dictionary."""
        with open(self.meta_path, 'r') as stream:
            data = yaml.safe_load(stream)
        return data

    def slice_section(self, frame, key):
        """Slice 2d section out of frame.

        Parameters
        ----------
        frame : array_like
            Full frame consistent with size given in frame_rows, frame_cols.
        key : str
            Keyword referencing section to be sliced; must exist in geom.

        """
        rows, cols, r0c0 = self._unpack_geom(key)

        section = frame[r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols]
        if section.size == 0:
            raise ReadMetadataException('Corners invalid')
        return section

    def _unpack_geom(self, key):
        """Safely check format of geom sub-dictionary and return values."""
        coords = self.geom[key]
        rows = coords['rows']
        cols = coords['cols']
        r0c0 = coords['r0c0']

        return rows, cols, r0c0

    #added in from MetadataWrapper
    def _imaging_area_geom(self):
        """Return geometry of imaging area in reference to full frame."""
        _, cols_pre, _ = self._unpack_geom('prescan')
        _, cols_serial_ovr, _ = self._unpack_geom('serial_overscan')
        rows_parallel_ovr, _, _ = self._unpack_geom('parallel_overscan')
        #_, _, r0c0_image = self._unpack_geom('image')
        fluxmap_rows, _, r0c0_image = self._unpack_geom('image')

        rows_im = self.frame_rows - rows_parallel_ovr
        cols_im = self.frame_cols - cols_pre - cols_serial_ovr
        r0c0_im = r0c0_image.copy()
        r0c0_im[0] = r0c0_im[0] - (rows_im - fluxmap_rows)

        return rows_im, cols_im, r0c0_im

    def imaging_slice(self, frame):
        """Select only the real counts from full frame and exclude virtual.

        Use this to transform mask and embed from acting on the full frame to
        acting on only the image frame.

        """
        rows, cols, r0c0 = self._imaging_area_geom()

        return frame[r0c0[0]:r0c0[0]+rows, r0c0[1]:r0c0[1]+cols]