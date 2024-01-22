"""Utility functions for testing CTC GSW VIs."""
import copy
import os
from pathlib import Path

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np


def convert_fits_to_png(inPath='.', outPath='.'):
    """
    Convert all FITS files in a directory into PNG files.
    
    The stem of the file name stays the same.
    
    Arrays are handled differently depending on their number of dimensions.
    1-D and 2-D arrays are plotted as-is.
    3-D arrays are plotted slice by slice.
    4-D or larger arrays are not plotted and return a warning message.

    Parameters
    ----------
    inPath : str, optional
        The absolute or relative path to the directory with the FITS files
        to convert. The default is '.'.
    outPath : str, optional
        The absolute or relative path to the directory into which to write
        the PNG files. The default is '.'.

    Returns
    -------
    None.

    """
    # Gather list of FITS file names
    fnList = []  # initialize
    for fn in os.listdir(inPath):
        if fn.endswith(".fits") or fn.endswith(".FITS"):
            fnList.append(fn)
    print('* %d FITS files to be converted to PNG. *' % (len(fnList)))
 
    # Convert all FITS files to PNG files
    figNum = 1
    for fn in fnList:

        try:
            data = fits.getdata(os.path.join(inPath, fn))
            fnStem = Path(fn).stem
            
            data = np.array(data, copy=False)  # cast to array
            naxis = len(data.shape)
            
            if (naxis == 1) or (naxis == 2):
                plt.figure(figNum)
                plt.clf()
                plt.imshow(data)
                plt.title(fn)
                plt.gca().invert_yaxis()
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(os.path.join(outPath, fnStem+'.png'))
                plt.pause(1e-2)
                
            if (naxis == 3):
                
                # Plot each slice in the datacute
                for ii in range(data.shape[0]):
                    fnOut = fnStem + ('_slice%dof%d.png' % (ii+1, data.shape[0]))
                    plt.figure(figNum)
                    plt.clf()
                    plt.imshow(np.squeeze(data[ii, :, :]))
                    plt.title(fnOut)
                    plt.gca().invert_yaxis()
                    plt.colorbar()
                    plt.tight_layout()
                    plt.savefig(os.path.join(outPath, fnOut))
                    plt.pause(1e-2)
                
            if (naxis > 3):
                print('Warning: This function can only plot 1-D, 2-D, and 3-D'
                      'arrays. %s has %d dimensions.' %
                      (os.path.join(inPath, fn), naxis))

        except:
            print('Error converting and saving file %s' % os.path.join(inPath, fn))
    
    # Close figure when done
    try:
        plt.close(figNum)
    except:
        pass


def plot_all_fits_in_dir(folder='.', frameRate=1):
    """
    Plot all the FITS files in a directory at the specified rate.

    Parameters
    ----------
    folder : str, optional
        The absolute or relative path to the directory with the FITS files
        to plot. The default is '.'.
    frameRate : TYPE, optional
        The frame display rate in Hertz. The default is 1.

    Returns
    -------
    None.

    """
    # folder = input('Enter the relative or absolute directory of the folder with the FITS files:')
    # frameRate = float(input('Enter the frame rate in Hertz:'))
    pauseTime = 1/frameRate  # time to pause for each figure [seconds]
    
    fnList = []  # initialize
    for fn in os.listdir(folder):
        if fn.endswith(".fits") or fn.endswith(".FITS"):
            fnList.append(fn)
            
    print('* %d FITS files will be plotted. *' % (len(fnList)))
    
    for fn in fnList:

        data = fits.getdata(os.path.join(folder, fn))

        plt.figure(1)
        plt.clf()
        plt.imshow(data)
        plt.title(fn)
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.pause(pauseTime)

