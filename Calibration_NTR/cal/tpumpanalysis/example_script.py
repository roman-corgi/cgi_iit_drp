import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cal.tpumpanalysis.tpump_final import tpump_analysis

if __name__ == '__main__':
    here = os.path.abspath(os.path.dirname(__file__))
    # To run the test data that was made for unit tests, see ut_tpump_final.py
    # for details.  generate_test_data.py created these data sets, and the
    # default folder it outputs to is test_data.  It can create full-sized
    # EDU frames or smaller frames using the corresponding metadata file
    # metadata_test.yaml.  These are smaller frames for testing and for
    # shorter run times.
    #We can specify the directory for the input data to be run:
    base_dir = Path(here, 'test_data')
    meta_path = Path(here, '..', 'util', 'metadata_test.yaml')
    # for full-sized EDU frames, use the commented-out line below:
    #meta_path = Path(here, '..', 'util', 'metadata.yaml')

    #For full details on function options, see the
    # doc string of tpump_analysis() in tpump_final.py.

    # If you are running sample data (which is real, not simulated)
    # from Alfresco, it is formatted differently than
    # EDU camera data (see tpump_analysis() doc string). You must set:
    #sample_data = True
    # By default, sample_data is False.
    # If residual non-linearity correction desired, enter string of
    # absolute path for file.  Otherwise:
    nonlin_path = None
    # Thresh_factor:  by default, 3.  This is the number of standard deviations
    # about the mean for dipole detection.
    thresh_factor = 3
    # Number of pumps during trap-pumping.  For example:
    num_pumps = {1:10000,2:10000,3:10000,4:10000}
    # By default, ill_corr is True.  For less accurate but somewhat
    # faster results, set ill_corr to False.
    ill_corr = True
    # tfit_const is True by default. It says whether to use
    # trap_fit_const() (for True)
    # or trap_fit() (for False).
    tfit_const = True
    # temperature in K at which to output tau, the release time constant for
    # each trap. For example:
    input_T = -85+273 # K
    # specifying header keys of the input .fits files for trap-pumped frames
    # for getting the phase time and EM gain for each frame:
    time_head = 'PHASE_T' #for example
    emgain_head = 'EM_GAIN' # for example
    # threshold for adjusted R^2 (called 'Rsq' below)
    # of curve fit for cross section:
    cs_fit_thresh = 0.8

    # The user has the option to save the temps dictionary, an
    # intermediate output that takes a while to generate by specifying the
    # absolute path (which includes the desired filename that ends in .npy)
    # with the save_temps parameter.
    # For example, save_temps = 'local_path/temps.npy'
    # By default, save_temps is None (i.e., no saving done).
    # To load in the temps dictionary for running the latter
    # half of the code, you can specify the absolute path (including the
    # filename ending in .npy) as the input load_temps.
    # By default, load_temps is None (i.e., no loading done).

    (trap_dict, trap_densities, bad_fit_counter, pre_sub_el_count,
        unused_fit_data, unused_temp_fit_data, two_or_less_count,
        noncontinuous_count)  = tpump_analysis(base_dir,
        time_head, emgain_head, num_pumps, meta_path, nonlin_path,
        thresh_factor = thresh_factor, ill_corr = ill_corr,
        tfit_const = tfit_const)

    # To change the binning involved in generating trap_densities, adjust
    # optional inputs bins_E and bins_cs.

    # You can save, for example, the main output trap_dict:
    #np.save('local_path/trap_dict.npy', trap_dict)
    # Or the entire output:
    # np.save('local_path/output.npy', (trap_dict,
    #     trap_densities, bad_fit_counter, pre_sub_el_count,
    #     unused_fit_data, unused_temp_fit_data, two_or_less_count,
    #     noncontinuous_count))

    # If you want to load in what you saved, you would do:
    # trap_dict = np.load('local_path/trap_dict.npy', allow_pickle = True)
    # trap_dict = trap_dict.tolist()
    # or:
    # output = np.load('local_path/output.npy', allow_pickle = True)
    # output = output.tolist()
    # (trap_dict, trap_densities, bad_fit_counter, pre_sub_el_count,
    #    unused_fit_data, unused_temp_fit_data, two_or_less_count,
    #    noncontinuous_count) = output

    # You can also re-bin for the trap densities with the following:
    E_vals = []
    cs_vals = []
    for pix_el in trap_dict.values():
        if pix_el['Rsq'] is not None:
            if pix_el['Rsq'] >= cs_fit_thresh:
                E_vals.append(pix_el['E'])
                cs_vals.append(pix_el['cs'])
    E_vals = np.array(E_vals)
    cs_vals = np.array(cs_vals)
    # re-bin and plot. Below are simply the default values for bins_E and
    # bins_cs. For the data used in the unit tests, only a few bins would be
    # non-zero-valued.
    bins_E = 100
    bins_cs = 10
    bins = [bins_cs, bins_E]
    fig, ax = plt.subplots()
    H = ax.hist2d(cs_vals, E_vals, bins = bins, cmap = 'viridis')
    ax.set_ylabel('E (eV)')
    ax.set_xlabel('cs (in cm^2)')
    ax.set_title('tau histogram (bins: {}x{})'.format(bins_cs, bins_E))
    fig.colorbar(H[3], ax=ax)
    plt.show()