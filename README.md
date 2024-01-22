`conda create -n iitdata python=3.10.4` to create a new environment 

`conda activate iitdata` to activate the environment

Use `conda env update -n iitdata -f iitdata-conda-env.yml` up install the basic necessary packages

3 custom packages are needed for IIT_L1_to_L2b_Processing: 
* Calibration_NTR, 
* gsw_testing_NTR 
* proc_cgi_frame_NTR. 
All are included in this archive. 
After creating and activating the iitdata conda env, navigate to each of these 3 folders and `pip install .`

1 public package is needed:
[https://github.com/wfirst-cgi/emccd_detect](https://github.com/wfirst-cgi/emccd_detect)
Download and pip install that package as well.

Check that all modelues are installed: `python testmodules.py`

Next, `cd IIT_L1_to_L2b_Processing` and follow the README there.

We also recommend installing the coralign repository. It contains various useful modules such as psffit, pupilfit, occastro that may come in handy for your future development.
[https://github.com/nasa-jpl/coralign](https://github.com/nasa-jpl/coralign)

To keep this repository to a manageable size, ~10GB of mock data for various unit tests are NOT included. Please contact Vanessa Bailey (vanessa.bailey@jpl.nasa.gov) for access.
These following folders are omitted:
- `Calibration/cal/calibrate_darks/testdata_small/`
- `Calibration/cal/tpumpanalysis/test_data*` (multiple folders)

## Authors

Jet Propulsion Laboratory, California Institute of Technology:
* Eric Cady
* A J Riggs
* Marie Ygouf
* Nick Bowman
* Vanessa Bailey

University of Alabama, Huntsville
* Kevin Ludwick
* Sam Miller
* Bijan Nemati



