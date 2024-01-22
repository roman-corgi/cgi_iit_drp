# tpumpanalysis
CGI Trap Pumping Analysis

This module analyzes trap-pumped frames and outputs the location of
each radiation trap (pixel and sub-electrode location within the pixel),
everything needed to determine the release time constant at any temperature
(the capture cross section for holes and the energy level), trap densities
(i.e., how many traps per pixel for each kind of trap found), and
information about the capture time constant (for potential future
analysis).  This function only works as intended for the EMCCD with its
electrodes' particular electric potential shape that will
be used for trap pumping on the Roman telescope.  It can find up to two
traps per sub-electrode location per pixel.  The function has an option to
save a preliminary output that takes a long time to create (the dictionary
called 'temps') and an option to load that in and start the function at
that point.

Some of the trap-finding and parameter-fitting code
adapted from Matlab code from Nathan Bush, and his code was used for his paper,
the basis for this code:
Nathan Bush, David Hall, and Andrew Holland,
J. of Astronomical Telescopes, Instruments, and Systems, 7(1), 016003 (2021).
https://doi.org/10.1117/1.JATIS.7.1.016003

The CTC requirements that this module fulfills:
Given 1) a set of clean focal plane images collected by a trap pumping
sequence, and 2) the detector parameters (temperature, phase time, number of
lines, number of cycles, and clocking scheme) associated with each set, the
CTC GSW shall compute the following values for each trap in the region of
interest:  location (pixel, electrode), capture cross-section, and release
time constant at the EXCAM observation operating temperature.

## Usage

See example_script.py for how to use the module.

## Author

Kevin Ludwick