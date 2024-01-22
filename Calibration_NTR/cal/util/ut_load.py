"""
Unit tests for load.py
"""

import unittest
import os

import numpy as np

from .load import load, load_ri, load_ap

class TestLoadRI(unittest.TestCase):
    """
    Test loading complex-valued functions from real and imaginary parts.

    Uses the following:
    ut_load_real_file.fits: 500x500 matrix of 1.0
    ut_load_imag_file.fits: 500x500 matrix of 2.0
    ut_load_smallimag_file.fits: 5x5 matrix of 2.0
    ut_load_bad_file.txt: text file not meeting FITS standard
    ut_load_does_not_exist: no file here

    """

    localpath = os.path.dirname(os.path.abspath(__file__))
    testpath = os.path.join(localpath, 'testdata')
    fnreal = os.path.join(testpath, 'ut_load_real_file.fits')
    fnimag = os.path.join(testpath, 'ut_load_imag_file.fits')
    fnsimag = os.path.join(testpath, 'ut_load_smallimag_file.fits')
    fnbad = os.path.join(testpath, 'ut_load_bad_file.txt')
    fndne = os.path.join(testpath, 'ut_load_does_not_exist')
    fnnot2d = os.path.join(testpath, 'ut_load_not2D.fits')

    # Success tests
    def test_load(self):
        """Verify loading works successfully"""
        mask = load_ri(self.fnreal, self.fnimag)
        self.assertTrue((mask == (1.0+2.0*1j)*np.ones((500, 500))).all())
        pass

    # Failure tests
    def test_no_real_file_present(self):
        """Check behavior with real file missing"""
        with self.assertRaises(IOError):
            load_ri(self.fndne, self.fnimag)
            pass
        pass

    def test_no_imag_file_present(self):
        """Check behavior with imaginary file missing"""
        with self.assertRaises(IOError):
            load_ri(self.fnreal, self.fndne)
            pass
        pass

    def test_real_file_read_error(self):
        """Check behavior with real file invalid FITS"""
        with self.assertRaises(IOError):
            load_ri(self.fnbad, self.fnimag)
            pass
        pass

    def test_imag_file_read_error(self):
        """Check behavior with imaginary file invalid FITS"""
        with self.assertRaises(IOError):
            load_ri(self.fnreal, self.fnbad)
            pass
        pass

    def test_real_file_not_2d(self):
        """Check behavior with real file wrong size FITS"""
        with self.assertRaises(TypeError):
            load_ri(self.fnnot2d, self.fnimag)
            pass
        pass


    def test_imag_file_not_2d(self):
        """Check behavior with imag file wrong size FITS"""
        with self.assertRaises(TypeError):
            load_ri(self.fnreal, self.fnnot2d)
            pass
        pass


    def test_masks_different_shapes(self):
        """
        Check behavior when real and imaginary parts have mismatched sizes
        """
        with self.assertRaises(TypeError):
            load_ri(self.fnreal, self.fnsimag)
            pass
        pass


class TestLoadAP(unittest.TestCase):
    """
    Test loading complex-valued functions from amplitude and phase.

    Uses the following:
    ut_load_amp_file.fits: 500x500 matrix of 1.0
    ut_load_ph_file.fits: 500x500 matrix of 0.0
    ut_load_smallph_file.fits: 5x5 matrix of 0.0
    ut_load_bad_file.txt: text file not meeting FITS standard
    ut_load_does_not_exist: no file here

    """

    localpath = os.path.dirname(os.path.abspath(__file__))
    testpath = os.path.join(localpath, 'testdata')
    fnamp = os.path.join(testpath, 'ut_load_amp_file.fits')
    fnph = os.path.join(testpath, 'ut_load_ph_file.fits')
    fnsph = os.path.join(testpath, 'ut_load_smallph_file.fits')
    fnbad = os.path.join(testpath, 'ut_load_bad_file.txt')
    fndne = os.path.join(testpath, 'ut_load_does_not_exist')
    fnnot2d = os.path.join(testpath, 'ut_load_not2D.fits')

    # Success tests
    def test_load(self):
        """Verify loading works successfully"""
        mask = load_ap(self.fnamp, self.fnph)
        self.assertTrue((mask == np.ones((500, 500))).all())
        pass

    # Failure tests
    def test_no_amp_file_present(self):
        """Check behavior with amplitude file missing"""
        with self.assertRaises(IOError):
            load_ap(self.fndne, self.fnph)
            pass
        pass

    def test_no_ph_file_present(self):
        """Check behavior with phase file missing"""
        with self.assertRaises(IOError):
            load_ap(self.fnamp, self.fndne)
            pass
        pass

    def test_amp_file_read_error(self):
        """Check behavior with amplitude file invalid FITS"""
        with self.assertRaises(IOError):
            load_ap(self.fnbad, self.fnph)
            pass
        pass

    def test_ph_file_read_error(self):
        """Check behavior with phase file invalid FITS"""
        with self.assertRaises(IOError):
            load_ap(self.fnamp, self.fnbad)
            pass
        pass

    def test_amp_file_not_2d(self):
        """Check behavior with amp file wrong size FITS"""
        with self.assertRaises(TypeError):
            load_ap(self.fnnot2d, self.fnph)
            pass
        pass


    def test_ph_file_not_2d(self):
        """Check behavior with ph file wrong size FITS"""
        with self.assertRaises(TypeError):
            load_ap(self.fnamp, self.fnnot2d)
            pass
        pass

    def test_masks_different_shapes(self):
        """
        Check behavior when amplitude and phase have mismatched sizes
        """
        with self.assertRaises(TypeError):
            load_ap(self.fnamp, self.fnsph)
            pass
        pass


class TestLoad(unittest.TestCase):
    """
    Test loading a real-valued representation from a single file.

    Uses the following:
    ut_load_amp_file.fits: 500x500 matrix of 1.0
    ut_load_bad_file.txt: text file not meeting FITS standard
    ut_load_does_not_exist: no file here

    """

    localpath = os.path.dirname(os.path.abspath(__file__))
    testpath = os.path.join(localpath, 'testdata')
    fn = os.path.join(testpath, 'ut_load_amp_file.fits')
    fnbad = os.path.join(testpath, 'ut_load_bad_file.txt')
    fndne = os.path.join(testpath, 'ut_load_does_not_exist')

    # Success tests
    def test_load(self):
        """Verify loading works successfully"""
        mask = load(self.fn)
        self.assertTrue((mask == np.ones((500, 500))).all())
        pass

    # Failure tests
    def test_no_file_present(self):
        """Check behavior with no file present"""
        with self.assertRaises(IOError):
            load(self.fndne)
            pass
        pass

    def test_file_read_error(self):
        """Check behavior with file invalid FITS"""
        with self.assertRaises(IOError):
            load(self.fnbad)
            pass
        pass





if __name__ == '__main__':
    unittest.main()
