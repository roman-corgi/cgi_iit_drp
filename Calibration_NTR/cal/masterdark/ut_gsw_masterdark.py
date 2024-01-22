"""
Unit tests for assembly of a dark frame onboard CGI
"""

import unittest

import numpy as np

from cal.masterdark.gsw_masterdark import build_dark, CLEANROW, CLEANCOL

class TestBuildDark(unittest.TestCase):
    """
    Tests for building the dark
    """

    def setUp(self):
        self.F = np.eye(1024)
        self.D = 3/3600*np.ones((1024, 1024))
        self.C = 0.02*np.ones((1024, 1024))
        self.g = 1
        self.t = 1
        pass


    def test_success(self):
        """Good inputs complete as expected"""
        build_dark(self.F, self.D, self.C, self.g, self.t)
        pass


    def test_output_size(self):
        """output is correct size"""
        M = build_dark(self.F, self.D, self.C, self.g, self.t)
        self.assertTrue(M.shape == (CLEANROW, CLEANCOL))
        pass


    def test_exact_case(self):
        """Exact case produces expected result"""
        tol = 1e-13

        F = 5*np.ones((1024, 1024))
        D = 1/7*np.ones((1024, 1024))
        C = 1*np.ones((1024, 1024))
        g = 5
        t = 7
        target = 3*np.ones((1024, 1024))

        M = build_dark(F, D, C, g, t)
        self.assertTrue(np.max(np.abs(M - target)) < tol)

        pass


    def test_gain_goes_as_1overg(self):
        """change in dark goes as 1/g"""
        tol = 1e-13

        F0 = build_dark(0*self.F, self.D, self.C, 1, self.t)
        M1 = build_dark(self.F, self.D, self.C, 1, self.t)
        M2 = build_dark(self.F, self.D, self.C, 2, self.t)
        M4 = build_dark(self.F, self.D, self.C, 4, self.t)
        dg1 = M1-F0
        dg2 = M2-F0
        dg4 = M4-F0

        self.assertTrue(np.max(np.abs(dg2 - dg1/2)) < tol)
        self.assertTrue(np.max(np.abs(dg4 - dg2/2)) < tol)
        pass


    def test_exptime_goes_as_t(self):
        """change in dark goes as t"""
        tol = 1e-13

        M0 = build_dark(self.F, self.D, self.C, self.g, 0)
        M2 = build_dark(self.F, self.D, self.C, self.g, 2)
        M4 = build_dark(self.F, self.D, self.C, self.g, 4)
        dg2 = M2-M0
        dg4 = M4-M0

        self.assertTrue(np.max(np.abs(dg4 - dg2*2)) < tol)
        pass


    def test_c_doesnt_change_with_g_or_t(self):
        """F = 0 and D = 0 implies C is constant"""
        tol = 1e-13

        M = build_dark(0*self.F, 0*self.D, self.C, 1, self.t)
        G2 = build_dark(0*self.F, 0*self.D, self.C, 2, self.t)
        G4 = build_dark(0*self.F, 0*self.D, self.C, 4, self.t)
        T2 = build_dark(0*self.F, 0*self.D, self.C, self.g, 2)
        T4 = build_dark(0*self.F, 0*self.D, self.C, self.g, 4)
        dg2 = G2-M
        dg4 = G4-M
        dt2 = T2-M
        dt4 = T4-M

        self.assertTrue(np.max(np.abs(dg2)) < tol)
        self.assertTrue(np.max(np.abs(dg4)) < tol)
        self.assertTrue(np.max(np.abs(dt2)) < tol)
        self.assertTrue(np.max(np.abs(dt4)) < tol)
        pass


    def test_bias_subtracted(self):
        """check there is no bias when all three noise terms are 0"""

        M = build_dark(0*self.F, 0*self.D, 0*self.C, 1, self.t)
        G2 = build_dark(0*self.F, 0*self.D, 0*self.C, 2, self.t)
        G4 = build_dark(0*self.F, 0*self.D, 0*self.C, 4, self.t)
        T2 = build_dark(0*self.F, 0*self.D, 0*self.C, self.g, 2)
        T4 = build_dark(0*self.F, 0*self.D, 0*self.C, self.g, 4)

        self.assertTrue((M == 0).all())
        self.assertTrue((G2 == 0).all())
        self.assertTrue((G4 == 0).all())
        self.assertTrue((T2 == 0).all())
        self.assertTrue((T4 == 0).all())

        pass



    def test_invalid_F(self):
        """Invalid inputs caught as expected"""
        for perr in [1, 0, -1.5, None, 1j, 'txt', np.ones((1024,)),
                     np.ones((1, 1024, 1024))]:
            with self.assertRaises(TypeError):
                build_dark(perr, self.D, self.C, self.g, self.t)
            pass
        pass


    def test_invalid_F_shape(self):
        """Invalid inputs caught as expected"""
        for perr in [np.ones((1024, 1)), np.ones((1, 1024)), np.ones((2, 2))]:
            with self.assertRaises(TypeError):
                build_dark(perr, self.D, self.C, self.g, self.t)
            pass
        pass


    def test_invalid_D(self):
        """Invalid inputs caught as expected"""
        for perr in [1, 0, -1.5, None, 1j, 'txt', np.ones((1024,)),
                     np.ones((1, 1024, 1024))]:
            with self.assertRaises(TypeError):
                build_dark(self.F, perr, self.C, self.g, self.t)
            pass
        pass


    def test_invalid_D_shape(self):
        """Invalid inputs caught as expected"""
        for perr in [np.ones((1024, 1)), np.ones((1, 1024)), np.ones((2, 2))]:
            with self.assertRaises(TypeError):
                build_dark(self.F, perr, self.C, self.g, self.t)
            pass
        pass


    def test_invalid_D_range(self):
        """Invalid inputs caught as expected"""
        for perr in [-1*np.ones_like(self.D)]:
            with self.assertRaises(TypeError):
                build_dark(self.F, perr, self.C, self.g, self.t)
            pass
        pass


    def test_invalid_C(self):
        """Invalid inputs caught as expected"""
        for perr in [1, 0, -1.5, None, 1j, 'txt', np.ones((1024,)),
                     np.ones((1, 1024, 1024))]:
            with self.assertRaises(TypeError):
                build_dark(self.F, self.D, perr, self.g, self.t)
            pass
        pass


    def test_invalid_C_shape(self):
        """Invalid inputs caught as expected"""
        for perr in [np.ones((1024, 1)), np.ones((1, 1024)), np.ones((2, 2))]:
            with self.assertRaises(TypeError):
                build_dark(self.F, self.D, perr, self.g, self.t)
            pass
        pass


    def test_invalid_C_range(self):
        """Invalid inputs caught as expected"""
        for perr in [-1*np.ones_like(self.C)]:
            with self.assertRaises(TypeError):
                build_dark(self.F, self.D, perr, self.g, self.t)
            pass
        pass


    def test_invalid_g(self):
        """Invalid inputs caught as expected"""
        for perr in [-1.5, None, 1j, 'txt', np.ones((1024,)),
                     np.ones((1024, 1024)), np.ones((1, 1024, 1024))]:
            with self.assertRaises(TypeError):
                build_dark(self.F, self.D, self.C, perr, self.t)
            pass
        pass


    def test_invalid_t(self):
        """Invalid inputs caught as expected"""
        for perr in [-1.5, None, 1j, 'txt', np.ones((1024,)),
                     np.ones((1024, 1024)), np.ones((1, 1024, 1024))]:
            with self.assertRaises(TypeError):
                build_dark(self.F, self.D, self.C, self.g, perr)
            pass
        pass


    def test_g_range_correct(self):
        """gain is valid >= 1 only"""

        for perr in [-10, -1, 0, 0.999]:
            with self.assertRaises(TypeError):
                build_dark(self.F, self.D, self.C, perr, self.t)
            pass

        for perr in [1, 1.5, 10]:
            build_dark(self.F, self.D, self.C, perr, self.t)
            pass
        pass




if __name__ == '__main__':
    unittest.main()
