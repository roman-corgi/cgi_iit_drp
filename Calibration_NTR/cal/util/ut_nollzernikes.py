"""
Unit tests for nollzernikes.py
"""

import unittest

import numpy as np

from .nollzernikes import xyzern

class TestXYZern(unittest.TestCase):
    """
    Unit test suite for xyzern()
    """

    def test_zernikes_as_expected(self):
        """
        Verify that known Zernike orders match shapes from this function
        """
        x = np.linspace(-5., 5., 100)
        y = np.linspace(-4., 6., 120)
        xm, ym = np.meshgrid(x, y)
        prad = 4.
        orders = [1, 2, 4, 8, 12, 13, 22]

        outarray = xyzern(xm, ym, prad, orders)

        # comparison data using Table 1 in Noll 1976
        rm = np.hypot(xm, ym)/float(prad)
        tm = np.arctan2(ym, xm)
        comparray = np.zeros((len(orders), xm.shape[0], xm.shape[1]))
        comparray[0] = np.ones_like(rm)
        comparray[1] = 2.*rm*np.cos(tm)
        comparray[2] = np.sqrt(3)*(2.*rm**2 - 1)
        comparray[3] = np.sqrt(8)*(3.*rm**2 - 2.)*rm*np.cos(tm)
        comparray[4] = np.sqrt(10)*(4*rm**2 - 3)*rm**2*np.cos(2*tm)
        comparray[5] = np.sqrt(10)*(4*rm**2 - 3)*rm**2*np.sin(2*tm)
        comparray[6] = np.sqrt(7)*(((20*rm**2 - 30)*rm**2 + 12)*rm**2 - 1)

        tol = 1e-12
        for j in range(len(orders)):
            self.assertTrue(np.max(np.abs(comparray[j] - outarray[j])) < tol)
            pass
        pass


    def test_unit_RMS(self):
        """Output shape should have rms of 1, up to discretization limits"""
        x = np.linspace(-5., 5., 100)
        y = np.linspace(-4., 6., 120)
        xm, ym = np.meshgrid(x, y)
        prad = 4.
        orders = [1, 2, 4, 8, 12, 13, 22]

        outarray = xyzern(xm, ym, prad, orders)

        mask = np.hypot(xm, ym) <= prad

        # could maybe tighten this, but really just want to check bulk
        # normalization is correct in an absolute sense. Getting this wrong
        # will make these numbers be 3.0, 7.0, 10.0, etc. instead of 1.
        tol = 0.01
        for j in range(len(orders)):
            rmsj = np.sqrt(np.mean(outarray[j, mask]**2))
            self.assertTrue(np.abs(rmsj - 1.0) < tol)
            pass
        pass


    def test_Nx1_2Darray(self):
        """
        Verify that Nx1 arrays go through without errors
        """
        x = np.linspace(-5., 5., 100)
        y = np.linspace(-4., 6., 120)
        xm, ym = np.meshgrid(x, y)
        prad = 4.
        orders = [1, 2, 4, 8, 12, 13, 22]

        xyzern(np.reshape(xm, (np.size(xm), 1)),
               np.reshape(ym, (np.size(ym), 1)), prad, orders)
        pass


    # Failure tests
    def test_x_2Darray(self):
        """Check correct failure on bad input array"""
        x = np.linspace(-5., 5., 100)
        y = np.linspace(-4., 6., 120)
        xm, ym = np.meshgrid(x, y)
        prad = 4.
        orders = [1, 2, 4, 8, 12, 13, 22]

        for badx in [xm[:, :-2], xm[:-2, :], np.ones((100,)),
                     np.ones((8, 8, 8)), 'text', 100, None]:
            with self.assertRaises(TypeError):
                xyzern(x=badx, y=ym, prad=prad, orders=orders)
                pass
            pass
        pass


    def test_y_2Darray(self):
        """Check correct failure on bad input array"""
        x = np.linspace(-5., 5., 100)
        y = np.linspace(-4., 6., 120)
        xm, ym = np.meshgrid(x, y)
        prad = 4.
        orders = [1, 2, 4, 8, 12, 13, 22]

        for bady in [ym[:, :-2], ym[:-2, :], np.ones((100,)),
                     np.ones((8, 8, 8)), 'text', 100, None]:
            with self.assertRaises(TypeError):
                xyzern(x=xm, y=bady, prad=prad, orders=orders)
                pass
            pass
        pass


    def test_prad_realpositivescalar(self):
        """Check correct failure on prad"""
        x = np.linspace(-5., 5., 100)
        y = np.linspace(-4., 6., 120)
        xm, ym = np.meshgrid(x, y)
        orders = [1, 2, 4, 8, 12, 13, 22]

        for badprad in [(4., 4.), [], (4.,), 'text',
                       4.*1j, -4., 0, None]:
            with self.assertRaises(TypeError):
                xyzern(x=xm, y=ym, prad=badprad, orders=orders)
                pass
            pass
        pass


    def test_orders_iterable(self):
        """Check correct failure if orders not iterable"""
        x = np.linspace(-5., 5., 100)
        y = np.linspace(-4., 6., 120)
        xm, ym = np.meshgrid(x, y)
        prad = 4.

        for badorders in [1, None, 'text']:
            with self.assertRaises(TypeError):
                xyzern(x=xm, y=ym, prad=prad, orders=badorders)
                pass
            pass
        pass


    def test_orders_elements_are_postive_scalar_integers(self):
        """Check correct failure if orders not iterable"""
        x = np.linspace(-5., 5., 100)
        y = np.linspace(-4., 6., 120)
        xm, ym = np.meshgrid(x, y)
        prad = 4.

        for badorders in [[5, 7, 8, -3], [1, 2, 4, 4.5], [3, 1j],
                          [1, 5, 'text'], [None, None, None], [0, 1, 2]]:
            with self.assertRaises(TypeError):
                xyzern(x=xm, y=ym, prad=prad, orders=badorders)
                pass
            pass
        pass



if __name__ == '__main__':
    unittest.main()
