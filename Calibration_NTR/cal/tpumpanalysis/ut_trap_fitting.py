"""Unit tests for trap_fitting.py."""
import unittest
from unittest.mock import patch

import numpy as np

import warnings
import cal.util.ut_check as ut_check
from cal.tpumpanalysis.trap_fitting import trap_fit_const, trap_fit, fit_cs

#makes unit tests run in order listed
unittest.defaultTestLoader.sortTestMethodsUsing = lambda *args: -1

num_pumps = 10000
# probability functions for trap_fit_const()
def P1c(time_data, offset, pc, tau):
    """Probability function 1, one trap.
    """
    return offset+(num_pumps*pc*(np.exp(-time_data/tau)-
        np.exp(-2*time_data/tau)))

def P1_P1c(time_data, offset, pc, tau, pc2, tau2):
    """Probability function 1, two traps.
    """
    return offset+num_pumps*pc*(np.exp(-time_data/tau)-
        np.exp(-2*time_data/tau))+ \
        num_pumps*pc2*(np.exp(-time_data/tau2)-
        np.exp(-2*time_data/tau2))

def P2c(time_data, offset, pc, tau):
    """Probability function 2, one trap.
    """
    return offset+(num_pumps*pc*(np.exp(-2*time_data/tau)-
        np.exp(-3*time_data/tau)))

def P1_P2c(time_data, offset, pc, tau, pc2, tau2):
    """One trap for probability function 1, one for probability
    function 2.
    """
    return offset+num_pumps*pc*(np.exp(-time_data/tau)-
        np.exp(-2*time_data/tau))+ \
        num_pumps*pc2*(np.exp(-2*time_data/tau2)-
        np.exp(-3*time_data/tau2))

def P2_P2c(time_data, offset, pc, tau, pc2, tau2):
    """Probability function 2, two traps.
    """
    return offset+num_pumps*pc*(np.exp(-2*time_data/tau)-
        np.exp(-3*time_data/tau))+ \
        num_pumps*pc2*(np.exp(-2*time_data/tau2)-
        np.exp(-3*time_data/tau2))

def P3c(time_data, offset, pc, tau):
    """Probability function 3, one trap.
    """
    return offset+(num_pumps*pc*(np.exp(-time_data/tau)-
        np.exp(-4*time_data/tau)))

def P3_P3c(time_data, offset, pc, tau, pc2, tau2):
    """Probability function 3, two traps.
    """
    return offset+num_pumps*pc*(np.exp(-time_data/tau)-
        np.exp(-4*time_data/tau))+ \
        num_pumps*pc2*(np.exp(-time_data/tau2)-
        np.exp(-4*time_data/tau2))

def P2_P3c(time_data, offset, pc, tau, pc2, tau2):
    """One trap for probability function 2, one for probability
    function 3.
    """
    return offset+num_pumps*pc*(np.exp(-2*time_data/tau)-
        np.exp(-3*time_data/tau))+ \
        num_pumps*pc2*(np.exp(-time_data/tau2)-
        np.exp(-4*time_data/tau2))

class TestTrapFitConst(unittest.TestCase):
    """Unit tests for trap_fit_const().  I avoid testing to see if output
    is as expected since I am not testing the function curve_fit() here."""
    def setUp(self):
        # make up some parameters
        self.fit_thresh = 0.9
        self.offset = 1
        self.pc = 1
        self.pc2 = 1
        #In order of amplitudes overall (given comparable tau and tau2):
        # P1 biggest, then P3, then P2
        self.tau = 7.5e-3
        self.tau2 = 8.8e-3
        self.time_data = np.logspace(-6, -2, 100)
        self.pc_min = 0
        self.pc_max = 2
        self.tau_min = 0.7e-6
        self.tau_max = 1.3e-2
        self.offset_min = -10
        self.offset_max = 10
        self.amps1 = P1c(self.time_data, self.offset, self.pc, self.tau)
        self.amps11 = P1_P1c(self.time_data, self.offset, self.pc, self.tau,
            self.pc2, self.tau2)
        self.amps2 = P2c(self.time_data, self.offset, self.pc, self.tau)
        self.amps12 = P1_P2c(self.time_data, self.offset, self.pc, self.tau,
            self.pc2, self.tau2)
        self.amps22 = P2_P2c(self.time_data, self.offset, self.pc, self.tau,
            self.pc2, self.tau2)
        self.amps3 = P3c(self.time_data, self.offset, self.pc, self.tau)
        self.amps33 = P3_P3c(self.time_data, self.offset, self.pc, self.tau,
            self.pc2, self.tau2)
        self.amps23 = P2_P3c(self.time_data, self.offset, self.pc, self.tau,
            self.pc2, self.tau2)
        # smaller time constant, self.tau, would fit this time range better
        self.both_a11 = {'amp': self.amps11[0:30], 't': self.time_data[0:30]}
        # a case where ALL the 'above' frames shared with the 'below' frames
        # so b/w P1 and P2, P1 would dominate (self.tau) since taus are of same
        # order of magnitude
        self.both_a12 = {'amp': self.amps12, 't': self.time_data}
        # larger time constant, self.tau2, would fit this time range better
        self.both_a22 = {'amp': self.amps22[95:], 't': self.time_data[95:]}
        # below used for scheme ==3 or 4 (self.both_a22 used for this and
        #scheme ==1 or 2)
        # smaller time constant, self.tau, would fit this time range better
        self.both_a33 = {'amp': self.amps33[0:30], 't': self.time_data[0:30]}
        # a case where ALL the 'above' frames shared with the 'below' frames
        # P3 should dominate (self.tau2)
        self.both_a23 = {'amp': self.amps23, 't': self.time_data}
        # filter out expected warnings
        warnings.filterwarnings('ignore', category=UserWarning,
                        module='cal.tpumpanalysis.trap_fitting')

    def test_sch12_R1_bigger(self):
        """R_value1 bigger than R_value2 (schemes 1 and 2)."""
        for sch in [1,2]:
            fd = trap_fit_const(sch, self.amps1, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.pc_min,
                self.pc_max, self.offset_min, self.offset_max)
            self.assertEqual(set([1]), set(fd.keys()))
            self.assertTrue(len(fd[1]) == 1)

    def test_sch12_R2_bigger(self):
        """R_value2 bigger than R_value1 (schemes 1 and 2)."""
        for sch in [1,2]:
            fd = trap_fit_const(sch, self.amps2, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.pc_min,
                self.pc_max, self.offset_min, self.offset_max)
            self.assertEqual(set([2]), set(fd.keys()))
            self.assertTrue(len(fd[2]) == 1)

    @patch('cal.tpumpanalysis.trap_fitting.curve_fit')
    def test_sch12_2trap11(self, mock_fit):
        """1-trap fits fail, but 2-trap P1_P1 fits work."""
        for sch in [1,2]:
            # I mock all the outputs.
            badout = ([0.1,0.1,1e8], 10*np.ones([3,3])) # bad output
            out = ([self.offset, self.pc, self.tau, self.pc2, self.tau2],
                1e-5*np.ones([5,5]))
            mock_fit.side_effect = [badout, badout, badout, badout,
                out, out, out, out, out, out]
            fd = trap_fit_const(sch, self.amps11, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.pc_min,
                self.pc_max, self.offset_min, self.offset_max)
            self.assertEqual(set([1]), set(fd.keys()))
            self.assertTrue(len(fd[1]) == 2)

    @patch('cal.tpumpanalysis.trap_fitting.curve_fit')
    def test_sch12_2trap12(self, mock_fit):
        """1-trap fits fail, but 2-trap P1_P2 fits work."""
        for sch in [1,2]:
            badout = ([0.1,0.1,1e8], 10*np.ones([3,3])) # bad output
            out = ([self.offset, self.pc, self.tau, self.pc2, self.tau2],
                1e-5*np.ones([5,5]))
            mock_fit.side_effect = [badout, badout, badout, badout,
                out, out, out, out, out, out]
            fd = trap_fit_const(sch, self.amps12, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.pc_min,
                self.pc_max, self.offset_min, self.offset_max)
            self.assertEqual(set([1,2]), set(fd.keys()))
            self.assertTrue(len(fd[1]) == 1 and len(fd[2]) == 1)

    @patch('cal.tpumpanalysis.trap_fitting.curve_fit')
    def test_sch12_2trap22(self, mock_fit):
        """1-trap fits fail, but 2-trap P2_P2 fits work."""
        for sch in [1,2]:
            badout = ([0.1,0.1,1e8], 10*np.ones([3,3])) # bad output
            out = ([self.offset, self.pc, self.tau, self.pc2, self.tau2],
                1e-5*np.ones([5,5]))
            mock_fit.side_effect = [badout, badout,badout, badout,
                out, out, out, out, out, out]
            fd = trap_fit_const(sch, self.amps22, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.pc_min,
                self.pc_max, self.offset_min, self.offset_max)
            self.assertEqual(set([2]), set(fd.keys()))
            self.assertTrue(len(fd[2]) == 2)

    @patch('cal.tpumpanalysis.trap_fitting.curve_fit')
    def test_sch12_none(self, mock_fit):
        """1-trap and 2-trap fits fail."""
        for sch in [1,2]:
            badout = ([0.1,0.1,1e8], 10*np.ones([3,3])) # bad output
            badout2 = ([0.1,0.1,1e8,0.1,1e8], 10*np.ones([5,5]))
            mock_fit.side_effect = [badout, badout, badout, badout,
                badout2, badout2, badout2, badout2, badout2, badout2]
            with self.assertWarns(UserWarning):
                fd = trap_fit_const(sch, self.amps22, self.time_data,
                num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.pc_min,
                self.pc_max, self.offset_min, self.offset_max)
            self.assertEqual(fd, None)

    @patch('cal.tpumpanalysis.trap_fitting.curve_fit')
    def test_sch12_2trap11_both_a(self, mock_fit):
        """both_a!=None, 2-trap P1_P1 fits work."""
        for sch in [1,2]:
            #I mock the outputs.  (And 1-trap outputs can be good; doesn't
            #matter when both_a!=None.)
            out = ([self.offset, self.pc, self.tau],
                1e-5*np.ones([3,3]))
            out2 = ([self.offset, self.pc, self.tau, self.pc2, self.tau2],
                1e-5*np.ones([5,5]))
            mock_fit.side_effect = [out2, out2, out2, out2, out2, out2]
            fd = trap_fit_const(sch, self.amps11, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.pc_min,
                self.pc_max, self.offset_min, self.offset_max, self.both_a11)
            self.assertEqual(set(['a','b']), set(fd.keys()))
            self.assertEqual(set([1]), set(fd['a'].keys()))
            self.assertEqual(set([1]), set(fd['b'].keys()))
            # 't' has the lower times, so self.tau better fit b/c
            # self.tau < self.tau2
            self.assertTrue(np.abs(fd['a'][1][0][2] - self.tau) <=
                np.abs(fd['a'][1][0][2] - self.tau2))

    @patch('cal.tpumpanalysis.trap_fitting.curve_fit')
    def test_sch12_2trap12_both_a(self, mock_fit):
        """both_a!=None, 2-trap P1_P2 fits work."""
        for sch in [1,2]:
            #I mock the outputs.  (And 1-trap outputs can be good; doesn't
            #matter when both_a!=None.)
            out = ([self.offset, self.pc, self.tau],
                1e-5*np.ones([3,3]))
            out2 = ([self.offset, self.pc, self.tau, self.pc2, self.tau2],
                1e-5*np.ones([5,5]))
            mock_fit.side_effect = [out2, out2, out2, out2, out2, out2]
            fd = trap_fit_const(sch, self.amps12, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.pc_min,
                self.pc_max, self.offset_min, self.offset_max, self.both_a12)
            self.assertEqual(set(['a','b']), set(fd.keys()))
            # 't' is actually the whole time_data for this one, and b/c
            # of note in setUp(), self.tau fits better.
            # For cases like these where time_data is completely shared b/w
            # 'above' and 'below': noted to handle these differently in v2;
            #but physically, shouldn't happen unless tau values are the same
            # or very close
            self.assertEqual(set([1]), set(fd['a'].keys()))
            self.assertEqual(set([2]), set(fd['b'].keys()))
            self.assertTrue(np.abs(fd['a'][1][0][2] - self.tau) <=
                np.abs(fd['a'][1][0][2] - self.tau2))

    @patch('cal.tpumpanalysis.trap_fitting.curve_fit')
    def test_sch12_2trap22_both_a(self, mock_fit):
        """both_a!=None, 2-trap P2_P2 fits work."""
        for sch in [1,2]:
            # I mock the outputs.  (And 1-trap outputs can be good; doesn't
            #matter when both_a!=None.)
            out = ([self.offset, self.pc, self.tau],
                1e-5*np.ones([3,3]))
            out2 = ([self.offset, self.pc, self.tau, self.pc2, self.tau2],
                1e-5*np.ones([5,5]))
            mock_fit.side_effect = [out2, out2, out2, out2, out2, out2]
            fd = trap_fit_const(sch, self.amps22, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.pc_min,
                self.pc_max, self.offset_min, self.offset_max, self.both_a22)
            self.assertEqual(set(['a','b']), set(fd.keys()))
            self.assertEqual(set([2]), set(fd['a'].keys()))
            self.assertEqual(set([2]), set(fd['b'].keys()))
            # 't' has the higher times, so self.tau2 better fit b/c
            # self.tau < self.tau2
            self.assertTrue(np.abs(fd['a'][2][0][2] - self.tau2) <=
                np.abs(fd['a'][2][0][2] - self.tau))

    @patch('cal.tpumpanalysis.trap_fitting.curve_fit')
    def test_sch12_both_a_none(self, mock_fit):
        """both_a!=None, 2-trap fits fail."""
        for sch in [1,2]:
            badout = ([0.1,0.1,1e8], 10*np.ones([3,3])) # bad output
            badout2 = ([0.1,0.1,1e8,0.1,1e8], 10*np.ones([5,5]))
            mock_fit.side_effect = [badout2, badout2, badout2, badout2,
                badout2, badout2]
            with self.assertWarns(UserWarning):
                fd = trap_fit_const(sch, self.amps22, self.time_data,
                num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.pc_min,
                self.pc_max, self.offset_min, self.offset_max, self.both_a22)
            self.assertEqual(fd, None)

    #tests for scheme == 3 or 4 below
    def test_sch34_R2_bigger(self):
        """R_value2 bigger than R_value3 (schemes 3 and 4)."""
        for sch in [3,4]:
            fd = trap_fit_const(sch, self.amps2, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.pc_min,
                self.pc_max, self.offset_min, self.offset_max)
            self.assertEqual(set([2]), set(fd.keys()))
            self.assertTrue(len(fd[2]) == 1)

    def test_sch34_R3_bigger(self):
        """R_value3 bigger than R_value2 (schemes 3 and 4)."""
        for sch in [3,4]:
            fd = trap_fit_const(sch, self.amps3, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.pc_min,
                self.pc_max, self.offset_min, self.offset_max)
            self.assertEqual(set([3]), set(fd.keys()))
            self.assertTrue(len(fd[3]) == 1)

    @patch('cal.tpumpanalysis.trap_fitting.curve_fit')
    def test_sch34_2trap22(self, mock_fit):
        """1-trap fits fail, but 2-trap P2_P2 fits work."""
        for sch in [3,4]:
            # I mock the outputs.
            badout = ([0.1,0.1,1e8], 10*np.ones([3,3])) # bad output
            out = ([self.offset, self.pc, self.tau, self.pc2, self.tau2],
                1e-5*np.ones([5,5]))
            mock_fit.side_effect = [badout, badout, badout, badout,
                out, out, out, out, out, out]
            fd = trap_fit_const(sch, self.amps22, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.pc_min,
                self.pc_max, self.offset_min, self.offset_max)
            self.assertEqual(set([2]), set(fd.keys()))
            self.assertTrue(len(fd[2]) == 2)

    @patch('cal.tpumpanalysis.trap_fitting.curve_fit')
    def test_sch34_2trap23(self, mock_fit):
        """1-trap fits fail, but 2-trap P2_P3 fits work."""
        for sch in [3,4]:
            badout = ([0.1,0.1,1e8], 10*np.ones([3,3])) # bad output
            out = ([self.offset, self.pc, self.tau, self.pc2, self.tau2],
                1e-5*np.ones([5,5]))
            mock_fit.side_effect = [badout, badout, badout, badout,
                out, out, out, out, out, out]
            fd = trap_fit_const(sch, self.amps23, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.pc_min,
                self.pc_max, self.offset_min, self.offset_max)
            self.assertEqual(set([2,3]), set(fd.keys()))
            self.assertTrue(len(fd[2]) == 1 and len(fd[3]) == 1)

    @patch('cal.tpumpanalysis.trap_fitting.curve_fit')
    def test_sch34_2trap33(self, mock_fit):
        """1-trap fits fail, but 2-trap P3_P3 fits work."""
        for sch in [3,4]:
            badout = ([0.1,0.1,1e8], 10*np.ones([3,3])) # bad output
            out = ([self.offset, self.pc, self.tau, self.pc2, self.tau2],
                1e-5*np.ones([5,5]))
            mock_fit.side_effect = [badout, badout, badout, badout,
                out, out, out, out, out, out]
            fd = trap_fit_const(sch, self.amps33, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.pc_min,
                self.pc_max, self.offset_min, self.offset_max)
            self.assertEqual(set([3]), set(fd.keys()))
            self.assertTrue(len(fd[3]) == 2)

    @patch('cal.tpumpanalysis.trap_fitting.curve_fit')
    def test_sch34_none(self, mock_fit):
        """1-trap and 2-trap fits fail."""
        for sch in [3,4]:
            badout = ([0.1,0.1,1e8], 10*np.ones([3,3])) # bad output
            badout2 = ([0.1,0.1,1e8,0.1,1e8], 10*np.ones([5,5]))
            mock_fit.side_effect = [badout, badout, badout, badout,
                badout2, badout2, badout2, badout2, badout2, badout2]
            with self.assertWarns(UserWarning):
                fd = trap_fit_const(sch, self.amps33, self.time_data,
                num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.pc_min,
                self.pc_max, self.offset_min, self.offset_max)
            self.assertEqual(fd, None)

    @patch('cal.tpumpanalysis.trap_fitting.curve_fit')
    def test_sch34_2trap22_both_a(self, mock_fit):
        """both_a!=None, 2-trap P2_P2 fits work."""
        for sch in [3,4]:
            # So I mock the outputs.  (And 1-trap outputs can be good; doesn't
            #matter when both_a!=None.)
            out = ([self.offset, self.pc, self.tau],
                1e-5*np.ones([3,3]))
            out2 = ([self.offset, self.pc, self.tau, self.pc2, self.tau2],
                1e-5*np.ones([5,5]))
            mock_fit.side_effect = [out2, out2, out2, out2, out2, out2]
            fd = trap_fit_const(sch, self.amps22, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.pc_min,
                self.pc_max, self.offset_min, self.offset_max, self.both_a22)
            self.assertEqual(set(['a','b']), set(fd.keys()))
            self.assertEqual(set([2]), set(fd['a'].keys()))
            self.assertEqual(set([2]), set(fd['b'].keys()))
            # 't' has the higher times, so self.tau2 better fit b/c
            # self.tau < self.tau2
            self.assertTrue(np.abs(fd['a'][2][0][2] - self.tau2) <=
                np.abs(fd['a'][2][0][2] - self.tau))

    @patch('cal.tpumpanalysis.trap_fitting.curve_fit')
    def test_sch34_2trap23_both_a(self, mock_fit):
        """both_a!=None, 2-trap P2_P3 fits work."""
        for sch in [3,4]:
            #  I mock the outputs.  (And 1-trap outputs can be good; doesn't
            #matter when both_a!=None.)
            out = ([self.offset, self.pc, self.tau],
                1e-5*np.ones([3,3]))
            out2 = ([self.offset, self.pc, self.tau, self.pc2, self.tau2],
                1e-5*np.ones([5,5]))
            mock_fit.side_effect = [out2, out2, out2, out2, out2, out2]
            fd = trap_fit_const(sch, self.amps23, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.pc_min,
                self.pc_max, self.offset_min, self.offset_max, self.both_a23)
            self.assertEqual(set(['a','b']), set(fd.keys()))
            # 't' is actually the whole time_data for this one, and b/c
            # of note in setUp(), self.tau2 fits better.
            # For cases like these where time_data is completely shared b/w
            # 'above' and 'below': noted to handle these differently in v2
            self.assertEqual(set([3]), set(fd['a'].keys()))
            self.assertEqual(set([2]), set(fd['b'].keys()))
            self.assertTrue(np.abs(fd['a'][3][0][2] - self.tau2) <=
                np.abs(fd['a'][3][0][2] - self.tau))

    @patch('cal.tpumpanalysis.trap_fitting.curve_fit')
    def test_sch34_2trap33_both_a(self, mock_fit):
        """both_a!=None, 2-trap P3_P3 fits work."""
        for sch in [3,4]:
            # I mock the outputs.  (And 1-trap outputs can be good; doesn't
            #matter when both_a!=None.)
            out = ([self.offset, self.pc, self.tau],
                1e-5*np.ones([3,3]))
            out2 = ([self.offset, self.pc, self.tau, self.pc2, self.tau2],
                1e-5*np.ones([5,5]))
            mock_fit.side_effect = [out2, out2, out2, out2, out2, out2]
            fd = trap_fit_const(sch, self.amps33, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.pc_min,
                self.pc_max, self.offset_min, self.offset_max, self.both_a33)
            self.assertEqual(set(['a','b']), set(fd.keys()))
            self.assertEqual(set([3]), set(fd['a'].keys()))
            self.assertEqual(set([3]), set(fd['b'].keys()))
            # 't' has the lower times, so self.tau better fit b/c
            # self.tau < self.tau2
            self.assertTrue(np.abs(fd['a'][3][0][2] - self.tau) <=
                np.abs(fd['a'][3][0][2] - self.tau2))

    @patch('cal.tpumpanalysis.trap_fitting.curve_fit')
    def test_sch34_both_a_none(self, mock_fit):
        """both_a!=None, 2-trap fits fail."""
        for sch in [3,4]:
            badout = ([0.1,0.1,1e8], 10*np.ones([3,3])) # bad output
            badout2 = ([0.1,0.1,1e8,0.1,1e8], 10*np.ones([5,5]))
            mock_fit.side_effect = [badout2, badout2, badout2, badout2,
                badout2, badout2]
            with self.assertWarns(UserWarning):
                fd = trap_fit_const(sch, self.amps33, self.time_data,
                num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.pc_min,
                self.pc_max, self.offset_min, self.offset_max, self.both_a33)
            self.assertEqual(fd, None)

    def test_curve_fit_fail(self):
            """Returns None if curve_fit fails."""
            amps = self.amps1.copy()
            amps[0] = np.inf # causes curve_fit to fail
            for sch in [1,2,3,4]:
                with self.assertWarns(UserWarning):
                    fd = trap_fit(sch, amps, self.time_data, num_pumps,
                    self.fit_thresh, self.tau_min, self.tau_max, self.pc_min,
                    self.pc_max, self.offset_min, self.offset_max,
                    self.both_a33)
                self.assertTrue(fd is None)

    def test_bad_scheme(self):
        """Scheme input bad."""
        for er in [1j, None, (1.0,), [5,5], 'txt', -1, 0, 5]:
            with self.assertRaises(TypeError):
                trap_fit_const(er, self.amps33, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.pc_min,
                self.pc_max, self.offset_min, self.offset_max, self.both_a33)

    def test_bad_amps(self):
        """amps input bad."""
        for er in ut_check.oneDlist:
            with self.assertRaises(TypeError):
                trap_fit_const(1, er, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.pc_min,
                self.pc_max, self.offset_min, self.offset_max, self.both_a33)

    def test_len_times(self):
        """times must have a number of unique phase times longer than the
        number of fitted parameters."""
        for times in [np.array([]), np.array([1,2,3,4,5]),
            np.array([1,2,3,4,5,5])]:
            with self.assertRaises(IndexError):
                # let amps be times so that they are arrays of same length
                trap_fit_const(1, times, times, num_pumps,
                    self.fit_thresh, self.tau_min, self.tau_max, self.pc_min,
                    self.pc_max, self.offset_min, self.offset_max,
                    self.both_a33)

    def test_bad_times(self):
        """times input bad."""
        for er in ut_check.oneDlist:
            with self.assertRaises(TypeError):
                trap_fit_const(1, self.amps1, er, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.pc_min,
                self.pc_max, self.offset_min, self.offset_max, self.both_a33)

    def test_len_amps(self):
        """times and amps should have same length."""
        #self.amps1 same length as self.time_data. times has one more element.
        times = np.linspace(1e-6, 1e-2, 101)
        with self.assertRaises(ValueError):
            trap_fit_const(1, self.amps1, times, num_pumps,
            self.fit_thresh, self.tau_min, self.tau_max, self.pc_min,
            self.pc_max, self.offset_min, self.offset_max, self.both_a33)

    def test_num_pumps(self):
        """num_pumps input bad."""
        for er in ut_check.psilist:
            with self.assertRaises(TypeError):
                trap_fit_const(1, self.amps1, self.time_data, er,
                self.fit_thresh, self.tau_min, self.tau_max, self.pc_min,
                self.pc_max, self.offset_min, self.offset_max, self.both_a33)

    def test_fit_thresh(self):
        """fit_thresh input bad."""
        for er in ut_check.rnslist:
            with self.assertRaises(TypeError):
                trap_fit_const(1, self.amps1, self.time_data, num_pumps,
                er, self.tau_min, self.tau_max, self.pc_min,
                self.pc_max, self.offset_min, self.offset_max, self.both_a33)

    def test_fit_thresh_value(self):
        """fit_thresh must be in (0,1)."""
        with self.assertRaises(ValueError):
            trap_fit_const(1, self.amps1, self.time_data, num_pumps,
            1.2, self.tau_min, self.tau_max, self.pc_min,
            self.pc_max, self.offset_min, self.offset_max, self.both_a33)

    def test_tau_min(self):
        """tau_min input bad."""
        for er in ut_check.rnslist:
            with self.assertRaises(TypeError):
                trap_fit_const(1, self.amps1, self.time_data, num_pumps,
                self.fit_thresh, er, self.tau_max, self.pc_min,
                self.pc_max, self.offset_min, self.offset_max, self.both_a33)

    def test_tau_max(self):
        """tau_max input bad."""
        for er in ut_check.rnslist:
            with self.assertRaises(TypeError):
                trap_fit_const(1, self.amps1, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, er, self.pc_min,
                self.pc_max, self.offset_min, self.offset_max, self.both_a33)

    def test_tau_max_value(self):
        """tau_max must be > tau_min."""
        for er in [self.tau_min, (self.tau_min*.9)]:
            with self.assertRaises(ValueError):
                trap_fit_const(1, self.amps1, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, er, self.pc_min,
                self.pc_max, self.offset_min, self.offset_max, self.both_a33)

    def test_pc_min(self):
        """pc_min input bad."""
        for er in ut_check.rnslist:
            with self.assertRaises(TypeError):
                trap_fit_const(1, self.amps1, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, er,
                self.pc_max, self.offset_min, self.offset_max, self.both_a33)

    def test_pc_max(self):
        """pc_max input bad."""
        for er in ut_check.rnslist:
            with self.assertRaises(TypeError):
                trap_fit_const(1, self.amps1, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.pc_min,
                er, self.offset_min, self.offset_max, self.both_a33)

    def test_pc_max_value(self):
        """pc_max must be > pc_min."""
        for er in [self.pc_min, (self.pc_min*.9)]:
            with self.assertRaises(ValueError):
                trap_fit_const(1, self.amps1, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.pc_min,
                er, self.offset_min, self.offset_max, self.both_a33)

    def test_offset_min(self):
        """offset_min input bad."""
        for er in ut_check.rslist:
            with self.assertRaises(TypeError):
                trap_fit_const(1, self.amps1, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.pc_min,
                self.pc_max, er, self.offset_max, self.both_a33)

    def test_offset_max(self):
        """offset_max input bad."""
        for er in ut_check.rslist:
            with self.assertRaises(TypeError):
                trap_fit_const(1, self.amps1, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.pc_min,
                self.pc_max, self.offset_min, er, self.both_a33)

    def test_offset_max_value(self):
        """offset_max must be > offset_min."""
        for er in [self.offset_min, (self.offset_min-1)]:
            with self.assertRaises(ValueError):
                trap_fit_const(1, self.amps1, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.pc_min,
                self.pc_max, self.offset_min, er, self.both_a33)

    def test_both_a(self):
        """both_a input bad."""
        for er in [np.array([1,2]), 'foo', 1, -2.3, 0, (3,3), [2,2]]:
            with self.assertRaises(TypeError):
                trap_fit_const(1, self.amps1, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.pc_min,
                self.pc_max, self.offset_min, self.offset_max, er)

    def test_both_a_key(self):
        """both_a doesn't have expected keys."""
        er = {'amp_bad': [1,2,3,4], 't': [.1,.2,.3,.4]}
        with self.assertRaises(KeyError):
            trap_fit_const(1, self.amps1, self.time_data, num_pumps,
            self.fit_thresh, self.tau_min, self.tau_max, self.pc_min,
            self.pc_max, self.offset_min, self.offset_max, er)

    def test_both_a_len(self):
        """both_a 'amp' and 't' must have same length."""
        er = {'amp': [1,2,3,4], 't': [.1,.2,.3,.4,.5]}
        with self.assertRaises(ValueError):
            trap_fit_const(1, self.amps1, self.time_data, num_pumps,
            self.fit_thresh, self.tau_min, self.tau_max, self.pc_min,
            self.pc_max, self.offset_min, self.offset_max, er)

# Now define probability functions as appropriate to trap_fit testing below.
def P1(time_data, offset, tauc, tau):
        """Probability function 1, one trap.
        """
        pc = 1 - np.exp(-time_data/tauc)
        return offset+(num_pumps*pc*(np.exp(-time_data/tau)-
            np.exp(-2*time_data/tau)))

def P1_P1(time_data, offset, tauc, tau, tauc2, tau2):
    """Probability function 1, two traps.
    """
    pc = 1 - np.exp(-time_data/tauc)
    pc2 = 1 - np.exp(-time_data/tauc2)
    return offset+num_pumps*pc*(np.exp(-time_data/tau)-
        np.exp(-2*time_data/tau))+ \
        num_pumps*pc2*(np.exp(-time_data/tau2)-np.exp(-2*time_data/tau2))

def P2(time_data, offset, tauc, tau):
    """Probability function 2, one trap.
    """
    pc = 1 - np.exp(-time_data/tauc)
    return offset+(num_pumps*pc*(np.exp(-2*time_data/tau)-
        np.exp(-3*time_data/tau)))

def P1_P2(time_data, offset, tauc, tau, tauc2, tau2):
    """One trap for probability function 1, one for probability function 2.
    """
    pc = 1 - np.exp(-time_data/tauc)
    pc2 = 1 - np.exp(-time_data/tauc2)
    return offset+num_pumps*pc*(np.exp(-time_data/tau)-
        np.exp(-2*time_data/tau))+ \
        num_pumps*pc2*(np.exp(-2*time_data/tau2)-np.exp(-3*time_data/tau2))

def P2_P2(time_data, offset, tauc, tau, tauc2, tau2):
    """Probability function 2, two traps.
    """
    pc = 1 - np.exp(-time_data/tauc)
    pc2 = 1 - np.exp(-time_data/tauc2)
    return offset+num_pumps*pc*(np.exp(-2*time_data/tau)-
        np.exp(-3*time_data/tau))+ \
        num_pumps*pc2*(np.exp(-2*time_data/tau2)-np.exp(-3*time_data/tau2))

def P3(time_data, offset, tauc, tau):
    """Probability function 3, one trap.
    """
    pc = 1 - np.exp(-time_data/tauc)
    return offset+(num_pumps*pc*(np.exp(-time_data/tau)-
        np.exp(-4*time_data/tau)))

def P3_P3(time_data, offset, tauc, tau, tauc2, tau2):
    """Probability function 3, two traps.
    """
    pc = 1 - np.exp(-time_data/tauc)
    pc2 = 1 - np.exp(-time_data/tauc2)
    return offset+num_pumps*pc*(np.exp(-time_data/tau)-
        np.exp(-4*time_data/tau))+ \
        num_pumps*pc2*(np.exp(-time_data/tau2)-np.exp(-4*time_data/tau2))

def P2_P3(time_data, offset, tauc, tau, tauc2, tau2):
    """One trap for probability function 2, one for probability function 3.
    """
    pc = 1 - np.exp(-time_data/tauc)
    pc2 = 1 - np.exp(-time_data/tauc2)
    return offset+num_pumps*pc*(np.exp(-2*time_data/tau)-
        np.exp(-3*time_data/tau))+ \
        num_pumps*pc2*(np.exp(-time_data/tau2)-np.exp(-4*time_data/tau2))

class TestTrapFit(unittest.TestCase):
    """Unit tests for trap_fit(). I avoid testing to see if output
    is as expected since I am not testing the function curve_fit() here."""
    def setUp(self):
        # make up some parameters
        self.fit_thresh = 0.9
        self.offset = 1
        self.tauc = 1e-6
        self.tauc2 = 1e-6
        #In order of amplitudes overall (given comparable tau and tau2):
        # P1 biggest, then P3, then P2
        self.tau = 7.5e-3
        self.tau2 = 8.8e-3
        self.time_data = np.logspace(-6, -2, 100)
        self.tauc_min = 0
        self.tauc_max = 1e-5
        self.tau_min = 0.7e-6
        self.tau_max = 1.3e-2
        self.offset_min = -10
        self.offset_max = 10
        self.amps1 = P1(self.time_data, self.offset, self.tauc, self.tau)
        self.amps11 = P1_P1(self.time_data, self.offset, self.tauc, self.tau,
            self.tauc2, self.tau2)
        self.amps2 = P2(self.time_data, self.offset, self.tauc, self.tau)
        self.amps12 = P1_P2(self.time_data, self.offset, self.tauc, self.tau,
            self.tauc2, self.tau2)
        self.amps22 = P2_P2(self.time_data, self.offset, self.tauc, self.tau,
            self.tauc2, self.tau2)
        self.amps3 = P3(self.time_data, self.offset, self.tauc, self.tau)
        self.amps33 = P3_P3(self.time_data, self.offset, self.tauc, self.tau,
            self.tauc2, self.tau2)
        self.amps23 = P2_P3(self.time_data, self.offset, self.tauc, self.tau,
            self.tauc2, self.tau2)
        # smaller time constant, self.tau, would fit this time range better
        self.both_a11 = {'amp': self.amps11[0:30], 't': self.time_data[0:30]}
        # a case where ALL the 'above' frames shared with the 'below' frames
        # so b/w P1 and P2, P1 would dominate (self.tau) since taus are of same
        # order of magnitude
        self.both_a12 = {'amp': self.amps12, 't': self.time_data}
        # larger time constant, self.tau2, would fit this time range better
        self.both_a22 = {'amp': self.amps22[95:], 't': self.time_data[95:]}
        # below used for scheme ==3 or 4 (self.both_a22 used for this and
        #scheme ==1 or 2)
        # smaller time constant, self.tau, would fit this time range better
        self.both_a33 = {'amp': self.amps33[0:30], 't': self.time_data[0:30]}
        # a case where ALL the 'above' frames shared with the 'below' frames
        # P3 should dominate (self.tau2)
        self.both_a23 = {'amp': self.amps23, 't': self.time_data}
        warnings.filterwarnings('ignore', category=UserWarning,
                        module='cal.tpumpanalysis.trap_fitting')

    def test_sch12_R1_bigger(self):
        """R_value1 bigger than R_value2 (schemes 1 and 2)."""
        for sch in [1,2]:
            fd = trap_fit(sch, self.amps1, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.tauc_min,
                self.tauc_max, self.offset_min, self.offset_max)
            self.assertEqual(set([1]), set(fd.keys()))
            self.assertTrue(len(fd[1]) == 1)

    def test_sch12_R2_bigger(self):
        """R_value2 bigger than R_value1 (schemes 1 and 2)."""
        for sch in [1,2]:
            fd = trap_fit(sch, self.amps2, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.tauc_min,
                self.tauc_max, self.offset_min, self.offset_max)
            self.assertEqual(set([2]), set(fd.keys()))
            self.assertTrue(len(fd[2]) == 1)

    @patch('cal.tpumpanalysis.trap_fitting.curve_fit')
    def test_sch12_2trap11(self, mock_fit):
        """1-trap fits fail, but 2-trap P1_P1 fits work."""
        for sch in [1,2]:
            badout = ([0.1,0.1,1e8], 10*np.ones([3,3])) # bad output
            out = ([self.offset, self.tauc, self.tau, self.tauc2, self.tau2],
                1e-5*np.ones([5,5]))
            mock_fit.side_effect = [badout, badout, badout, badout,
                out, out, out, out, out, out]
            fd = trap_fit(sch, self.amps11, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.tauc_min,
                self.tauc_max, self.offset_min, self.offset_max)
            self.assertEqual(set([1]), set(fd.keys()))
            self.assertTrue(len(fd[1]) == 2)

    @patch('cal.tpumpanalysis.trap_fitting.curve_fit')
    def test_sch12_2trap12(self, mock_fit):
        """1-trap fits fail, but 2-trap P1_P2 fits work."""
        for sch in [1,2]:
            #  I mock all the outputs.
            badout = ([0.1,0.1,1e8], 10*np.ones([3,3])) # bad output
            out = ([self.offset, self.tauc, self.tau, self.tauc2, self.tau2],
                1e-5*np.ones([5,5]))
            mock_fit.side_effect = [badout, badout, badout, badout,
                out, out, out, out, out, out]
            fd = trap_fit(sch, self.amps12, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.tauc_min,
                self.tauc_max, self.offset_min, self.offset_max)
            self.assertEqual(set([1,2]), set(fd.keys()))
            self.assertTrue(len(fd[1]) == 1 and len(fd[2]) == 1)

    @patch('cal.tpumpanalysis.trap_fitting.curve_fit')
    def test_sch12_2trap22(self, mock_fit):
        """1-trap fits fail, but 2-trap P2_P2 fits work."""
        for sch in [1,2]:
            badout = ([0.1,0.1,1e8], 10*np.ones([3,3])) # bad output
            out = ([self.offset, self.tauc, self.tau, self.tauc2, self.tau2],
                1e-5*np.ones([5,5]))
            mock_fit.side_effect = [badout, badout, badout, badout,
                out, out, out, out, out, out]
            fd = trap_fit(sch, self.amps22, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.tauc_min,
                self.tauc_max, self.offset_min, self.offset_max)
            self.assertEqual(set([2]), set(fd.keys()))
            self.assertTrue(len(fd[2]) == 2)

    @patch('cal.tpumpanalysis.trap_fitting.curve_fit')
    def test_sch12_none(self, mock_fit):
        """1-trap and 2-trap fits fail."""
        for sch in [1,2]:
            badout = ([0.1,0.1,1e8], 10*np.ones([3,3])) # bad output
            badout2 = ([0.1,0.1,1e8,0.1,1e8], 10*np.ones([5,5]))
            mock_fit.side_effect = [badout, badout, badout, badout,
                badout2, badout2, badout2, badout2, badout2, badout2]
            with self.assertWarns(UserWarning):
                fd = trap_fit(sch, self.amps22, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.tauc_min,
                self.tauc_max, self.offset_min, self.offset_max)
            self.assertEqual(fd, None)

    @patch('cal.tpumpanalysis.trap_fitting.curve_fit')
    def test_sch12_2trap11_both_a(self, mock_fit):
        """both_a!=None, 2-trap P1_P1 fits work."""
        for sch in [1,2]:
            #  I mock the outputs.  (And 1-trap outputs can be good; doesn't
            #matter when both_a!=None.)
            out = ([self.offset, self.tauc, self.tau],
                1e-5*np.ones([3,3]))
            out2 = ([self.offset, self.tauc, self.tau, self.tauc2, self.tau2],
                1e-5*np.ones([5,5]))
            mock_fit.side_effect = [out2, out2, out2, out2, out2, out2]
            fd = trap_fit(sch, self.amps11, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.tauc_min,
                self.tauc_max, self.offset_min, self.offset_max, self.both_a11)
            self.assertEqual(set(['a','b']), set(fd.keys()))
            self.assertEqual(set([1]), set(fd['a'].keys()))
            self.assertEqual(set([1]), set(fd['b'].keys()))
            # 't' has the lower times, so self.tau better fit b/c
            # self.tau < self.tau2
            self.assertTrue(np.abs(fd['a'][1][0][2] - self.tau) <=
                np.abs(fd['a'][1][0][2] - self.tau2))

    @patch('cal.tpumpanalysis.trap_fitting.curve_fit')
    def test_sch12_2trap12_both_a(self, mock_fit):
        """both_a!=None, 2-trap P1_P2 fits work."""
        for sch in [1,2]:
            # I mock the outputs.
            out = ([self.offset, self.tauc, self.tau],
                1e-5*np.ones([3,3]))
            out2 = ([self.offset, self.tauc, self.tau, self.tauc2, self.tau2],
                1e-5*np.ones([5,5]))
            mock_fit.side_effect = [out2, out2, out2, out2, out2, out2]
            fd = trap_fit(sch, self.amps12, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.tauc_min,
                self.tauc_max, self.offset_min, self.offset_max, self.both_a12)
            self.assertEqual(set(['a','b']), set(fd.keys()))
            # 't' is actually the whole time_data for this one, and b/c
            # of note in setUp(), self.tau fits better.
            # For cases like these where time_data is completely shared b/w
            # 'above' and 'below': noted to handle these differently in v2
            self.assertEqual(set([1]), set(fd['a'].keys()))
            self.assertEqual(set([2]), set(fd['b'].keys()))
            self.assertTrue(np.abs(fd['a'][1][0][2] - self.tau) <=
                np.abs(fd['a'][1][0][2] - self.tau2))

    @patch('cal.tpumpanalysis.trap_fitting.curve_fit')
    def test_sch12_2trap22_both_a(self, mock_fit):
        """both_a!=None, 2-trap P2_P2 fits work."""
        for sch in [1,2]:
            # I mock the outputs.
            out = ([self.offset, self.tauc, self.tau],
                1e-5*np.ones([3,3]))
            out2 = ([self.offset, self.tauc, self.tau, self.tauc2, self.tau2],
                1e-5*np.ones([5,5]))
            mock_fit.side_effect = [out2, out2, out2, out2, out2, out2]
            fd = trap_fit(sch, self.amps22, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.tauc_min,
                self.tauc_max, self.offset_min, self.offset_max, self.both_a22)
            self.assertEqual(set(['a','b']), set(fd.keys()))
            self.assertEqual(set([2]), set(fd['a'].keys()))
            self.assertEqual(set([2]), set(fd['b'].keys()))
            # 't' has the higher times, so self.tau2 better fit b/c
            # self.tau < self.tau2
            self.assertTrue(np.abs(fd['a'][2][0][2] - self.tau2) <=
                np.abs(fd['a'][2][0][2] - self.tau))

    @patch('cal.tpumpanalysis.trap_fitting.curve_fit')
    def test_sch12_both_a_none(self, mock_fit):
        """both_a!=None, 2-trap fits fail."""
        for sch in [1,2]:
            badout = ([0.1,0.1,1e8], 10*np.ones([3,3])) # bad output
            badout2 = ([0.1,0.1,1e8,0.1,1e8], 10*np.ones([5,5]))
            mock_fit.side_effect = [badout2, badout2, badout2, badout2,
                badout2, badout2]
            with self.assertWarns(UserWarning):
                fd = trap_fit(sch, self.amps22, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.tauc_min,
                self.tauc_max, self.offset_min, self.offset_max, self.both_a22)
            self.assertEqual(fd, None)

    #tests for scheme == 3 or 4 below
    def test_sch34_R2_bigger(self):
        """R_value2 bigger than R_value3 (schemes 3 and 4)."""
        for sch in [3,4]:
            fd = trap_fit(sch, self.amps2, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.tauc_min,
                self.tauc_max, self.offset_min, self.offset_max)
            self.assertEqual(set([2]), set(fd.keys()))
            self.assertTrue(len(fd[2]) == 1)

    def test_sch34_R3_bigger(self):
        """R_value3 bigger than R_value2 (schemes 3 and 4)."""
        for sch in [3,4]:
            fd = trap_fit(sch, self.amps3, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.tauc_min,
                self.tauc_max, self.offset_min, self.offset_max)
            self.assertEqual(set([3]), set(fd.keys()))
            self.assertTrue(len(fd[3]) == 1)

    @patch('cal.tpumpanalysis.trap_fitting.curve_fit')
    def test_sch34_2trap22(self, mock_fit):
        """1-trap fits fail, but 2-trap P2_P2 fits work."""
        for sch in [3,4]:
            badout = ([0.1,0.1,1e8], 10*np.ones([3,3])) # bad output
            out = ([self.offset, self.tauc, self.tau, self.tauc2, self.tau2],
                1e-5*np.ones([5,5]))
            mock_fit.side_effect = [badout, badout, badout, badout,
                out, out, out, out, out, out]
            fd = trap_fit(sch, self.amps22, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.tauc_min,
                self.tauc_max, self.offset_min, self.offset_max)
            self.assertEqual(set([2]), set(fd.keys()))
            self.assertTrue(len(fd[2]) == 2)

    @patch('cal.tpumpanalysis.trap_fitting.curve_fit')
    def test_sch34_2trap23(self, mock_fit):
        """1-trap fits fail, but 2-trap P2_P3 fits work."""
        for sch in [3,4]:
            badout = ([0.1,0.1,1e8], 10*np.ones([3,3])) # bad output
            out = ([self.offset, self.tauc, self.tau, self.tauc2, self.tau2],
                1e-5*np.ones([5,5]))
            mock_fit.side_effect = [badout, badout, badout, badout,
                out, out, out, out, out, out]
            fd = trap_fit(sch, self.amps23, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.tauc_min,
                self.tauc_max, self.offset_min, self.offset_max)
            self.assertEqual(set([2,3]), set(fd.keys()))
            self.assertTrue(len(fd[2]) == 1 and len(fd[3]) == 1)

    @patch('cal.tpumpanalysis.trap_fitting.curve_fit')
    def test_sch34_2trap33(self, mock_fit):
        """1-trap fits fail, but 2-trap P3_P3 fits work."""
        for sch in [3,4]:
            badout = ([0.1,0.1,1e8], 10*np.ones([3,3])) # bad output
            out = ([self.offset, self.tauc, self.tau, self.tauc2, self.tau2],
                1e-5*np.ones([5,5]))
            mock_fit.side_effect = [badout, badout, badout, badout,
                out, out, out, out, out, out]
            fd = trap_fit(sch, self.amps33, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.tauc_min,
                self.tauc_max, self.offset_min, self.offset_max)
            self.assertEqual(set([3]), set(fd.keys()))
            self.assertTrue(len(fd[3]) == 2)

    @patch('cal.tpumpanalysis.trap_fitting.curve_fit')
    def test_sch34_none(self, mock_fit):
        """1-trap and 2-trap fits fail."""
        for sch in [3,4]:
            badout = ([0.1,0.1,1e8], 10*np.ones([3,3])) # bad output
            badout2 = ([0.1,0.1,1e8,0.1,1e8], 10*np.ones([5,5]))
            mock_fit.side_effect = [badout, badout, badout, badout,
                badout2, badout2, badout2, badout2, badout2, badout2]
            with self.assertWarns(UserWarning):
                fd = trap_fit(sch, self.amps33, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.tauc_min,
                self.tauc_max, self.offset_min, self.offset_max)
            self.assertEqual(fd, None)

    @patch('cal.tpumpanalysis.trap_fitting.curve_fit')
    def test_sch34_2trap22_both_a(self, mock_fit):
        """both_a!=None, 2-trap P2_P2 fits work."""
        for sch in [3,4]:
            #I mock the outputs, and these are tested without mocks in main
            #unit test file.
            out = ([self.offset, self.tauc, self.tau],
                1e-5*np.ones([3,3]))
            out2 = ([self.offset, self.tauc, self.tau, self.tauc2, self.tau2],
                1e-5*np.ones([5,5]))
            mock_fit.side_effect = [out2, out2, out2, out2, out2, out2]
            fd = trap_fit(sch, self.amps22, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.tauc_min,
                self.tauc_max, self.offset_min, self.offset_max, self.both_a22)
            self.assertEqual(set(['a','b']), set(fd.keys()))
            self.assertEqual(set([2]), set(fd['a'].keys()))
            self.assertEqual(set([2]), set(fd['b'].keys()))
            # 't' has the higher times, so self.tau2 better fit b/c
            # self.tau < self.tau2
            self.assertTrue(np.abs(fd['a'][2][0][2] - self.tau2) <=
                np.abs(fd['a'][2][0][2] - self.tau))

    @patch('cal.tpumpanalysis.trap_fitting.curve_fit')
    def test_sch34_2trap23_both_a(self, mock_fit):
        """both_a!=None, 2-trap P2_P3 fits work."""
        for sch in [3,4]:
            # I mock the outputs.
            out = ([self.offset, self.tauc, self.tau],
                1e-5*np.ones([3,3]))
            out2 = ([self.offset, self.tauc, self.tau, self.tauc2, self.tau2],
                1e-5*np.ones([5,5]))
            mock_fit.side_effect = [out2, out2, out2, out2, out2, out2]
            fd = trap_fit(sch, self.amps23, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.tauc_min,
                self.tauc_max, self.offset_min, self.offset_max, self.both_a23)
            self.assertEqual(set(['a','b']), set(fd.keys()))
            # 't' is actually the whole time_data for this one, and b/c
            # of note in setUp(), self.tau2 fits better.
            # For cases like these where time_data is completely shared b/w
            # 'above' and 'below': noted to handle these differently in v2
            self.assertEqual(set([3]), set(fd['a'].keys()))
            self.assertEqual(set([2]), set(fd['b'].keys()))
            self.assertTrue(np.abs(fd['a'][3][0][2] - self.tau2) <=
                np.abs(fd['a'][3][0][2] - self.tau))

    @patch('cal.tpumpanalysis.trap_fitting.curve_fit')
    def test_sch34_2trap33_both_a(self, mock_fit):
        """both_a!=None, 2-trap P3_P3 fits work."""
        for sch in [3,4]:
            # I mock the outputs.
            out = ([self.offset, self.tauc, self.tau],
                1e-5*np.ones([3,3]))
            out2 = ([self.offset, self.tauc, self.tau, self.tauc2, self.tau2],
                1e-5*np.ones([5,5]))
            mock_fit.side_effect = [out2, out2, out2, out2, out2, out2]
            fd = trap_fit(sch, self.amps33, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.tauc_min,
                self.tauc_max, self.offset_min, self.offset_max, self.both_a33)
            self.assertEqual(set(['a','b']), set(fd.keys()))
            self.assertEqual(set([3]), set(fd['a'].keys()))
            self.assertEqual(set([3]), set(fd['b'].keys()))
            # 't' has the lower times, so self.tau better fit b/c
            # self.tau < self.tau2
            self.assertTrue(np.abs(fd['a'][3][0][2] - self.tau) <=
                np.abs(fd['a'][3][0][2] - self.tau2))

    @patch('cal.tpumpanalysis.trap_fitting.curve_fit')
    def test_sch34_both_a_none(self, mock_fit):
        """both_a!=None, 2-trap fits fail."""
        for sch in [3,4]:
            badout = ([0.1,0.1,1e8], 10*np.ones([3,3])) # bad output
            badout2 = ([0.1,0.1,1e8,0.1,1e8], 10*np.ones([5,5]))
            mock_fit.side_effect = [badout2, badout2, badout2,
                badout2, badout2, badout2]
            with self.assertWarns(UserWarning):
                fd = trap_fit(sch, self.amps33, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.tauc_min,
                self.tauc_max, self.offset_min, self.offset_max, self.both_a33)
            self.assertEqual(fd, None)

    def test_curve_fit_fail(self):
            """Returns None if curve_fit fails."""
            amps = self.amps1.copy()
            amps[0] = np.inf # causes curve_fit to fail
            for sch in [1,2,3,4]:
                with self.assertWarns(UserWarning):
                    fd = trap_fit(sch, amps, self.time_data, num_pumps,
                    self.fit_thresh, self.tau_min, self.tau_max, self.tauc_min,
                    self.tauc_max, self.offset_min, self.offset_max,
                    self.both_a33)
                self.assertTrue(fd is None)

    def test_bad_scheme(self):
        """Scheme input bad."""
        for er in [1j, None, (1.0,), [5,5], 'txt', -1, 0, 5]:
            with self.assertRaises(TypeError):
                trap_fit(er, self.amps33, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.tauc_min,
                self.tauc_max, self.offset_min, self.offset_max, self.both_a33)

    def test_bad_amps(self):
        """amps input bad."""
        for er in ut_check.oneDlist:
            with self.assertRaises(TypeError):
                trap_fit_const(1, er, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.tauc_min,
                self.tauc_max, self.offset_min, self.offset_max, self.both_a33)

    def test_len_times(self):
        """times must have a number of unique phase times longer than the
        number of fitted parameters."""
        for times in [np.array([]), np.array([1,2,3,4,5]),
            np.array([1,2,3,4,5,5])]:
            with self.assertRaises(IndexError):
                x = trap_fit_const(1, times, times, num_pumps,
                    self.fit_thresh, self.tau_min, self.tau_max, self.tauc_min,
                    self.tauc_max, self.offset_min, self.offset_max,
                    self.both_a33)

    def test_bad_times(self):
        """times input bad."""
        for er in ut_check.oneDlist:
            with self.assertRaises(TypeError):
                trap_fit_const(1, self.amps1, er, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.tauc_min,
                self.tauc_max, self.offset_min, self.offset_max, self.both_a33)

    def test_len_amps(self):
        """times and amps should have same length."""
        #self.amps1 same length as self.time_data. times has one more element.
        times = np.linspace(1e-6, 1e-2, 101)
        with self.assertRaises(ValueError):
            trap_fit_const(1, self.amps1, times, num_pumps,
            self.fit_thresh, self.tau_min, self.tau_max, self.tauc_min,
            self.tauc_max, self.offset_min, self.offset_max, self.both_a33)

    def test_num_pumps(self):
        """num_pumps input bad."""
        for er in ut_check.psilist:
            with self.assertRaises(TypeError):
                trap_fit(1, self.amps1, self.time_data, er,
                self.fit_thresh, self.tau_min, self.tau_max, self.tauc_min,
                self.tauc_max, self.offset_min, self.offset_max, self.both_a33)

    def test_fit_thresh(self):
        """fit_thresh input bad."""
        for er in ut_check.rnslist:
            with self.assertRaises(TypeError):
                trap_fit(1, self.amps1, self.time_data, num_pumps,
                er, self.tau_min, self.tau_max, self.tauc_min,
                self.tauc_max, self.offset_min, self.offset_max, self.both_a33)

    def test_fit_thresh_value(self):
        """fit_thresh must be in (0,1)."""
        with self.assertRaises(ValueError):
            trap_fit(1, self.amps1, self.time_data, num_pumps,
            1.2, self.tau_min, self.tau_max, self.tauc_min,
            self.tauc_max, self.offset_min, self.offset_max, self.both_a33)

    def test_tau_min(self):
        """tau_min input bad."""
        for er in ut_check.rnslist:
            with self.assertRaises(TypeError):
                trap_fit(1, self.amps1, self.time_data, num_pumps,
                self.fit_thresh, er, self.tau_max, self.tauc_min,
                self.tauc_max, self.offset_min, self.offset_max, self.both_a33)

    def test_tau_max(self):
        """tau_max input bad."""
        for er in ut_check.rnslist:
            with self.assertRaises(TypeError):
                trap_fit(1, self.amps1, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, er, self.tauc_min,
                self.tauc_max, self.offset_min, self.offset_max, self.both_a33)

    def test_tau_max_value(self):
        """tau_max must be > tau_min."""
        for er in [self.tau_min, (self.tau_min*.9)]:
            with self.assertRaises(ValueError):
                trap_fit(1, self.amps1, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, er, self.tauc_min,
                self.tauc_max, self.offset_min, self.offset_max, self.both_a33)

    def test_tauc_min(self):
        """tauc_min input bad."""
        for er in ut_check.rnslist:
            with self.assertRaises(TypeError):
                trap_fit(1, self.amps1, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, er,
                self.tauc_max, self.offset_min, self.offset_max, self.both_a33)

    def test_tauc_max(self):
        """tauc_max input bad."""
        for er in ut_check.rnslist:
            with self.assertRaises(TypeError):
                trap_fit(1, self.amps1, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.tauc_min,
                er, self.offset_min, self.offset_max, self.both_a33)

    def test_tauc_max_value(self):
        """tauc_max must be > tauc_min."""
        for er in [self.tauc_min, (self.tauc_min*.9)]:
            with self.assertRaises(ValueError):
                trap_fit(1, self.amps1, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.tauc_min,
                er, self.offset_min, self.offset_max, self.both_a33)

    def test_offset_min(self):
        """offset_min input bad."""
        for er in ut_check.rslist:
            with self.assertRaises(TypeError):
                trap_fit(1, self.amps1, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.tauc_min,
                self.tauc_max, er, self.offset_max, self.both_a33)

    def test_offset_max(self):
        """offset_max input bad."""
        for er in ut_check.rslist:
            with self.assertRaises(TypeError):
                trap_fit(1, self.amps1, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.tauc_min,
                self.tauc_max, self.offset_min, er, self.both_a33)

    def test_offset_max_value(self):
        """offset_max must be > offset_min."""
        for er in [self.offset_min, (self.offset_min-1)]:
            with self.assertRaises(ValueError):
                trap_fit(1, self.amps1, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.tauc_min,
                self.tauc_max, self.offset_min, er, self.both_a33)

    def test_both_a(self):
        """both_a input bad."""
        for er in [np.array([1,2]), 'foo', 1, -2.3, 0, (3,3), [2,2]]:
            with self.assertRaises(TypeError):
                trap_fit(1, self.amps1, self.time_data, num_pumps,
                self.fit_thresh, self.tau_min, self.tau_max, self.tauc_min,
                self.tauc_max, self.offset_min, self.offset_max, er)

    def test_both_a_key(self):
        """both_a doesn't have expected keys."""
        er = {'amp_bad': [1,2,3,4], 't': [.1,.2,.3,.4]}
        with self.assertRaises(KeyError):
            trap_fit(1, self.amps1, self.time_data, num_pumps,
            self.fit_thresh, self.tau_min, self.tau_max, self.tauc_min,
            self.tauc_max, self.offset_min, self.offset_max, er)

    def test_both_a_len(self):
        """both_a 'amp' and 't' must have same length."""
        er = {'amp': [1,2,3,4], 't': [.1,.2,.3,.4,.5]}
        with self.assertRaises(ValueError):
            trap_fit(1, self.amps1, self.time_data, num_pumps,
            self.fit_thresh, self.tau_min, self.tau_max, self.tauc_min,
            self.tauc_max, self.offset_min, self.offset_max, er)

def tau_temp(temp_data, E, cs):
        k = 8.6173e-5 # eV/K
        kb = 1.381e-23 # mks units
        hconst = 6.626e-34 # mks units
        Eg = 1.1692 - (4.9e-4)*temp_data**2/(temp_data+655)
        me = 9.109e-31 # kg
        mlstar = 0.1963 * me
        mtstar = 0.1905 * 1.1692 * me / Eg
        mstardc = 6**(2/3) * (mtstar**2*mlstar)**(1/3)
        vth = np.sqrt(3*kb*temp_data/mstardc) # m/s
        Nc = 2*(2*np.pi*mstardc*kb*temp_data/(hconst**2))**1.5 # 1/m^3
        # added a factor of 1e-19 so that curve_fit step size reasonable
        return np.e**(E/(k*temp_data))/(cs*Nc*vth*1e-19)

class TestFitCS(unittest.TestCase):
    """Unit tests for fit_cs(). I avoid testing to see if output
    is as expected since I am not testing the function curve_fit() here."""
    def setUp(self):
        # reasonable parameters
        self.temp_data = np.linspace(160, 180, 11) # in K
        self.cs_fit_thresh = 0.8
        self.E = 0.24 #eV
        self.cs = 5 #in 1e-19 m^2
        self.taus = tau_temp(self.temp_data, self.E, self.cs)
        self.tau_errs = 0.01*self.taus
        self.E_min = 0
        self.E_max = 5
        self.cs_min = 0
        self.cs_max = 50 # in 1e-19 m^2, or 1e-15 cm^2
        self.input_T = 190 #outside of temp_data, just for fun
        warnings.filterwarnings('ignore', category=UserWarning,
                        module='cal.tpumpanalysis.trap_fitting')

    def test_success(self):
        """Successful fit as expected."""
        _, _, _, _, Rsq, tau_input_T, _ = fit_cs(self.taus, self.tau_errs,
            self.temp_data, self.cs_fit_thresh, self.E_min, self.E_max,
            self.cs_min, self.cs_max, self.input_T)
        self.assertTrue(tau_input_T >= 0)
        self.assertTrue(Rsq >= self.cs_fit_thresh)

    def test_curve_fit_fail(self):
        """Returns None if curve_fit fails."""
        taus = self.taus.copy()
        taus[0] = np.inf # causes curve_fit to fail
        with self.assertWarns(UserWarning):
            x = fit_cs(taus, self.tau_errs,
            self.temp_data, self.cs_fit_thresh, self.E_min, self.E_max,
            self.cs_min, self.cs_max, self.input_T)
        self.assertTrue(x == (None, None, None, None, None, None, None))

    @patch('cal.tpumpanalysis.trap_fitting.curve_fit')
    def test_failure(self, mock_fit):
        """Bad fit."""
        mock_fit.side_effect = [([1e-9,2.5e-15], 10*np.ones([2,2])),
                                ([1e-9,2.5e-15], 10*np.ones([2,2]))]
        with self.assertWarns(UserWarning):
            _, _, _, _, Rsq, _, _ = fit_cs(self.taus, self.tau_errs,
            self.temp_data, self.cs_fit_thresh, self.E_min, self.E_max,
            self.cs_min, self.cs_max, self.input_T)
        self.assertTrue(Rsq < self.cs_fit_thresh)

    def test_taus(self):
        """taus input bad."""
        for er in ut_check.oneDlist:
            with self.assertRaises(TypeError):
                fit_cs(er, self.tau_errs, self.temp_data, self.cs_fit_thresh,
                self.E_min, self.E_max, self.cs_min, self.cs_max, self.input_T)

    def test_temps_len(self):
        """temps must have a unique number of temperatures longer than the
        number of fitted parameters."""
        for er in [np.array([]), np.array([1,2]), np.array([1,2,2])]:
            with self.assertWarns(UserWarning):
                # let taus and tau_errs be er so that they're of same length
                x = fit_cs(er, er, er, self.cs_fit_thresh,
                self.E_min, self.E_max, self.cs_min, self.cs_max, self.input_T)
        self.assertTrue(x == (None, None, None, None, None, None, None))

    def test_tau_errs(self):
        """tau_errs input bad."""
        for er in ut_check.oneDlist:
            with self.assertRaises(TypeError):
                fit_cs(self.taus, er, self.temp_data, self.cs_fit_thresh,
                self.E_min, self.E_max, self.cs_min, self.cs_max, self.input_T)

    def test_tau_errs_value(self):
        """taus and tau_errs must have same length."""
        #one longer and shorter than length of self.taus
        for er in [0.1*np.ones(12), 0.1*np.ones(10)]:
            with self.assertRaises(ValueError):
                fit_cs(self.taus, er, self.temp_data, self.cs_fit_thresh,
                self.E_min, self.E_max, self.cs_min, self.cs_max, self.input_T)

    def test_temps(self):
        """temps input bad."""
        for er in ut_check.oneDlist:
            with self.assertRaises(TypeError):
                fit_cs(self.taus, self.tau_errs, er, self.cs_fit_thresh,
                self.E_min, self.E_max, self.cs_min, self.cs_max, self.input_T)

    def test_temps_value(self):
        """taus and temps must have same length."""
        #one longer and shorter than length of self.taus
        for er in [0.1*np.ones(12), 0.1*np.ones(10)]:
            with self.assertRaises(ValueError):
                fit_cs(self.taus, er, self.temp_data, self.cs_fit_thresh,
                self.E_min, self.E_max, self.cs_min, self.cs_max, self.input_T)

    def test_cs_fit_thresh(self):
        """cs_fit_thresh input bad."""
        for er in ut_check.rnslist:
            with self.assertRaises(TypeError):
                fit_cs(self.taus, self.tau_errs, self.temp_data, er,
                self.E_min, self.E_max, self.cs_min, self.cs_max, self.input_T)

    def test_cs_fit_thresh_value(self):
        """cs_fit_thresh should fall in (0,1)."""
        for er in [1.2, 5]:
            with self.assertRaises(ValueError):
                fit_cs(self.taus, self.tau_errs, self.temp_data, er,
                self.E_min, self.E_max, self.cs_min, self.cs_max, self.input_T)

    def test_E_min(self):
        """E_min input bad."""
        for er in ut_check.rnslist:
            with self.assertRaises(TypeError):
                fit_cs(self.taus, self.tau_errs, self.temp_data,
                self.cs_fit_thresh, er, self.E_max, self.cs_min, self.cs_max,
                self.input_T)

    def test_E_max(self):
        """E_max input bad."""
        for er in ut_check.rnslist:
            with self.assertRaises(TypeError):
                fit_cs(self.taus, self.tau_errs, self.temp_data,
                self.cs_fit_thresh, self.E_min, er, self.cs_min, self.cs_max,
                self.input_T)

    def test_E_max_value(self):
        """E_max must be > E_min."""
        for er in [self.E_min, self.E_min*0.9]:
            with self.assertRaises(ValueError):
                fit_cs(self.taus, self.tau_errs, self.temp_data,
                self.cs_fit_thresh, self.E_min, er, self.cs_min, self.cs_max,
                self.input_T)

    def test_cs_min(self):
        """cs_min input bad."""
        for er in ut_check.rnslist:
            with self.assertRaises(TypeError):
                fit_cs(self.taus, self.tau_errs, self.temp_data,
                self.cs_fit_thresh, self.E_min, self.E_max, er, self.cs_max,
                self.input_T)

    def test_cs_max(self):
        """cs_max input bad."""
        for er in ut_check.rnslist:
            with self.assertRaises(TypeError):
                fit_cs(self.taus, self.tau_errs, self.temp_data,
                self.cs_fit_thresh, self.E_min, self.E_max, self.cs_min, er,
                self.input_T)

    def test_cs_max_value(self):
        """cs_max must be > cs_min."""
        for er in [self.cs_min, self.cs_min*0.9]:
            with self.assertRaises(ValueError):
                fit_cs(self.taus, self.tau_errs, self.temp_data,
                self.cs_fit_thresh, self.E_min, self.E_max, self.cs_min, er,
                self.input_T)

    def test_input_T(self):
        """input_T input bad."""
        for er in ut_check.rpslist:
            with self.assertRaises(TypeError):
                fit_cs(self.taus, self.tau_errs, self.temp_data,
                self.cs_fit_thresh, self.E_min, self.E_max, self.cs_min,
                self.cs_max, er)

if __name__ == '__main__':
    unittest.main()