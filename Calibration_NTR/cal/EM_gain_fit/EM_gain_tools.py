from pathlib import Path
import numpy as np
import warnings

from scipy.special import factorial, hyp0f1, erf
from scipy.optimize import minimize, Bounds
from scipy.interpolate import UnivariateSpline

# a huge-number calculation can cause this sometimes
warnings.filterwarnings('ignore', category=RuntimeWarning)

def _ENF(g, Nem):
    """
    Returns the ENF (extra noise factor) for gamma distribution noise.
    """
    return np.sqrt(2*(g-1)*g**(-(Nem+1)/Nem) + 1/g)


def _log_rn_dist(rn, mu, x):
    '''Log of normal distribution for read noise, centered at mu with standard
    deviation rn.'''
    x = x.astype(float)
    log_rn_dist = -(x-mu)**2/(2*rn**2) - np.log(np.sqrt(2*np.pi)*rn)
    return log_rn_dist


def _LogPoisson(L, x):
    '''Log of Poisson distribution for ungained electron flux, with mean value
    L.'''
    x = x.astype(float)
    out = np.zeros_like(x).astype(float)
    xless0 = np.where(x < 0)
    out[xless0] = -np.inf
    xgreat0 = np.where(x>=0)
    X = x[xgreat0]
    out[xgreat0] = -L + X*np.log(L) - np.log(factorial(X))
    # In case nans occur for huge x values, use approximation
    outinf = np.where(np.isinf(out))
    Xinf = x[outinf]
    out[outinf] = (-L + Xinf*(1 + np.log(L) - np.log(Xinf))
                   - 0.5*np.log(2*np.pi*Xinf))

    return out

def _LogGamma(n, g, x):
    '''Log of Gamma distribution (which is Erlang distribution when n is an
    integer), with EM gain g.'''
    # n>=1 and integer; g>=1
    x = x.astype(float)
    out = np.zeros_like(x).astype(float)

    xgreat1 = np.where(x >= n)
    X = x[xgreat1]
    if g > 1:
        out[xgreat1] = (-(X/g) - n*np.log(g) + (-1 + n)*np.log(X)
                        - np.log(factorial(-1 + n)))
        # for large n, the log of factorial will be np.inf
        outinf = np.where(np.isinf(out))
        Xinf = x[outinf]
        # Stirling's approximation for large n
        out[outinf] = -(Xinf/g) - n*np.log(g) + (-1 + n)*np.log(Xinf) - \
                    (n*(-1 + np.log(n)) - np.log(n)/2 + np.log(2*np.pi)/2)

    # Gamma distribution (or more specifically here, the Erlang distribution,
    # which is what it is when n is an integer) is normalized over all x from
    # 0 to infinity.  But the distribution is technically valid only for
    # integer x from n to infinity.
    # There is an analytic normalization we could apply, but it is so close to
    # 1 for all frames that will be used here (bright frames for low gain or
    # high-gain dark frames), it doesn't matter.

    xlessn = np.where(x < n)
    out[xlessn] = -np.inf # which is log(0)

    return out


def _LogPoissonGamma(L, g, x):
    '''Log of the composition of the Gamma distribution (with EM gain g)
    with the Poisson distribution, meaning the n value for the Gamma
    distribution comes from the output of the Poisson distribution with mean
    value L.  '''
    x = x.astype(float)
    out = np.zeros_like(x).astype(float)
    xless1 = np.where(x < 1)
    xless0 = np.where(x < 0)
    out[xless1] = -L

    xgreat1 = np.where(x >= 1)
    X = x[xgreat1]
    if g > 1:
        out[xgreat1] = -(L + X/g) + np.log(L/g) + np.log(hyp0f1(2,(L*X)/g))

        # In case nans occur for huge x values, use approximation
        outinf = np.where(np.isinf(out))
        Xinf = x[outinf]

        # do sum of terms a few sigma around L
        if L>1:
            R = np.arange(max(L-5*np.sqrt(L), 1), max(L+5*np.sqrt(L)+1, 2+1))
        else:
            R = np.arange(1, 10)
        logarray = np.array([(_LogGamma(i,g,Xinf)+
                              _LogPoisson(L,np.array([i]))) for i in R])
        nn = np.exp(logarray)
        SUM = np.sum(nn, axis=0)
        # in case SUM contains some zeros, to avoid RuntimeWarning, split up 
        # in this way
        zero_ind = np.where(SUM == 0)
        non_ind = np.where(SUM != 0)
        out[outinf[0][non_ind[0]]] = np.log(SUM[non_ind])
        out[outinf[0][zero_ind[0]]] = - np.inf

        # if the above didn't work, then do rough estimate:
        # take the biggest log to represent the whole sum of
        # logs; exp it, but we want the log for the return anyways
        # So it's just the max log term (for the peak of the distribution)
        # times 2*std dev (for the rough full width), then take log of that:
        outinf2 = np.where(np.isinf(out))

        Xinf2 = x[outinf2]
        logarray2 = np.array([(_LogGamma(i,g,Xinf2)+
                               _LogPoisson(L,np.array([i]))) for i in R])
        out[outinf2] = np.max(logarray2, axis=0) + np.log(2) + 0.5*np.log(L)

        out[xless0] = -np.inf # overwrites any that were written in line above
    else:
        out[xgreat1] = _LogPoisson(L, X)

    return out


def _PoissonGammaConvFFT(L,g,x,rn, mu, xbounds=None):
    '''Convolution of read noise (normal) distribution with the Poisson Gamma
    distribution.  This is theoretically what an ideal histogram of a frame
    should follow.'''
    sp=1
    if xbounds is None:
        xmin = x.min() - 1000
        xmax = x.max() + 1000
    else:
        xmin = xbounds[0] 
        xmax = xbounds[1] 
    xrange = np.arange(xmin, xmax, sp)

    PGpad = np.exp(_LogPoissonGamma(L,g,xrange))
    rnpad = np.exp(_log_rn_dist(rn,mu,xrange))
    PGfft = np.fft.fft((PGpad))
    rnfft = np.fft.fft((rnpad))
    fx = np.fft.fftfreq(PGpad.shape[0])
    # include a shift to shift 0 to the min index position
    conv = np.abs((np.fft.ifft(PGfft*rnfft*
                            np.exp(2j*np.pi*fx*(-xrange.min()*1/sp)))))

    interp = UnivariateSpline(xrange, conv, s = 0, ext=3)
    out = interp(x)
    # occasionally the interpolation may return a negative number very close to
    # 0; for a probability distribution over its domain, we want all non-zero
    # positive numbers, Corrected here 
    # by setting these to the first non-zero minimum.
    if out[out>0].size > 0:
        out[out<=0] = np.min(out[out>0])
    else:
        out[out<=0] = 0
    return out


def _TruncGaussMean(m,s,r):
    '''The mean of a truncated Gaussian (used for approximating to the normal
    distribution when there's a cut).
    m: mean of untruncated.
    s:  std dev of untruncated.
    r: cut value.'''
    try:
        out = ((m + (np.sqrt(2/np.pi)*s)/np.e**((-m + r)**2/(2.*s**2)) -
        m*erf((-m + r)/(np.sqrt(2)*s)))/(1 - erf((-m + r)/(np.sqrt(2)*s))))
    except: # if overflow occurs
        out = (r**4 + (m**2 + m*r + r**2)*s**2 - 2*s**4)/r**3
    # if r too big, out is inf; then take the series expansion for large r to
    # 3rd order
    if np.isinf(out) or np.isnan(out):
        out = (r**4 + (m**2 + m*r + r**2)*s**2 - 2*s**4)/r**3
    return out

def _TruncGaussVar(m,s,r):
    '''The variance of a truncated Gaussian (used for approximating to the
    normal distribution when there's a cut).
    m: mean of untruncated.
    s:  std dev of untruncated.
    r: cut value.'''
    try:
        out = (-((np.e**((-m + r)**2/(2.*s**2))*m + np.sqrt(2/np.pi)*s -
             np.e**((-m + r)**2/(2.*s**2))*m*erf((-m + r)/(np.sqrt(2)*s)))**2/
          (np.e**((-m + r)**2/s**2)*(-1 + erf((-m + r)/(np.sqrt(2)*s)))**2)) +
        ((2*(m + r)*s)/np.e**((-m + r)**2/(2.*s**2)) + np.sqrt(2*np.pi)*(m**2
        + s**2) -
          np.sqrt(2*np.pi)*(m**2 + s**2)*erf((-m + r)/(np.sqrt(2)*s)))/
        (np.sqrt(2*np.pi)*(1 - erf((-m + r)/(np.sqrt(2)*s)))))
    except: # if overflow occurs
        out = ((2*m + r)*s**4)/r**3
    if np.isinf(out) or np.isnan(out):
        # if r too big, out is inf; then take the series
        # expansion for large r to 3rd order
        out = ((2*m + r)*s**4)/r**3
    return out


def _EM_gain_fit_conv(frames, fluxe, gain, gmax, rn, mu, fluxe_dark,
                     lthresh, divisor=1,
                     gain_factor=0.5, lambda_factor=0.5,
                     tol=1e-16, cut=None, num_summed_frames=None, Nem=604,
                     diff_tol=1e-5):
    '''This function uses maximum likelihood estimation (MLE) to determine
    which values of ungained mean electron counts (L, mean of Poisson
    distribution) and EM gain (g, for gamma distribution)
    maximize the likelihood of the expected probability distribution
    for the frame or frames.  The distribution assumed is the composition
    of the gamma distribution with the Poisson distribution (Poisson-gamma)
    convolved with a normal distribution for the read noise.

    In general, we recommend at least 5 frames for analysis for EXCAM and
    at least 15 for LOCAM.  A single EXCAM frame
    is doable (and about 5 LOCAM frames is doable), but the result is less
    reliable in theory.  For example, a
    single illuminated EXCAM frame can be analyzed if divisor is around 20, but
    because of the higher variance in the histogram profile for 1 frame vs
    many, it is better to do multiple frames. The bigger the number of frames,
    the less variance there is in the histogram profile, and thus the closer
    to 1 divisor should be so as to refrain from losing information.

    For illuminated frames (the case of LOCAM frames and EXCAM low-gain
    frames), the illumination may be uneven across
    the frame, causing multiple local maxima in the histogram profile of the
    data, so a cut is made to analyze only the data above the mean value of
    the frames, since the biggest peak (highest incidence) is expected to be
    the highest-flux peak as well and also be the dominant influence on the
    mean.  We recommend divisor=1 for these frames, and we recommend lthresh=0
    in general.
    For high-flux illumination, there may be low-incidence counts that
    extend to high-count levels where the numerical value of the
    probability is so small that it is challenging for machine
    precision to accurately handle.  If machine precision fails, 
    an exception will be raised, and the user can
    try to increase lthresh by 1.

    However, LOCAM frames do not follow the same probability distribution as
    the EXCAM frames because the frames are summed.  It would be a convolution
    n times for n frames comprising a summed frame, and this approximates to a
    normal distribution.  For the LOCAM case, machine precision
    with respect to the effect of low lthresh is not an issue, so lthresh is
    not active when LOCAM mode is on (signaled by whether num_summed_frames is
    not None).

    For dark frames (the case of EXCAM high-gain frames),
    the tail of the distribution is important in determining
    the EM gain, and the statistics are poorer there. There may be a similar
    effect as described in the previous paragraph affecting the counts near
    the read-noise peak near the lowest-count level, so we make a cut to
    analyze only the data above the domain of the read-noise peak.  The read
    noise is the only source of counts below 0 (after converting a frame to
    e-), so our cut is the absolute value of the smallest value in
    frames.  Because of the poor stats in the tail, we recommend
    divisor=1 and lthresh=0 for dark frames.

    For high values of L*g, a normal distribution with the same mean and
    variance as the convolution of the normal distribution with the
    Poisson-gamma nicely approximates the actual distribution, and that is used
    when appropriate.

    Parameters
    ----------
    frames : array-like
        Array containing data from a frame or frames with non-unity EM gain.

    fluxe : float
        Mean number of electrons per pixel expected to be present in frames.
        This parameter is used as the initial guess for the mean for the
        Poisson distribution for the optimization process.  >= 0.

    gain : float
        EM gain expected for frames. This is used as the initial guess for the
        EM gain for the gamma distribution for the optimization process. > 1.

    rn : float
        Read noise for frames.  This is used as the fixed value for
        the standard deviation of the normal distribution used for the read
        noise for the optimization process.  >0.

    mu : float
        The mean for the normal distribution representing the read
        noise distribution. This is used as a fixed value for the
        optimization process.

    fluxe_dark : float
        The mean number of electron counts on a dark frame.  Only used in LOCAM
        mode (i.e., when num_summed_frames is not None) to account for the
        effect of bias summed frames.

    lthresh : float
        The minimium frequency for a histogram bin for the data from
        frames that is used for MLE analysis. Not active when LOCAM mode used
        (e.g., when num_summed_frames is not None).  >= 0.

    divisor : float, optional
        The size of the range of integer values found in frames is divided by
        this parameter, and the result is used as the number of bins in the
        histogram.  Defaults to 1.

    gain_factor : float, optional
        The bounds of allowed EM gain used for MLE is determined by
        gain_factor.  The upper bound is given by gain*(1+gain_factor), as long
        as that is not bigger than gmax.  The lower bound is given by
        gain*(1-gain_factor), as long as that is not smaller than 1.  Defaults
        to 0.5.  It should be > 0 and < 1, but if gain_factor is such that the
        range allowed dips below 1 or goes above gmax, those are instead used
        as the bounds.  Defaults to 0.5.

    lambda_factor : float, optional
        The bounds of allowed Poisson mean used for MLE is determined by
        lambda_factor.  The upper bound is given by
        (1+lambda_factor)*(the mean value of frames)/gain, as long as that
        is not bigger than the maximum value found in frames.  The lower bound
        is given by (1-lambda_factor)*(the mean value of frames)/gain, as long
        as that is not smaller than 1. It should be > 0 and < 1, but if
        gain_factor is such that the range allowed dips below 0 or goes above
        the maximum value found in frames, those are instead used as the
        bounds.  Defaults to 0.5.

    tol : float, optional
        Tolerance used in the MLE analysis, used in scipy.optimize.minimize.
        Defaults to 1e-16.

    cut : float, optional
        If the user wants to apply MLE over a subset of the domain of the
        probability distribution, the user can specify cut, which takes the
        domain above this value and normalizes the probability over that
        region.  This is useful if part of the histogram is not useable below a
        certain count level.  Defaults to None, which means no cut is employed.

    num_summed_frames : int, optional
        Number of frames in a LOCAM summed frame.  If EXCAM in mind, leave as
        None, which means it will be 1 in the function (i.e., have no effect).
        Defaults to None.

    Nem : int, optional
        Number of gain registers.  Defaults to 604.

    diff_tol : float, optional
        The maximum difference allowed between the normal distribution
        and the theoretically expected convolution of the read noise with
        the Poisson-gamma distribution, between any given point on one and the
        corresponding point on the other.  Defaults to 1e-5.

    Returns
    -------
    EMgain : float
            EM gain value found via MLE.

    e_mean : float
        Ungained mean electron counts found via MLE.

    success : bool
        If success is True, the optimization process for MLE was
        successful.  If success is False, the optimization process for MLE
        was unsuccessful.
    '''

    f = np.ravel(frames)
    divisor = min(divisor, (f.max()-f.min())/2) #ensures at least 2 bins
    y_vals, bin_edges = np.histogram(f, bins=int((f.max()-f.min())/divisor))
    if divisor > 1:
        x_vals = (bin_edges + np.roll(bin_edges, -1))/2
    x_vals = bin_edges[:-1]

    lthresh = min(lthresh, y_vals.max())
    good_ind = np.where(y_vals >= lthresh)
    yv = y_vals[good_ind]
    xv = x_vals[good_ind]

    if cut is not None:
        cut = min(cut, xv.max())
        gind = np.where(xv >= cut)
        yv = yv[gind]
        xv = xv[gind]

    fluxe_l = max((1-lambda_factor)*f.mean()/gain, np.finfo(float).eps)
    fluxe_u = min((1+lambda_factor)*f.mean()/gain, f.max())
    gain_l = max(1+np.finfo(float).eps, gain*(1-gain_factor))
    gain_u = min(gain*(1+gain_factor), gmax)
    if fluxe_l >= fluxe_u:
        raise ValueError('Lower bound for fluxe >= upper bound for fluxe. '
                         'Some frame set is not good '
                         '(has a max value less than float '
                         'machine tolerance or a negative mean value) '
                         'or the input commanded gain '
                         'is very inaccurate or the eperdn is off.')
    if gain_l >= gain_u:
        raise ValueError('Lower bound for gain >= upper bound for gain. '
                         'Consider a higher input for the max '
                         'gain or a lower input for the commanded gain (if '
                         'no unity gain frames provdied in \'locam\' or '
                         '\'excam_low\' case) or '
                         'a different eperdn (or '
                         'a higher exposure time, lower CIC, or '
                         'lower dark current if excam_high).')
    # make sure initial guesses fall between bounds
    fluxe = max(fluxe, fluxe_l)
    fluxe = min(fluxe, fluxe_u)
    gain = max(gain, gain_l)
    gain = min(gain, gain_u)

    # if Poisson-gamma distribution similar enough to normal
    # distribution, then go with the exact MLE solution for normal
    # distribution; this will be the case for large L*g.  If the
    # distributions differ by less than 1e-5 in absolute value for
    # all data points, then it is good enough to go with normal
    # since the mean should go like L*g, and
    # 1e-5 * highest gain possible will will be a change to the
    # hundredths place in gain, which isn't siginificant

    # the effect of non-zero read noise mean: utterly negligible effect
    # on overall mean when Poisson-gamma convolved with read noise
    # if similar to normal distribution

    # to use later
    yes_locam = num_summed_frames is not None

    if num_summed_frames is None:
        num_summed_frames = 1 #for EXCAM case
    # num_summed_frames*(fluxe for single frame)*gain
    exp_mean_l = fluxe_l * gain_l
    exp_mean_u = fluxe_u * gain_u
    # num_summed_frames*(rn**2 + g**2*ENF**2*(fluxe for single frame))
    exp_var_l = (num_summed_frames*rn**2+(gain_l*
                                    _ENF(gain_l,Nem))**2*fluxe_l)
    exp_var_u = (num_summed_frames*rn**2+(gain_u*
                                    _ENF(gain_u,Nem))**2*fluxe_u)
    normal_l = np.exp(_log_rn_dist(np.sqrt(exp_var_l), exp_mean_l,
                                    xv))
    normal_u = np.exp(_log_rn_dist(np.sqrt(exp_var_u), exp_mean_u,
                                    xv))
    # compare those to the convolution of Poisson-gamma with read noise
    a_l = _PoissonGammaConvFFT(fluxe_l, gain_l, xv, rn, mu)

    a_u = _PoissonGammaConvFFT(fluxe_u, gain_u, xv, rn, mu)

    diff_l = np.abs(normal_l - a_l).max()
    diff_u = np.abs(normal_u - a_u).max()

    bounds = Bounds(lb=np.array([fluxe_l, gain_l]),
                    ub=np.array([fluxe_u, gain_u]))

    if (diff_l <= diff_tol and diff_u <= diff_tol) or (yes_locam == True):

        # for this Gaussian approx, don't apply lthresh, just cut:
        if cut is not None:
            cut = min(cut, x_vals.max())
            cutind = np.where(x_vals >= cut)
            ycut = y_vals[cutind]
            xcut = x_vals[cutind]
        else:
            ycut = y_vals
            xcut = x_vals

        # mean and variance for binned and potentially cut distribution
        m = np.sum(ycut*xcut)/np.sum(ycut)
        v = np.sum(ycut*(xcut - m)**2)/np.sum(ycut)

        if yes_locam and cut is None:
            def equations(p, Ld, m, v, n, rn, N):
                # variance takes into account the combined variance from the
                # frames and the bias frames that were subtracted
                L, g = p
                mean = (L-Ld)*g
                var = 2*n*rn**2 + (g*_ENF(g,N))**2*(L+Ld)
                mean_eq = mean - m
                var_eq = var - v
                return np.abs(mean_eq) +np.abs(var_eq)
        elif yes_locam and cut is not None:
            def equations(p, Ld, m, v, n, rn, N):
                # variance takes into account the combined variance from the
                # frames and the bias frames that were subtracted
                L, g = p
                mean = (L-Ld)*g
                var = 2*n*rn**2 + (g*_ENF(g,N))**2*(L+Ld)
                mean_eq = _TruncGaussMean(mean,np.sqrt(var),cut) - m
                var_eq = _TruncGaussVar(mean, np.sqrt(var),cut) - v
                return np.abs(mean_eq) +np.abs(var_eq)
        # EXCAM cases where no bias/dark subtracted
        elif not yes_locam and cut is None:
            def equations(p, Ld, m, v, n, rn, N): # Ld a dud here
                L, g = p
                mean = L*g
                var = n*rn**2 + (g*_ENF(g,N))**2*(L)
                mean_eq = mean - m
                var_eq = var - v
                return np.abs(mean_eq) +np.abs(var_eq)
        elif not yes_locam and cut is not None:
            def equations(p, Ld, m, v, n, rn, N): # Ld is a dud here
                L, g = p
                mean = L*g
                var = n*rn**2 + (g*_ENF(g,N))**2*(L)
                mean_eq = _TruncGaussMean(mean,np.sqrt(var),cut) - m
                var_eq = _TruncGaussVar(mean, np.sqrt(var),cut) - v
                return np.abs(mean_eq) +np.abs(var_eq)

        res = minimize(equations, x0=(fluxe,gain), bounds=bounds,
                                args=(fluxe_dark, m, v, num_summed_frames,
                                      rn, Nem), tol=tol, method="Powell")

    else:
        def _loglik(v):
            '''The negative of the log of the likelihood.'''
            L, g = v
            ar = _PoissonGammaConvFFT(L,g,xv,rn,mu,
                                        xbounds=(x_vals.min(), x_vals.max()))

            if ar.min() <= 0 or np.isnan(ar.any()):
                # read noise so distinct from Poisson-gamma that it doesn't
                # register; so go back to just Poisson-gamma
                logar = _LogPoissonGamma(L,g,xv)
            else:
                logar = np.log(ar)

            if cut is not None:
                norm = np.sum(_PoissonGammaConvFFT(L,g,np.arange(cut,
                    x_vals.max()),rn,mu,xbounds=(x_vals.min(), x_vals.max())))
                # same reasoning as above
                if norm <= 0 or np.isnan(norm):
                    norm = np.sum(np.exp(_LogPoissonGamma(L,g,np.arange(cut,
                                                            x_vals.max()))))
                logar = logar - np.log(norm)

            out = -np.sum(yv*logar)
            if np.isinf(out):
                raise ValueError('The log of the likelihood was calculated to '
                                    'be -infinity in an iteration. Try '
                                    'increasing lthresh by 1.')

            return out

        res = minimize(fun=_loglik,
                x0=np.array([fluxe, gain]),
                bounds=bounds,
                tol=tol,
                method="Powell"
                )

    e_mean = res.x[0]
    EMgain = res.x[1]
    success = res.success
    lam_b_g = (fluxe_l, fluxe, fluxe_u)
    gain_b_g = (gain_l, gain, gain_u)

    return EMgain, e_mean, success, lam_b_g, gain_b_g

if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    from emccd_detect.emccd_detect import EMCCDDetect
    from scipy.stats import chisquare, chi2
    from astropy.io import fits
    from cal.util.gsw_process import Process
    from cal.util.read_metadata import Metadata as MetadataWrapper
    from EM_gain_fitting import EMGainFit

    here = os.path.abspath(os.path.dirname(__file__))
    meta_path = Path(here,'..', 'util', 'metadata.yaml')
    nonlin_path = Path(here, '..', 'util', 'nonlin_sample.csv')
    meta = MetadataWrapper(meta_path)
    image_rows, image_cols, r0c0 = meta._unpack_geom('image')

    # simulating LOCAM summed frame
    num_summed_frames = 5 #420 to have same number of entries as 1 EXCAM frame
    num_frames = 10000
    frametime = 0.000441
    flux = 1000
    fluxmap = flux*np.ones((50,50)) #for LOCAM size, photons/s

    fwc_em_e = 109000 #e-
    gain= 20
    fwc_pp_e = 98000 #e-
    dark_current = 8.33e-4 #e-/pix/s
    cic=0.02  # e-/pix/frame
    read_noise=110 # e-/pix/frame
    bias= 0#10000 # e-
    qe=0.9  # quantum efficiency, e-/photon
    cr_rate=0.  # hits/cm^2/s
    pixel_pitch=13e-6  # m
    eperdn =1#7 # e-/DN conversion
    nbits=64 # number of ADU bits
    numel_gain_register=604 #number of gain register elements
    fluxe = (flux*qe + dark_current)*frametime + cic

    sim_data = True

    if sim_data:
        emccd = EMCCDDetect(
                em_gain=gain,
                full_well_image=fwc_pp_e,
                full_well_serial=fwc_em_e,
                dark_current=dark_current,
                cic=cic,
                read_noise=read_noise,
                bias=bias,
                qe=qe,
                cr_rate=cr_rate,
                pixel_pitch=pixel_pitch,
                eperdn=eperdn,
                nbits=nbits,
                numel_gain_register=numel_gain_register,
                )

        def save_files(dir_name, fname, fr):
            hdr = fits.Header()
            prim = fits.PrimaryHDU(header=hdr)
            img = fits.ImageHDU(fr)
            hdul = fits.HDUList([prim, img])
            hdul.writeto(Path(here, 'data', 'locam', 'simulated', dir_name,
                            fname), overwrite=True)

        for j in range(num_summed_frames):

            frames = np.zeros_like(fluxmap)
            bias = np.zeros_like(fluxmap)
            unity_frames = np.zeros_like(fluxmap)

            for i in range(num_frames):
                # illuminated non-unity gain
                fr = emccd.sim_sub_frame(fluxmap, frametime).astype(float)
                frames += fr
                # bias frames at non-unity gain
                bias_fr = emccd.sim_sub_frame(np.zeros_like(fluxmap),
                                              frametime).astype(float)
                bias += bias_fr

            save_files('brights_G', 'brightsG_summed_'+str(num_frames)+
                    '_eperdn_'+str(eperdn)+'_comgain_'+str(gain)+'_'+
                    str(j)+'.fits',frames)
            save_files('bias_G', 'biasG_summed_'+str(num_frames)+
                    '_eperdn_'+str(eperdn)+'_comgain_'+str(gain)+'_'+
                    str(j)+'.fits', bias)

        e = EMGainFit(cam_mode='locam', eperdn=eperdn, bias_offset=None,
                      com_gain=gain)
        e.num_summed_frames = num_frames
        directory = Path(here,'data', 'locam', 'simulated', 'brights_G')
        bias_directory = Path(here,'data', 'locam', 'simulated', 'bias_G')

        frames = e.read_in_locam_files(directory, bias_directory, gain)

        y_vals, bin_edges = np.histogram(frames, bins=int(frames.max()-
                                                          frames.min()))
        x_vals = bin_edges[:-1]
        good_ind = np.where(y_vals > 0)
        yv = y_vals[good_ind]
        xv = x_vals[good_ind]
        EMgain, e_mean, success = e.EM_gain_fit(frames)