import os, h5py
import numpy as np 
import fsps
from astropy.cosmology import Planck13

from provabgs import infer as Infer
from provabgs import models as Models


ssp = fsps.StellarPopulation(
    zcontinuous=1, # interpolate metallicities
    sfh=0,         # tabulated SFH
    dust_type=2,   # calzetti(2000)
    imf_type=1)    # chabrier IMF

# rebin NMF bases to same t-space
nmf = Models.NMF()
t_mid = np.array([2.5000e-03, 1.0000e-02, 2.0000e-02, 3.0000e-02, 4.0000e-02,
    5.0000e-02, 6.0000e-02, 7.0000e-02, 8.0000e-02, 9.0000e-02,
    1.1000e-01, 1.5000e-01, 2.0000e-01, 2.5000e-01, 3.0000e-01,
    3.5000e-01, 4.0000e-01, 4.5000e-01, 5.1250e-01, 6.0000e-01,
    7.0000e-01, 8.0000e-01, 9.0000e-01, 1.0375e+00, 1.2500e+00,
    1.5000e+00, 1.7500e+00, 2.0000e+00, 2.2500e+00, 2.5000e+00,
    2.7500e+00, 3.0000e+00, 3.2500e+00, 3.5000e+00, 3.7500e+00,
    4.0625e+00, 4.5000e+00, 5.0000e+00, 5.5000e+00, 6.0000e+00,
    6.5000e+00, 7.0000e+00, 7.5000e+00, 8.0000e+00, 8.5000e+00,
    9.0000e+00, 9.5000e+00, 1.0000e+01, 1.0500e+01, 1.1000e+01,
    1.1500e+01, 1.2000e+01, 1.2500e+01, 1.3000e+01, 1.3500e+01])
dt = np.array([0.005, 0.01 , 0.01 , 0.01 , 0.01 , 0.01 , 0.01 , 0.01 , 0.01 ,
    0.01 , 0.03 , 0.05 , 0.05 , 0.05 , 0.05 , 0.05 , 0.05 , 0.05 ,
    0.075, 0.1  , 0.1  , 0.1  , 0.1  , 0.175, 0.25 , 0.25 , 0.25 ,
    0.25 , 0.25 , 0.25 , 0.25 , 0.25 , 0.25 , 0.25 , 0.25 , 0.375,
    0.5  , 0.5  , 0.5  , 0.5  , 0.5  , 0.5  , 0.5  , 0.5  , 0.5  ,
    0.5  , 0.5  , 0.5  , 0.5  , 0.5  , 0.5  , 0.5  , 0.5  , 0.5  ,
    0.5  ])
rebinned_bases = np.array([np.interp(t_mid, nmf._nmf_t_lb_sfh, basis) for basis in nmf._nmf_sfh_basis])
rebinned_bases = np.array([basis / np.sum(basis * dt) for basis in rebinned_bases])


def SFH(theta, name='tau'):
    ''' tau star formation history for given parameter values.
    '''
    if 'tau' in name:
        # tau model SFR ~ exp(−(t−ti)/tau)
        # delay tau model SFR ~ (t-ti) exp(-(t-ti)/tau)
        tau_sfh, sf_start = theta

        # tau or delayed-tau
        power = 1
        if 'delay' in name: power = 2

        normalized_t = (-(t_mid - sf_start)) / tau_sfh

        sfr_tau = ((normalized_t * tau_sfh) **(power - 1) * np.exp(-normalized_t))
        sfh_tau = sfr_tau * dt

        sfh = np.zeros(len(t_mid))
        sfh[t_mid < sf_start] += sfh_tau[t_mid < sf_start]
        return sfh/np.sum(sfh)

    elif 'dirichlet' in name: # Dirichlet SFH
        assert np.isclose(np.sum(theta), 1)
        # log-space lookback time bins (for dirichlet prior)
        #_tlb_log = np.array([0.] + list(np.logspace(-2, np.log10(t_high[-1]), 7)))
        #i_tlb_log_low = np.array([np.argmin(np.abs(t_low - _t)) for _t in _tlb_log[:-1]])
        #i_tlb_log_high = np.array(list(i_tlb_log_low[1:]) + [len(t_high)-1])
        #tlb_log = np.concatenate([t_low[i_tlb_log_low], [t_high[-1]]])

        tlb_log = np.array([ 0., 0.015, 0.035, 0.125, 0.375, 1.125, 4.25 , 13.75])
        _dts = np.diff(tlb_log)

        sfh = np.zeros(len(t_mid))
        # rebin to t_low, t_mid, t_high
        for ilow, ihigh, tt, _dt in zip([0, 2, 4, 11, 16, 24, 36], [2, 4, 11, 16, 24, 36, 54], theta, _dts):
            sfh[ilow:ihigh] = tt / _dt
        sfh *= dt
        return sfh

    elif 'nmf' in name:
        assert np.isclose(np.sum(theta), 1)

        sfr = np.sum(np.array(theta)[:,None] * rebinned_bases, axis=0)
        sfh = sfr * dt
        return sfh/np.sum(sfh)


def SED(theta, name='tau'):
    # get SFH
    sfh = SFH(theta[1:], name=name)
    assert np.isclose(np.sum(sfh), 1.)

    # sfh parameters
    ssp.params['logzsol']  = 0 # log(z/zsun)
    ssp.params['dust2']    = 0.3  # dust2 parameter in fsps

    for i, tage in enumerate(t_mid):
        m = sfh[i] # mass formed in this bin
        if m == 0 and i != 0: continue

        wave_rest, lum_i = ssp.get_spectrum(tage=np.clip(tage, 1e-8, None), peraa=True) # in units of Lsun/AA

        # note that this spectrum is normalized such that the total formed
        # mass = 1 Msun
        if i == 0: lum_ssp = np.zeros(len(wave_rest))
        lum_ssp += m * lum_i

    return lum_ssp * 10**theta[0]
