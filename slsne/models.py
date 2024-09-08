"""
This file contains functions to replicate the MOSFiT models to
create slsnni light curves.

Warning: This code is intended to be a copy of the original MOSFiT
         function, and it thefore uses the dust excintion model
         model from MOSFiT, not the more recent Gordon23 model.

Warning: All magnitudes derived here are given in the AB system.

Written by Sebastian Gomez, 2024.
"""

# Import necessary packages
import numpy as np
from astropy.cosmology import Planck18 as cosmo
from scipy.interpolate import interp1d
from astropy import constants as c
from astropy import units as u
from extinction import odonnell94
from extinction import apply as eapp
import numexpr as ne
import os

# Needed constants
NI56_LUM = 6.45e43
CO56_LUM = 1.45e43
NI56_LIFE = 8.8
CO56_LIFE = 111.3
DAY_CGS = 86400.0
C_CGS = 29979245800.0
KM_CGS = 100000.0
M_SUN_CGS = 1.9884754153381438e+33
FOUR_PI = 12.566370614359172
N_INT_TIMES = 100
MIN_LOG_SPACING = -3
MW_RV = 3.1
LYMAN = 912.0
MPC_CGS = 3.085677581467192e+24
MAG_FAC = 2.5
AB_ZEROPOINT = 3631
JY_TO_GS2 = 1.0E-23
ANGSTROM_CGS = 1.0E-8
LIGHTSPEED = 2.9979245800E10
DIFF_CONST = 2.0 * M_SUN_CGS / (13.7 * C_CGS * KM_CGS)
TRAP_CONST = 3.0 * M_SUN_CGS / (FOUR_PI * KM_CGS ** 2)
STEF_CONST = (FOUR_PI * c.sigma_sb).cgs.value
RAD_CONST = KM_CGS * DAY_CGS
C_CONST = c.c.cgs.value
FLUX_CONST = FOUR_PI * (2.0 * c.h * c.c ** 2 * np.pi).cgs.value * u.Angstrom.cgs.scale
X_CONST = (c.h * c.c / c.k_B).cgs.value
STEF_CONST = (4.0 * np.pi * c.sigma_sb).cgs.value
N_TERMS = 1000
ANG_CGS = u.Angstrom.cgs.scale
KEV_CGS = u.keV.cgs.scale
H_C_CGS = c.h.cgs.value * c.c.cgs.value
FLUX_STD = AB_ZEROPOINT * JY_TO_GS2 / ANGSTROM_CGS * LIGHTSPEED

# Get directory with reference data
current_file_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_file_dir, 'ref_data')

# Aliases for common bands
band_map = {'u': 'SLOAN_SDSS.u_AB',
            'g': 'SLOAN_SDSS.g_AB',
            'r': 'SLOAN_SDSS.r_AB',
            'i': 'SLOAN_SDSS.i_AB',
            'z': 'SLOAN_SDSS.z_AB',
            'y': 'PAN-STARRS_PS1.y_AB',
            'U': 'Generic_Bessell.U_AB',
            'V': 'Generic_Bessell.V_AB',
            'B': 'Generic_Bessell.B_AB',
            'R': 'Generic_Bessell.R_AB',
            'I': 'Generic_Bessell.I_AB',
            'J': '2MASS_2MASS.J_AB',
            'H': '2MASS_2MASS.H_AB',
            'Ks': '2MASS_2MASS.Ks_AB',
            'swift_B': 'Swift_UVOT.B_AB',
            'swift_U': 'Swift_UVOT.U_AB',
            'swift_UVM2': 'Swift_UVOT.UVM2_AB',
            'swift_UVW1': 'Swift_UVOT.UVW1_AB',
            'swift_UVW2': 'Swift_UVOT.UVW2_AB',
            'swift_V': 'Swift_UVOT.V_AB',
            'W1': 'WISE_WISE.W1_AB',
            'W2': 'WISE_WISE.W2_AB'}


def nickelcobalt(times, fnickel, mejecta, rest_t_explosion):
    """
    This function calculates the luminosity of a nickel-cobalt powered
    supernova light curve.

    Parameters
    ----------
    times : numpy.ndarray
        The times at which to calculate the luminosity in days.
    fnickel : float
        The mass fraction of nickel in the ejecta.
    mejecta : float
        The total ejecta mass in solar masses.
    rest_t_explosion : float
        The time of explosion in rest frame days.

    Returns
    -------
    luminosities : numpy.ndarray
        The luminosity at each time in erg/s.
    """

    # Get Nickel mass
    mnickel = fnickel * mejecta

    # Calculations from 1994ApJS...92..527N
    ts = np.empty_like(times)
    t_inds = times >= rest_t_explosion
    ts[t_inds] = times[t_inds] - rest_t_explosion

    luminosities = np.zeros_like(times)
    luminosities[t_inds] = mnickel * (
        NI56_LUM * np.exp(-ts[t_inds] / NI56_LIFE) +
        CO56_LUM * np.exp(-ts[t_inds] / CO56_LIFE))

    # Make sure nan's are zero
    luminosities[np.isnan(luminosities)] = 0.0

    return luminosities


def magnetar(times, Pspin, Bfield, Mns, thetaPB, rest_t_explosion):
    """
    This function calculates the luminosity of a magnetar powered
    supernova light curve.

    Parameters
    ----------
    times : numpy.ndarray
        The times at which to calculate the luminosity in days.
    Pspin : float
        The spin period of the magnetar in milliseconds.
    Bfield : float
        The magnetic field of the magnetar in units of 10^14 Gauss.
    Mns : float
        The mass of the neutron star in solar masses.
    thetaPB : float
        The angle between the magnetic and rotation axes in radians.
    rest_t_explosion : float
        The time of explosion in rest frame days.

    Returns
    -------
    luminosities : numpy.ndarray
        The luminosity at each time in erg/s.
    """

    # Rotational Energy
    # E_rot = 1/2 I (2pi/P)^2, unit = erg
    Ep = 2.6e52 * (Mns / 1.4) ** (3. / 2.) * Pspin ** (-2)

    # tau_spindown = P/(dP/dt), unit = s
    # Magnetic dipole: power = 2/(3c^3)*(R^3 Bsin(theta))^2 * (2pi/P)^4
    # Set equal to -d/dt(E_rot) to derive tau
    tp = 1.3e5 * Bfield ** (-2) * Pspin ** 2 * (
        Mns / 1.4) ** (3. / 2.) * (np.sin(thetaPB)) ** (-2)

    ts = [
        np.inf
        if rest_t_explosion > x else (x - rest_t_explosion)
        for x in times
    ]

    # From Ostriker and Gunn 1971 eq 4
    luminosities = [2 * Ep / tp / (
        1. + 2 * t * DAY_CGS / tp) ** 2 for t in ts]
    luminosities = [0.0 if np.isnan(x) else x for x in luminosities]

    return luminosities


def total_luminosity(times, fnickel, mejecta, Pspin, Bfield, Mns, thetaPB, rest_t_explosion):
    """
    This function calculates the total luminosity of a supernova light curve by
    summing the nickel-cobalt and magnetar contributions.

    Parameters
    ----------
    times : numpy.ndarray
        The times at which to calculate the luminosity in days.
    fnickel : float
        The mass fraction of nickel in the ejecta.
    mejecta : float
        The total ejecta mass in solar masses.
    Pspin : float
        The spin period of the magnetar in milliseconds.
    Bfield : float
        The magnetic field of the magnetar in units of 10^14 Gauss.
    Mns : float
        The mass of the neutron star in solar masses.
    thetaPB : float
        The angle between the magnetic and rotation axes in radians.
    rest_t_explosion : float
        The time of explosion in rest frame days.

    Returns
    -------
    luminosities : numpy.ndarray
        The luminosity at each time in erg/s.
    """
    nickel_lum = nickelcobalt(times, fnickel, mejecta, rest_t_explosion)
    magnetar_lum = magnetar(times, Pspin, Bfield, Mns, thetaPB, rest_t_explosion)
    luminosities = nickel_lum + magnetar_lum
    return luminosities


def diffusion(times, input_luminosities, kappa, kappa_gamma, mejecta, v_ejecta, rest_t_explosion):
    """
    This function calculates the diffusion of the light from a supernova.

    Parameters
    ----------
    times : numpy.ndarray
        The times at which to calculate the luminosity in days.
    input_luminosities : numpy.ndarray
        The luminosity at each time in erg/s.
    kappa : float
        The opacity of the ejecta in cm^2/g.
    kappa_gamma : float
        The opacity of the gamma-rays in cm^2/g.
    mejecta : float
        The total ejecta mass in solar masses.
    v_ejecta : float
        The ejecta velocity in km/s.
    rest_t_explosion : float
        The time of explosion in rest frame days.

    Returns
    -------
    luminosities : numpy.ndarray
        The luminosity at each time in erg/s.
    """

    # Calculate the diffusion timescale
    tau_diff = np.sqrt(DIFF_CONST * kappa * mejecta / v_ejecta) / DAY_CGS
    trap_coeff = (TRAP_CONST * kappa_gamma * mejecta / (v_ejecta ** 2)) / DAY_CGS ** 2
    td2, A = tau_diff ** 2, trap_coeff

    # Times since explosion
    times_since_explosion = times-rest_t_explosion

    # Interpolate the input luminosities
    tau_diff = np.sqrt(DIFF_CONST * kappa * mejecta / v_ejecta) / DAY_CGS
    trap_coeff = (TRAP_CONST * kappa_gamma * mejecta / (v_ejecta ** 2)) / DAY_CGS ** 2
    td2, A = tau_diff ** 2, trap_coeff

    # Calculate the luminosities
    luminosities = np.zeros_like(times_since_explosion)
    min_te = min(times_since_explosion)
    tb = max(0.0, min_te)
    linterp = interp1d(times_since_explosion, input_luminosities, copy=False, assume_sorted=True)

    # Interpolate the input luminosities
    lu = len(times_since_explosion)
    num = int(round(N_INT_TIMES / 2.0))
    lsp = np.logspace(np.log10(tau_diff / times_since_explosion[-1]) + MIN_LOG_SPACING, 0, num)
    xm = np.unique(np.concatenate((lsp, 1 - lsp)))

    # Calculate the integral
    int_times = np.clip(tb + (times_since_explosion.reshape(lu, 1) - tb) * xm, tb, times_since_explosion[-1])
    int_te2s = int_times[:, -1] ** 2
    int_lums = linterp(int_times)  # noqa: F841
    int_args = int_lums * int_times * np.exp((int_times ** 2 - int_te2s.reshape(lu, 1)) / td2)
    int_args[np.isnan(int_args)] = 0.0

    # Return the final luminosities
    uniq_lums = np.trapz(int_args, int_times)

    # Make sure they are positive
    int_te2s[int_te2s <= 0] = np.nan
    luminosities = uniq_lums * (-2.0 * np.expm1(-A / int_te2s) / td2)
    luminosities[np.isnan(luminosities)] = 0.0

    return luminosities


def photosphere(times, luminosities, v_ejecta, temperature, rest_t_explosion):
    """
    This function calculates the photospheric radius and temperature of a
    light curve.

    Parameters
    ----------
    times : numpy.ndarray
        The times at which to calculate the luminosity in days.
    luminosities : numpy.ndarray
        The luminosity at each time in erg/s.
    v_ejecta : float
        The ejecta velocity in km/s.
    temperature : float
        The temperature floor of the photosphere in Kelvin.
    rest_t_explosion : float
        The time of explosion in rest frame days.

    Returns
    -------
    rphot : numpy.ndarray
        The photospheric radius at each time in cm.
    Tphot : numpy.ndarray
        The photospheric temperature at each time in Kelvin.
    """

    # Calculate radius squared
    radius2_in = [(RAD_CONST * v_ejecta * max(x - rest_t_explosion, 0.0)) ** 2 for x in times]
    rec_radius2_in = [
        x / (STEF_CONST * temperature ** 4)
        for x in luminosities
    ]
    rphot = []
    Tphot = []
    for li, lum in enumerate(luminosities):

        radius2 = radius2_in[li]
        rec_radius2 = rec_radius2_in[li]
        if lum == 0.0:
            temperature_out = 0.0
        elif radius2 < rec_radius2:
            temperature_out = (lum / (STEF_CONST * radius2)) ** 0.25
        else:
            radius2 = rec_radius2
            temperature_out = temperature

        rphot.append(np.sqrt(radius2))

        Tphot.append(temperature_out)

    return np.array(rphot), np.array(Tphot)


def mod_blackbody(lam, T, R2, sup_lambda, power_lambda):
    '''
    Calculate the corresponding blackbody radiance for a set
    of wavelengths given a temperature and radiance and a
    suppresion factor

    Parameters
    ---------------
    lam: Reference wavelengths in Angstroms
    T:   Temperature in Kelvin
    R2:   Radius in cm, squared

    Output
    ---------------
    Spectral radiance in units of erg/s/Angstrom

    (calculation and constants checked by Sebastian Gomez)
    '''

    # Planck Constant in cm^2 * g / s
    h = 6.62607E-27
    # Speed of light in cm/s
    c = 2.99792458E10

    # Convert wavelength to cm
    lam_cm = lam * 1E-8

    # Boltzmann Constant in cm^2 * g / s^2 / K
    k_B = 1.38064852E-16

    # Calculate Radiance B_lam, in units of (erg / s) / cm ^ 2 / cm
    if T > 0:
        exponential = (h * c) / (lam_cm * k_B * T)
        B_lam = ((2 * np.pi * h * c ** 2) / (lam_cm ** 5)) / (np.exp(exponential) - 1)
    else:
        B_lam = np.zeros_like(lam_cm) * np.nan

    # Multiply by the surface area
    A = 4*np.pi*R2

    # Output radiance in units of (erg / s) / Angstrom
    Radiance = B_lam * A / 1E8

    # Apply Supression below sup_lambda wavelength
    blue = lam < sup_lambda
    Radiance[blue] *= (lam[blue]/sup_lambda)**power_lambda

    return Radiance


def blackbody_supressed(times, luminosities, rphot, Tphot, cutoff_wavelength, alpha, sample_wavelengths, redshift):
    """
    This function calculates the blackbody spectrum using a modified blackbody fuction that is
    suppressed bluewards of a certain wavelength.

    Parameters
    ----------
    times : numpy.ndarray
        The times at which to calculate the luminosity in days.
    luminosities : numpy.ndarray
        The luminosity at each time in erg/s.
    rphot : numpy.ndarray
        The photospheric radius at each time in cm.
    Tphot : numpy.ndarray
        The photospheric temperature at each time in Kelvin.
    cutoff_wavelength : float
        The wavelength at which to start the suppression in Angstroms.
    alpha : float
        The power of the suppression.
    sample_wavelengths : numpy.ndarray
        The wavelengths at which to sample the blackbody spectrum in Angstroms.
    redshift : float
        The redshift of the object.

    Returns
    -------
    seds : numpy.ndarray
        The spectral energy distribution at each time in erg/s/Angstrom.
    """

    # Constants
    xc = X_CONST  # noqa: F841
    fc = FLUX_CONST  # noqa: F841
    cc = C_CONST  # noqa: F841
    ac = ANG_CGS
    cwave_ac = cutoff_wavelength * ac
    cwave_ac2 = cwave_ac * cwave_ac
    cwave_ac3 = cwave_ac2 * cwave_ac  # noqa: F841
    zp1 = 1.0 + redshift

    lt = len(times)
    seds = np.empty(lt, dtype=object)
    rp2 = np.array(rphot) ** 2
    tp = Tphot

    # Calculate the rest wavelengths
    rest_wavs = sample_wavelengths * ac / zp1

    # The power needs to add up to 5
    sup_power = alpha
    wavs_power = (5 - sup_power)  # noqa: F841

    for li, lum in enumerate(luminosities):
        # Apply absorption to SED only bluewards of cutoff wavelength
        ab = rest_wavs < cwave_ac  # noqa: F841
        tpi = tp[li]  # noqa: F841
        rp2i = rp2[li]  # noqa: F841

        sed = ne.evaluate(
            "where(ab, fc * (rp2i / cwave_ac ** sup_power/ "
            "rest_wavs ** wavs_power) / expm1(xc / rest_wavs / tpi), "
            "fc * (rp2i / rest_wavs ** 5) / "
            "expm1(xc / rest_wavs / tpi))"
            )

        sed[np.isnan(sed)] = 0.0
        seds[li] = sed

    bb_wavelengths = np.linspace(100, 100000, N_TERMS)

    norms = np.array([(R2 * STEF_CONST * T ** 4) /
                      np.trapz(mod_blackbody(bb_wavelengths, T, R2, cutoff_wavelength, alpha),
                      bb_wavelengths) for T, R2 in zip(tp, rp2)])

    # Apply renormalisation
    seds *= norms

    # Units of `seds` is ergs / s / Angstrom.
    return seds


def mm83(nh, waves):
    """
    This function calculates the extinction of some photometry using the
    extinction model of Morrison & McCammon 1983.

    Parameters
    ----------
    nh : float
        The column density of hydrogen in cm^-2.
    waves : numpy.ndarray
        The wavelengths at which to calculate the extinction in Angstroms.

    Returns
    -------
    extinctions : numpy.ndarray
        The extinction at each wavelength in magnitudes.
    """

    # Milky Way Extinction
    number_mm83 = np.array(
        [[0.03, 17.3, 608.1, -2150.0],
         [0.1, 34.6, 267.9, -476.1],
         [0.284, 78.1, 18.8, 4.3],
         [0.4, 71.4, 66.8, -51.4],
         [0.532, 95.5, 145.8, -61.1],
         [0.707, 308.9, -380.6, 294.0],
         [0.867, 120.6, 169.3, -47.7],
         [1.303, 141.3, 146.8, -31.5],
         [1.84, 202.7, 104.7, -17.0],
         [2.471, 342.7, 18.7, 0.0],
         [3.21, 352.2, 18.7, 0.0],
         [4.038, 433.9, -2.4, 0.75],
         [7.111, 629.0, 30.9, 0.0],
         [8.331, 701.2, 25.2, 0.0]
         ])
    min_xray = 0.03
    max_xray = 10.0
    almin = 1.0e-24 * (
               number_mm83[0, 1] + number_mm83[0, 2] * min_xray +
               number_mm83[0, 3] * min_xray ** 2) / min_xray ** 3
    almax = 1.0e-24 * (
               number_mm83[-1, 1] + number_mm83[-1, 2] * max_xray +
               number_mm83[-1, 3] * max_xray ** 2) / max_xray ** 3

    # X-ray extinction in the ISM from Morisson & McCammon 1983
    y = np.array([H_C_CGS / (x * ANG_CGS * KEV_CGS) for x in waves])
    i = np.array([np.searchsorted(number_mm83[:, 0], x) - 1 for x in y])
    al = [1.0e-24 * (number_mm83[x, 1] + number_mm83[x, 2] * y[j] +
                     number_mm83[x, 3] * y[j] ** 2) / y[j] ** 3
          for j, x in enumerate(i)]

    # For less than 0.03 keV assume cross-section scales as E^-3.
    # http://ned.ipac.caltech.edu/level5/Madau6/Madau1_2.html
    # See also Rumph, Boyer, & Vennes 1994.
    al = [al[j] if x < min_xray
          else almin * (min_xray / x) ** 3
          for j, x in enumerate(y)]
    al = [al[j] if x > max_xray
          else almax * (max_xray / x) ** 3
          for j, x in enumerate(y)]
    return nh * np.array(al)


def observations(seds, sample_wavelengths, ebv, nh_host, band_wavelengths, transmissions, redshift, rv_host=3.1):
    """
    This function calculates the observed magnitudes of a set of SEDs.

    Parameters
    ----------
    seds : numpy.ndarray
        The spectral energy distributions at each time in erg/s/Angstrom.
    sample_wavelengths : numpy.ndarray
        The wavelengths at which the SEDs are sampled in Angstroms.
    ebv : float
        The E(B-V) extinction.
    nh_host : float
        The column density of hydrogen in the host galaxy in cm^-2.
    band_wavelengths : numpy.ndarray
        The wavelengths of the filters in Angstroms.
    transmissions : numpy.ndarray
        The transmittance of the filters.
    redshift : float
        The redshift of the object.
    rv_host : float
        The R_V value of the host galaxy.

    Returns
    -------
    model_observations : numpy.ndarray
        The observed magnitudes at the time of each SED.
    """
    # Calculate Filter integral
    filter_integral = FLUX_STD * np.trapz(np.array(transmissions) / np.array(band_wavelengths) ** 2, band_wavelengths)

    # Calculate Extinction
    av_mw = MW_RV * ebv
    nh_mw = av_mw * 1.8e21

    # Calculate Extinction from Milky Way
    Lyman_break = np.where(sample_wavelengths < LYMAN)
    mw_extinct = odonnell94(sample_wavelengths, av_mw, MW_RV)
    mw_extinct[Lyman_break] = mm83(nh_mw, sample_wavelengths[Lyman_break])

    av_host = nh_host / 1.8e21
    zp1 = 1.0 + redshift

    # Calculate Extinction from Host
    band_rest_wavelength = sample_wavelengths / zp1
    extinct_cache = odonnell94(band_rest_wavelength, av_host, rv_host)
    extinct_cache[Lyman_break] = mm83(nh_host, band_rest_wavelength[Lyman_break])

    # Add host and MW contributions
    seds_out = [eapp(mw_extinct + extinct_cache, x, inplace=False) for x in seds]

    # Luminosity Distance
    # If redshift is 0, set luminosity distance to 10 pc = 1e-5 Mpc
    if redshift == 0:
        lumdist = 1e-5
    else:
        lumdist = (cosmo.luminosity_distance(redshift) / u.Mpc).value
    dist_const = FOUR_PI * (lumdist * MPC_CGS) ** 2
    ldist_const = np.log10(dist_const)

    # Calculate Fluxes
    yvals = np.interp(sample_wavelengths, band_wavelengths, transmissions) * seds_out / zp1
    eff_fluxes = np.trapz(yvals, sample_wavelengths) / filter_integral

    # Calcualte Magnitudes
    model_observations = np.full(len(eff_fluxes), np.inf)
    ef_mask = eff_fluxes != 0.0
    model_observations[ef_mask] = - MAG_FAC * (np.log10(eff_fluxes[ef_mask]) - ldist_const)

    return model_observations


def slsnni(times, Pspin, Bfield, Mns, thetaPB, texplosion, kappa, log_kappa_gamma, mejecta, fnickel, v_ejecta,
           temperature, cut_wave, alpha, redshift, log_nh_host, ebv, bands=None, cenwaves=None):
    """
    This function calculates the observed magnitudes at a set of filters given the physical
    input parameters of the MOSFiT slsnnni model.

    Parameters
    ----------
    times : numpy.ndarray
        The times at which to calculate the luminosity in days.
    Pspin : float
        The spin period of the magnetar in milliseconds.
    Bfield : float
        The magnetic field of the magnetar in units of 10^14 Gauss.
    Mns : float
        The mass of the neutron star in solar masses.
    thetaPB : float
        The angle between the magnetic and rotation axes in radians.
    texplosion : float
        The time of explosion in observer frame days.
    kappa : float
        The opacity of the ejecta in cm^2/g.
    log_kappa_gamma : float
        The log of the opacity of the gamma-rays in cm^2/g.
    mejecta : float
        The total ejecta mass in solar masses.
    fnickel : float
        The mass fraction of nickel in the ejecta.
    v_ejecta : float
        The ejecta velocity in km/s.
    temperature : float
        The temperature floor of the photosphere in Kelvin.
    cut_wave : float
        The wavelength at which to start the suppression in Angstroms.
    alpha : float
        The power of the suppression.
    redshift : float
        The redshift of the object.
    log_nh_host : float
        The log10 of the column density of hydrogen in the host galaxy in cm^-2
    ebv : float
        The E(B-V) extinction.
    bands : list
        The names of the filters in which to calculate the observed magnitudes.
    cenwaves : numpy.ndarray, optional
        The central wavelengths of the filters in Angstroms.

    Returns
    -------
    If cenwaves is provided, seds will be returned. Else, model_observations will.

    model_observations : dict
        A dictionary containing the observed magnitudes at each filter.
    seds : numpy.ndarray
        The spectral energy distributions at each time in erg/s/Angstrom.
    """

    # Calculate rest frame explosion time
    rest_t_explosion = texplosion / (1 + redshift)

    # Convert log_nh_host to nh_host
    # And log_kappa_gamma to kappa_gamma
    nh_host = 10 ** log_nh_host
    kappa_gamma = 10 ** log_kappa_gamma

    # Input luminosities
    input_luminosities = total_luminosity(times, fnickel, mejecta, Pspin, Bfield, Mns, thetaPB, rest_t_explosion)

    # Output luminosities
    luminosities = diffusion(times, input_luminosities, kappa, kappa_gamma, mejecta, v_ejecta, rest_t_explosion)

    # Photospheric radius and temperature
    rphot, Tphot = photosphere(times, luminosities, v_ejecta, temperature, rest_t_explosion)

    # Empty dictionary to store the observations
    model_observations = {}

    # If cenwaves was provided use those as the input wavelengths
    if cenwaves is not None:
        seds = blackbody_supressed(times, luminosities, rphot, Tphot, cut_wave, alpha, cenwaves, redshift)
        return seds
    else:
        # Calculate the magnitudes in each band
        for band in bands:
            # Import the band transmission file from ref_data/filters
            if band in band_map:
                band_name = band_map[band]
            else:
                band_name = band
            band_file = os.path.join(data_dir, 'filters', f'{band_name}.dat')
            wavelength, transmission = np.loadtxt(band_file, unpack=True)

            seds = blackbody_supressed(times, luminosities, rphot, Tphot, cut_wave, alpha, wavelength, redshift)
            model_observation = observations(seds, wavelength, ebv, nh_host, wavelength, transmission, redshift)

            model_observations[band] = model_observation

        return model_observations
