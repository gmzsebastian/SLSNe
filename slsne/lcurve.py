"""
This file contains general utilities to plot light curves of SLSNe, their photometry,
or models.

Written by Sebastian Gomez, 2024.
"""

# Import necessary packages
import os
import numpy as np
from astropy import table
from scipy import interpolate
import glob
from scipy.optimize import minimize
from .utils import quick_cenwave_zeropoint, calc_DM, get_cenwave

# Get directory with reference data
current_file_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_file_dir, 'ref_data')

# Import reference data and supernovae names
data_table = table.Table.read(os.path.join(data_dir, 'sne_data.txt'), format='ascii')
use_names = np.genfromtxt(os.path.join(data_dir, 'use_names.txt'), dtype='str')

# Only use names that are in the reference data directory
directories = glob.glob(os.path.join(data_dir, 'supernovae', '*'))
exists = [i.split('/')[-1] for i in directories]
# Only select the items in data_table that are in use_names and in good
data_table = data_table[np.isin(data_table['Name'], use_names) & np.isin(data_table['Name'], exists)]

# Optimal parameters for the bolometric scaling
OPTIMAL_PARAMS = np.array([5.82642369e+02, -6.84939538e+02,  2.95856107e+02, -5.59918505e+01,
                           3.91355582e+00,  2.16517179e+01, -2.52987467e+01,  1.09807710e+01,
                           -2.10200072e+00,  1.49953925e-01, -7.39934841e-02,  8.37511551e-02,
                           -3.55398351e-02,  6.69229149e-03, -4.71480374e-04, -8.87603076e-04,
                           1.04363068e-03, -4.55003242e-04,  8.74187767e-05, -6.25775202e-06,
                           3.47732275e-06, -4.05653087e-06,  1.75915344e-06, -3.36719198e-07,
                           2.40370778e-08])


def interpolate_1D(time, flux, samples, left=np.nan, right=np.nan):
    """
    Interpolates the flux values at arbitrary points using linear interpolation,
    after sorting them in order of time.

    Parameters
    ----------
    time : array
        The array of time values.
    flux : array
        The array of flux values corresponding to the time values.
    samples : array
        The array of arbitrary points where the flux values need to be interpolated.
    left : float, default np.nan
        The value to return for samples outside the time range.
    right : float, default np.nan
        The value to return for samples outside the time range.

    Returns
    -------
    interpolated_flux : array
        The interpolated flux values at the arbitrary points.

    """
    # Sort the input time and flux arrays by time
    sorted_indices = np.argsort(time)
    sorted_time = time[sorted_indices]
    sorted_flux = flux[sorted_indices]

    # Interpolate the flux to the arbitrary points
    interpolated_flux = np.interp(samples, sorted_time, sorted_flux, left=left, right=right)

    return interpolated_flux


def interpolate_2D(time, wavelength, flux, out_wave=None, out_phase=None):
    """
    Interpolate a 2D array that is a function of time and wavelength
    to either a new wavelength or a new time.

    Parameters
    ----------
    time : array
        The array of time values.
    wavelength : array
        The array of wavelength values.
    flux : array
        The 2D array of flux values corresponding to the
        time and wavelength values.
    out_wave : array, default None
        The array of new wavelength values.
    out_phase : array, default None
        The array of new time values.

    Returns
    -------
    new_flux : array
        The interpolated array of flux values at the new
        wavelength or time values.
    """

    # Make sure at least one output parameter is specified
    if (out_wave is None) and (out_phase is None):
        raise ValueError('At least one of out_wave or out_phase must be specified.')

    # Check that the shapes of the array are compatible
    if flux.shape != (len(time), len(wavelength)):
        raise ValueError('The shape of flux must be (len(time), len(wavelength)).')

    # For output parameters that are not specified, use the input parameters
    if out_wave is None:
        out_wave = wavelength
    if out_phase is None:
        out_phase = time

    # Create interinterpolator object
    f = interpolate.RectBivariateSpline(wavelength, time, flux.T)
    new_flux = np.array([f(wave, phase)[0][0] for wave, phase in zip(out_wave, out_phase)])

    # Return the interpolated flux
    return new_flux


def get_all_lcs(band, names=None, data_table=data_table, include_bronze=False, shift_to_peak=True,
                min_time=-70, max_time=300, time_samples=300, sigma=1, return_individual=True):
    """
    Get the light curves of supernovae from the reference data directory.

    Parameters
    ----------
    band : str
        Name of the filter to get the light curves from.
    names : list, default None
        List of names of the supernovae to get the light curves from.
        If None, all the supernovae in the reference data directory will be used.
    data_table : astropy.table.table.Table
        Table with the supernova data.
    incldue_bronze : bool, default False
        If True, include the bronze quality supernovae.
    shift_to_peak : bool, default True
        If True, shift the light curves to the peak of the supernova.
    min_time : float, default -70
        Minimum time to interpolate the light curves to.
    max_time : float, default 300
        Maximum time to interpolate the light curves to.
    time_samples : int, default 300
        Number of samples to interpolate the light curves to.
    sigma : float, default 1
        Number of standard deviations to return for the light curve range.
        Can only be 1, 2, or 3.
    return_individual : bool, default True
        Return individual light curves if True, otherwise just return
        the mean light curve.

    Returns
    -------
    If names is just one name, then the function will only return mag_mean,
    otherwise it will return mag_lo, mag_mean, and mag_hi.
    mag_lo : array
        Lower magnitude limit of the light curves.
    mag_mean : array
        Mean magnitude of the light curves.
    mag_hi : array
        Upper magnitude limit of the light curves.
    mean_array : ndarray
        Array of all light curves that went into making
        the mean and sigma values.
    """

    # Select either all names, or only the requested SNe names
    if names is None:
        use_names = list(data_table['Name'])
    else:
        use_names = names

    # Make sure use_names is a list
    if isinstance(use_names, str):
        use_names = [use_names]

    # Remove the bronze SNe if include_bronze is False
    if not include_bronze:
        use_names = [i for i in use_names if data_table['Quality'][data_table['Name'] == i][0] != 'Bronze']

    # Create array for interpolating the light curves
    samples = np.linspace(min_time, max_time, time_samples)

    print(f'\nGetting light curves of {len(use_names)} supernovae in {band} filter...')
    # Get the rest-frame light curve of each supernova
    for i in range(len(use_names)):
        object_name = use_names[i]
        print(i + 1, '/', len(use_names), object_name)

        # Get matching object data
        match = data_table['Name'] == object_name
        redshift = data_table['Redshift'][match][0]
        peak = data_table['Peak'][match][0]
        boom = data_table['Explosion'][match][0]

        # Import rest-frame data
        rest = table.Table.read(os.path.join(data_dir, 'supernovae', object_name,
                                             f'{object_name}_rest.txt'), format='ascii')

        # Get requested filter from data
        rest = rest[rest['Filter'] == band]

        if len(rest) == 0:
            print(f'No data for {object_name} in {band} filter.')
            continue

        # Calculate the rest-frame phase
        rest['Phase'] = (rest['MJD'] - peak) / (1 + redshift)
        # Shift the light curve to have the peak at phase 0 in the requested band
        if shift_to_peak:
            offset = rest['Phase'][rest['Mean'].argmin()]
            rest['Phase'] = rest['Phase'] - offset

        # Interpolate the light curve to the requested time samples
        mean = interpolate_1D(rest['Phase'], rest['Mean'], samples)

        # Append the interpolated light curve to the array of light curves.
        if ('mean_array' not in locals()) or (i == 0):
            mean_array = mean
        else:
            mean_array = np.vstack([mean_array, mean])

    # If mean_array does not exist, no data was found in that filter.
    if 'mean_array' not in locals():
        raise ValueError(f'No data found in {band} filter for the requested supernovae.')

    # Calculate the mean and standard deviation of the light curves
    # For the mean, 1, 2, and 3 sigma values
    if mean_array.shape[0] == time_samples:
        return (samples, mean)
    else:
        mag_lo3, mag_lo2, mag_lo1, mag_mean, mag_hi1, mag_hi2, mag_hi3 = \
            np.nanpercentile(mean_array, [0.13, 2.28, 15.87, 50, 84.13, 97.72, 99.87], axis=0)

    # Return the requested light curves and sigma scatter
    if sigma == 1:
        sigmas = (mag_lo1, mag_mean, mag_hi1)
    elif sigma == 2:
        sigmas = (mag_lo2, mag_mean, mag_hi2)
    elif sigma == 3:
        sigmas = (mag_lo3, mag_mean, mag_hi3)
    else:
        raise ValueError(f'sigma {sigma} must be 1, 2, or 3.')

    if return_individual:
        return sigmas, (samples, mean_array)
    else:
        return sigmas


def get_kcorr(phot, redshift, peak=None, boom=None, output_band=None, remove_ignore=True, stretch=1, offset=0):
    """
    Get the K-correction of the photometry.

    Parameters
    ----------
    phot : astropy.table.table.Table
        Table with the photometry.
    redshift : float
        Redshift of the supernova.
    peak : float, default None
        Peak date of the supernova in units of MJD.
    boom : float, default None
        Explosion date of the supernova in units of MJD.
    output_band : str, default None
        If None, the data will be corrected to the rest-frame
        version of the input data. If a string, the data will
        be corrected to the central wavelength of that band.
    remove_ignore : bool, default True
        Remove photometry that should be ignored.
    stretch : float, default 1
        Stretch factor for the light curve duration.
    offset : float, default 0
        Offset in phase for the light curve.
    """

    # Make sure required keys are in the photometry table
    required_keys = ['MJD', 'Telescope', 'Instrument', 'System', 'Filter']
    missing_keys = [key for key in required_keys if key not in phot.keys()]
    if missing_keys:
        raise ValueError(f'{", ".join(missing_keys)} key(s) not found in photometry table.')

    # Select only the useful photometry
    use = np.array([True] * len(phot))
    # Remove photometry that should be ignored, or that are upper limits
    if ('Ignore' in phot.keys()) and remove_ignore:
        use[phot['Ignore'] == 'True'] = False
    # Always remove upper limits
    if 'UL' in phot.keys():
        use[phot['UL'] == 'True'] = False

    # Make sure the photometry table is not empty
    if len(phot) == 0:
        raise ValueError(f'No usable photometry found in {output_band} filter.')

    # Get filter wavelengths, zeropoints, and phase
    phot['cenwave'], phot['zeropoint'] = quick_cenwave_zeropoint(phot)
    if peak is not None:
        phase0 = peak
        map_dir = os.path.join(data_dir, 'peak_mesh.npz')
    elif boom is not None:
        phase0 = boom
        map_dir = os.path.join(data_dir, 'boom_mesh.npz')
    else:
        raise ValueError('Either peak or boom must be specified.')

    # Calculate the phase of the photometry
    phot['phase'] = (phot['MJD'] - phase0) / (1 + redshift)

    # Read in magnitude map
    mag_map = np.load(map_dir)
    map_phase = (mag_map.get('phase') - offset) / stretch
    map_wavelength = mag_map.get('wavelength')
    map_magnitude = mag_map.get('magnitude')

    # Get input filter wavelength
    obswave = np.array(phot['cenwave'][use])

    if output_band is not None:
        # Make sure output_band is a string
        if not isinstance(output_band, str):
            raise ValueError(f'output_band {output_band} must be a string.')
        # Get the rest-frame wavelength of the output band
        restwave = np.array([get_cenwave(output_band, verbose=False)] * len(obswave)) / (1 + redshift)
    else:
        # Get output filter wavelength
        restwave = obswave / (1 + redshift)

    # Interpolate the magnitude map to the observed and rest wavelengths
    mean_rest_mag = interpolate_2D(map_phase.T[0], map_wavelength[0], map_magnitude, out_wave=restwave)
    mean_obs_mag = interpolate_2D(map_phase.T[0], map_wavelength[0], map_magnitude, out_wave=obswave)

    # Calculate K-correction
    K_corr = np.nan * np.ones(len(phot))
    K_corr[use] = mean_rest_mag - mean_obs_mag - 2.5 * np.log10(1 + redshift)

    # Return the K-correction
    return K_corr


def map_model(stretch, amplitude, offset, obs_wave, obs_phase, map_wavelength, map_phase, map_magnitude):
    """
    Interpolates the given map data to generate a model light curve, with some stretch,
    amplitude in magnitude, and offset in phase.

    Parameters
    ----------
    stretch : float
        The stretch factor for the phase.
    amplitude : float
        The amplitude to be added to the magnitude.
    offset : float
        The offset to be subtracted from the phase.
    obs_wave : array
        Array of observed wavelengths.
    obs_phase : array
        Array of observed phases.
    map_wavelength : array
        Array of map wavelengths.
    map_phase : array
        Array of map phases.
    map_magnitude : array
        Array of map magnitudes.

    Returns
    -------
    f : array-like
        Model light curve generated by interpolating the map data.

    """
    f = interpolate.RectBivariateSpline(map_wavelength, (map_phase - offset) / stretch, map_magnitude + amplitude)
    return np.array([f.ev(w, p) for w, p in zip(obs_wave, obs_phase)])


def objective(params, obs_wave, obs_phase, obs_mag, map_wavelength, map_phase, map_magnitude):
    """
    Objective function to minimize the difference between the observed and model light curves.

    Parameters
    ----------
    params : array
        Array of parameters to be optimized.
    obs_wave : array
        Array of observed wavelengths.
    obs_phase : array
        Array of observed phases.
    obs_mag : array
        Array of observed magnitudes.

    Returns
    -------
    float
        Sum of the squared differences between the observed and model light curves.
    """
    stretch, amplitude, offset = params
    out_mag = map_model(stretch, amplitude, offset, obs_wave, obs_phase, map_wavelength, map_phase, map_magnitude)
    return np.sum((out_mag - obs_mag)**2)


def fit_map(phot, redshift, peak=None, boom=None, remove_ignore=True):
    """
    Fit the chosen magnitude map (Either peak or boom) to a photometry
    table phot.

    Parameters
    ----------
    phot : astropy.table.table.Table
        Table with the photometry.
    redshift : float
        Redshift of the supernova.
    peak : float, default None
        Peak date of the supernova in units of MJD.
    boom : float, default None
        Explosion date of the supernova in units of MJD.
    remove_ignore : bool, default True
        Remove photometry that should be ignored.

    Returns
    -------
    stretch : float
        The stretch factor for the phase.
    amplitude : float
        The amplitude to be added to the magnitude.
    offset : float
        The offset to be subtracted from the phase.
    """

    # Make sure required keys are in the photometry table
    required_keys = ['MJD', 'Mag', 'Telescope', 'Instrument', 'System', 'Filter']
    missing_keys = [key for key in required_keys if key not in phot.keys()]
    if missing_keys:
        raise ValueError(f'{", ".join(missing_keys)} key(s) not found in photometry table.')

    # Remove ignored photometry
    if ('Ignore' in phot.keys()) and remove_ignore:
        phot = phot[phot['Ignore'] == 'False']

    # Make sure the photometry table is not empty
    if len(phot) == 0:
        raise ValueError('No usable photometry found in phot.')

    # Get filter wavelengths, zeropoints
    phot['cenwave'], phot['zeropoint'] = quick_cenwave_zeropoint(phot)
    if peak is not None:
        phase0 = peak
        map_dir = os.path.join(data_dir, 'peak_mesh.npz')
    elif boom is not None:
        phase0 = boom
        map_dir = os.path.join(data_dir, 'boom_mesh.npz')
    else:
        raise ValueError('Either peak or boom must be specified.')
    phot['restwave'] = phot['cenwave'] / (1 + redshift)

    # Read in magnitude map
    mag_map = np.load(map_dir)
    map_phase = mag_map.get('phase').T[0]
    map_wavelength = mag_map.get('wavelength')[0]
    map_magnitude = mag_map.get('magnitude').T

    # Calculate the phase of the photometry
    phot['phase'] = (phot['MJD'] - phase0) / (1 + redshift)

    # Get the distance modulus
    DM = calc_DM(redshift)

    # Calculate absolute magnitudes
    phot['abs_mag'] = phot['Mag'] - DM + 2.5 * np.log10(1 + redshift)

    # Get data to fit
    obs_wave = np.array(phot['restwave'])
    obs_phase = np.array(phot['phase'])
    obs_mag = np.array(phot['abs_mag'])

    # Fit the data
    initial_guess = [1.0, 0.0, 0.0]
    result = minimize(objective, initial_guess, args=(obs_wave, obs_phase, obs_mag,
                                                      map_wavelength, map_phase, map_magnitude),
                      bounds=[(0.3, 2), (-5, 5), (-8, 8)])
    stretch, amplitude, offset = result.x

    return stretch, amplitude, offset


def bol_model(phase, wavelength, degree_phase=4, degree_wavelength=4, params=OPTIMAL_PARAMS):
    """
    Calculate the bolometric scaling of a light curve based on the optimally
    determined parameters of a polynomial fit.

    Parameters
    ----------
    phase : array
        Phase of the light curve.
    wavelength : array
        Wavelength of the light curve.
    params : array
        Array of parameters for the polynomial fit, must have shape
        ((degree_phase), (degree_wavelength)).

    Returns
    -------
    scaling : arary
        The scaling of the light curve based on the polynomial fit.
    """
    terms = []
    for i in range(degree_phase + 1):
        for j in range(degree_wavelength + 1):
            terms.append((phase ** i) * (wavelength ** j))
    return np.dot(params, terms)


def get_bolcorr(phot, redshift, peak, remove_ignore=True):
    """
    Get the bolometric correction of the photometry.

    Parameters
    ----------
    phot : astropy.table.table.Table
        Table with the photometry.
    redshift : float
        Redshift of the supernova.
    peak : float
        Peak date of the supernova in units of MJD.
    remove_ignore : bool, default True
        Remove photometry that should be ignored.

    Returns
    -------
    bol_scaling : array
        The bolometric scaling factor of the photometry.
    """

    # Make sure required keys are in the photometry table
    required_keys = ['MJD', 'Telescope', 'Instrument', 'System', 'Filter']
    missing_keys = [key for key in required_keys if key not in phot.keys()]
    if missing_keys:
        raise ValueError(f'{", ".join(missing_keys)} key(s) not found in photometry table.')

    # Remove photometry that should be ignored, or that are upper limits
    use = np.array([True] * len(phot))
    if ('Ignore' in phot.keys()) and remove_ignore:
        use[phot['Ignore'] == 'True'] = False
    # Always remove upper limits
    if 'UL' in phot.keys():
        use[phot['UL'] == 'True'] = False

    # Make sure the photometry table is not empty
    if len(phot) == 0:
        raise ValueError('No usable photometry found in phot table.')

    # Get filter wavelengths, zeropoints, and phase
    phot['cenwave'], phot['zeropoint'] = quick_cenwave_zeropoint(phot)
    phase0 = peak
    phot['phase'] = (phot['MJD'] - phase0) / (1 + redshift)

    # Calculate the bolometric scaling
    use_phase = np.array(phot['Phase'])
    use_wave = np.log10(phot['cenwave'] / (1 + redshift))
    bol_scaling = 10 ** bol_model(use_phase, use_wave)

    # Return the bolometric scaling
    return bol_scaling


def get_lc(object_name, lc_type='phot'):
    """
    Get the light curve of a supernova from the reference data directory.

    Parameters
    ----------
    object_name : str
        Name of the supernova to get the light curve from.
    lc_type : str, default 'phot'
        Type of light curve to get. Can be 'phot', 'model',
        'bol' or 'rest'.
        'phot' - Observed photometry of the SN.
        'model' - MOSFiT light curve model of the photometry.
        'bol' - Bolometric parameters of the SN.
        'rest' - Rest-frame MOSFiT light curve model.
    Returns
    -------
    lc : astropy.table.table.Table
        Light curve of the supernova.
    """

    # Import light curve data
    if lc_type == 'phot':
        lc = table.Table.read(os.path.join(data_dir, 'supernovae',
                                           object_name, f'{object_name}.txt'), format='ascii')
    elif lc_type == 'model':
        lc = table.Table.read(os.path.join(data_dir, 'supernovae',
                                           object_name, f'{object_name}_model.txt'), format='ascii')
    elif lc_type == 'rest':
        lc = table.Table.read(os.path.join(data_dir, 'supernovae',
                                           object_name, f'{object_name}_rest.txt'), format='ascii')
    elif lc_type == 'bol':
        lc = table.Table.read(os.path.join(data_dir, 'supernovae',
                                           object_name, f'{object_name}_bol.txt'), format='ascii')
    else:
        raise ValueError(f'lc_type {lc_type} must be phot, model, bol, or rest.')

    return lc
