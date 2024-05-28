"""
This file contains general utilities to plot light curves of SLSNe, their photometry,
or models.

Written by Sebastian Gomez, 2024.
"""

# Import necessary packages
import os
import numpy as np
from astropy import table
import glob

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


def interpolate(time, flux, samples, left=np.nan, right=np.nan):
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
        mean = interpolate(rest['Phase'], rest['Mean'], samples)

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
