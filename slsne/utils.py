"""
This file contains general utilities to process the data from the slsne package.

Written by Sebastian Gomez, 2024.
"""

# Import necessary packages
import os
import numpy as np
from astropy import table
import re
from matplotlib.pyplot import cm
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo

# Get directory with reference data
current_file_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_file_dir, 'ref_data')

# Color-blind friendly red and green colors
cb_g = [0.288921, 0.758394, 0.428426, 1.]
cb_r = [0.862745, 0.078431, 0.235294, 1.]


def define_filters(data_dir=data_dir):
    """
    Create an Astropy table with a list of filters, their central
    wavelengths in angstroms, and their zeropoint in Jy, accounting
    for the different values for different telescopes in the database.
    Telescopes with known central wavelengths and zeropoints will be read
    out of 'filter_reference.txt', and anything that is not there will
    be assigned a generic filter from 'generic_reference.txt'

    Parameters
    ----------
    data_dir : str
        Name of directory that contains the 'filter_reference.txt' and
        'generic_reference.txt' files.

    Returns
    -------
    filters : astropy.table.table.Table
        Astropy table with filters, their telescope, instrument, system (AB or Vega),
        their central wavelength, and zeropoint
    """

    # Import filter parameters
    generics = table.Table.read(os.path.join(data_dir, 'generic_reference.txt'), format='ascii')
    filters_in = table.Table.read(os.path.join(data_dir, 'filter_reference.txt'), format='ascii')

    # Assign generic values to filters
    filters_in['Cenwave'][filters_in['Cenwave'] == 'Generic'] = \
                         [generics['Cenwave'][(generics['Filter'] == i['Filter']) &
                          (generics['System'] == i['System'])][0]
                          for i in filters_in[filters_in['Cenwave'] == 'Generic']]
    filters_in['Zeropoint'][filters_in['Zeropoint'] == 'Generic'] = \
                           [generics['Zeropoint'][(generics['Filter'] == i['Filter']) &
                            (generics['System'] == i['System'])][0]
                            for i in filters_in[filters_in['Zeropoint'] == 'Generic']]

    # Create a generic column
    generic_col = table.Column(['Generic'] * len(generics))
    generics.add_columns((generic_col, generic_col), names=('Telescope', 'Instrument'))

    # Reorder columns
    columns = filters_in.colnames
    generics = generics[columns]

    # Make sure all types are floats
    filters_in['Cenwave'] = filters_in['Cenwave'].astype(float)
    filters_in['Zeropoint'] = filters_in['Zeropoint'].astype(float)

    # Append generics to filters
    filters = table.vstack([filters_in, generics])

    return filters


# Import filter parameters
filters = define_filters()


def get_cenwave(filter_name, telescope=None, instrument=None, system=None, filters=filters,
                verbose=True, return_zp=False):
    """
    Get the central wavelength in angstroms for a filter from
    an optional telescope and instrument. Optionally also
    return the zeropoint of that filter in Janskys.

    Parameters
    ----------
    filter_name : str
        Name of filter
    telescope : str, default None
        Optional name of telescope
    instrument : str, default None
        Optional name of instrument
    system : str, default 'AB'
        System of filter, either AB or Vega
        Required if return_zp=True
    filters : astropy.Table
        Table containing filter information
        generated with define_filters()
    verbose : bool, default True
        Option to print statements about
        filter choices.
    return_zp : bool, default False
        Optionally return the zeropoint

    Returns
    -------
    cenwave : float
        Value of central wavelength of filter in angstroms
    zeropoint : float
        Value of zeropoint of filter in Jy
    """

    if verbose:
        print('\n')
    if return_zp:
        if system not in ['Vega', 'AB']:
            raise KeyError(f'system "{system}" not known, it must be "AB" or "Vega".')
        elif system == 'AB':
            zeropoint = 3631.0
        elif system == 'Vega':
            pass
    else:
        # Set default system to AB if no zeropoint is requested, since the system
        # does not matter for the central wavelength.
        system = 'AB'
        zeropoint = 3631.0

    # Is filter a Swift filter?
    is_swift = any(i in filter_name.lower() for i in ['swift', 'uvot'])

    if is_swift:
        if verbose:
            print(f'Assuming "{filter_name}" is a Swift/UVOT filter.')
        # Use appropriate table and rename filter name
        filter_name = re.sub(r"swift|uvot|_|-|\.", "", filter_name, flags=re.IGNORECASE)
        use_filters = filters[(filters['Telescope'] == 'Swift') & (filters['System'] == system) &
                              (filters['Filter'] == filter_name)]
        instrument = 'UVOT'
        telescope = 'Swift'
        if len(use_filters) == 0:
            raise KeyError(f'filter_name "{filter_name}" with system "{system}" is not a known Swift filter.')
    else:
        if (telescope is not None) & (instrument is not None):
            use_filters = filters[(filters['Filter'] == filter_name) & (filters['System'] == system) &
                                  (filters['Telescope'] == telescope) & (filters['Instrument'] == instrument)]
            if len(use_filters) == 0:
                raise KeyError(f'filter_name "{filter_name}" with system "{system}",'
                               f' telescope "{telescope}", and instrument "{instrument}" is not known.')
        elif (telescope is not None):
            use_filters = filters[(filters['Filter'] == filter_name) & (filters['System'] == system) &
                                  (filters['Telescope'] == telescope)]
            new_telescopes = ', '.join(list(np.unique(filters[(filters['Filter'] == filter_name) &
                                                      (filters['System'] == system)]['Telescope'])))
            if len(use_filters) == 0:
                raise KeyError(f'filter_name "{filter_name}" with system "{system}"'
                               f' telescope "{telescope}" is not known, try from {new_telescopes}')
        elif (instrument is not None):
            use_filters = filters[(filters['Filter'] == filter_name) & (filters['System'] == system) &
                                  (filters['Instrument'] == instrument)]
            new_instruments = ', '.join(list(np.unique(filters[(filters['Filter'] == filter_name) &
                                                       (filters['System'] == system)]['Instrument'])))
            if len(use_filters) == 0:
                raise KeyError(f'filter_name "{filter_name}" with system "{system}"'
                               f' instrument "{instrument}" is not known, try from {new_instruments}')
        else:
            use_filters = filters[(filters['Filter'] == filter_name) & (filters['System'] == system)]
            if len(use_filters) == 0:
                raise KeyError(f'filter_name "{filter_name}" with system "{system}" is not known.')

    # If only one filter found, use that one
    if len(use_filters) == 1:
        cenwave = use_filters['Cenwave'][0]
        new_telescope = use_filters['Telescope'][0]
        new_instrument = use_filters['Instrument'][0]
        if return_zp:
            zeropoint = use_filters['Zeropoint'][0]
            zp_message = f'and zeropoint of {zeropoint} '
        else:
            zp_message = ''
        if verbose:
            print(f'Central wavelength of {cenwave} {zp_message}found for filter "{filter_name}"'
                  f' with system "{system}", telescope "{new_telescope}", and instrument "{new_instrument}".')
    else:
        new_telescopes = list(table.unique(use_filters['Telescope', 'Instrument'])['Telescope'])
        new_instruments = list(table.unique(use_filters['Telescope', 'Instrument'])['Instrument'])
        if verbose:
            print(f'Warning: Multiple instances found for filter "{filter_name}" with system '
                  f'"{system}" from telescopes {new_telescopes} and instruments {new_instruments}.')
        # If multiple filters found, default to generic.
        if 'Generic' in use_filters['Telescope']:
            cenwave = use_filters['Cenwave'][use_filters['Telescope'] == 'Generic'][0]
            if return_zp:
                zeropoint = use_filters['Zeropoint'][use_filters['Telescope'] == 'Generic'][0]
                zp_message = f'and zeropoint of {zeropoint} '
            else:
                zp_message = ''
            if verbose:
                print(f'Central wavelength of {cenwave} {zp_message}selected for "{filter_name}" with system'
                      f' "{system}" and "Generic" instrument and telescope.')
        else:
            cenwave = use_filters['Cenwave'][0]
            new_telescope = use_filters['Telescope'][0]
            new_instrument = use_filters['Instrument'][0]
            new_system = use_filters['System'][0]
            if return_zp:
                zeropoint = use_filters['Zeropoint'][0]
                zp_message = f'and zeropoint of {zeropoint} '
            else:
                zp_message = ''
            if verbose:
                print(f'No "Generic" entry found for filter "{filter_name}" with system "{system}",'
                      f' instrument "{instrument}", and telescope "{telescope}"; defaulting to a central'
                      f' wavelength of {cenwave} {zp_message}with system "{new_system}" from telescope'
                      f' "{new_telescope}" and instrument "{new_instrument}"')

    if return_zp:
        return cenwave, zeropoint
    else:
        return cenwave


def quick_cenwave_zeropoint(phot, filters=filters):
    """
    Obtain the central wavelength in angstroms and zeropoints in Jansky
    for a set of photometry phot that has already been verified and
    properly processed, with only known filters. Unlike get_cenwave(),
    this function will only run minimal checks. If this fails, use
    check_filters() to diagnose.

    Parameters
    ----------
    phot : astropy.Table
        Table with phot
    filters : astropy.Table
        Table containing filter information
        generated with define_filters()

    Returns
    -------
    cenwaves : np.array
        Array of central wavelengths in angstroms
    zeropoints : np.array
        Array of zeropoints in Jy
    """

    # Verify that the columns exist
    for column in ['Telescope', 'Instrument', 'System', 'Filter']:
        if column not in phot.colnames:
            raise KeyError(f'Column {column} is a required column in phot Table')

    # Obtain the corresponding central wavelengths and zeropoints of all items in phot
    cenwaves, zeropoints = np.array([list(filters['Cenwave', 'Zeropoint'][((filters['Telescope'] == k['Telescope']) &
                                                                           (filters['Instrument'] == k['Instrument']) &
                                                                           (filters['System'] == k['System']) &
                                                                           (filters['Filter'] == k['Filter']))][0])
                                    for k in phot]).astype(float).T

    return cenwaves, zeropoints


def check_filters(phot, filters=filters):
    """
    If quick_cenwave_zeropoint() failed, that is likely because
    one of the filters in the phot table is not found in filters.
    This function will find the broken filters and point them out.

    Parameters
    ----------
    phot : astropy.Table
        Table with phot
    filters : astropy.Table
        Table containing filter information
        generated with define_filters()

    Returns
    -------
    Nothing, it just tells you what you did wrong.
    """

    # Verify that the columns exist
    for column in ['Telescope', 'Instrument', 'System', 'Filter']:
        if column not in phot.colnames:
            raise KeyError(f'Column {column} is a required column in phot Table')

    for k in phot:
        telescope, instrument, system, band = k['Telescope'], k['Instrument'], k['System'], k['Filter']
        use_filters = (filters['Telescope'] == telescope) & (filters['Instrument'] == instrument) & \
                      (filters['System'] == system) & (filters['Filter'] == band)
        if sum(use_filters) == 0:
            print(f'Combination of telescope "{telescope}", instrument "{instrument}",'
                  f' system "{system}", and filter "{band}" not found in filter database.')
        elif sum(use_filters) > 1:
            print(f'Combination of telescope "{telescope}", instrument "{instrument}",'
                  f' system "{system}", and filter "{band}" has more than one entry, {len(use_filters)}.')
        else:
            pass


def plot_colors(band):
    """
    Assign a matplotlib color to a given band, if the filter
    is not known, the color will be black.

    Parameters
    ----------
    band : str
        Name of filter.

    Returns
    -------
    color : str
        Matplotlib color for the given band.
    """

    # Assign matplotlib colors to Swift bands
    colors_UVOT = cm.rainbow(np.linspace(0, 1, 7))

    # Create a dictionary mapping bands to colors
    band_color_map = {
        "u": 'navy', "u'": 'navy', "U": 'navy',
        "g": 'g', "g'": 'g',
        'r': 'r', "r'": 'r', 'R': 'r', 'Rs': 'r',
        'i': 'maroon', "i'": 'maroon', 'I': 'maroon',
        "z": 'saddlebrown', "z'": 'saddlebrown',
        'V': 'lawngreen',
        'B': 'darkcyan',
        'C': 'c',
        'w': 'goldenrod',
        'G': 'orange',
        'W1': 'deeppink',
        'W2': 'tomato',
        'orange': 'gold',
        'cyan': 'blue',
        'Clear': 'magenta',
        'UVM2': colors_UVOT[0],
        'UVW1': colors_UVOT[1],
        'UVW2': colors_UVOT[2],
        'F475W': 'lightsteelblue',
        'F625W': 'slategray',
        'F775W': 'tan',
        'F850LP': 'gray',
        'H': 'hotpink',
        'J': 'mediumvioletred',
        'K': 'palevioletred', "Ks": 'palevioletred',
        'Y': 'indigo', "y": 'indigo',
        'v': 'aquamarine'
    }

    # Get the color for the given band, default to 'k' if not found
    return band_color_map.get(band, 'k')


def calc_DL(redshift):
    """
    Calculate the luminosity distance in pc for a given redshift.

    Parameters
    ----------
    redshift : float
        Redshift of the object

    Returns
    -------
    DL : float
        Luminosity distance in pc
    """
    DL = cosmo.luminosity_distance(z=redshift).to(u.pc).value
    return DL


def calc_DM(redshift):
    """
    Calculate the distance modulus for a given redshift.

    Parameters
    ----------
    redshift : float
        Redshift of the object

    Returns
    -------
    DM : float
        Distance modulus
    """
    DL = calc_DL(redshift)
    DM = 5 * np.log10(DL / 10)
    return DM


def read_phot(object_name):
    """
    Read in a photometry file for a given SLSN from the
    reference database.

    Parameters
    ----------
    object_name : str
        Name of the supernova.

    Returns
    -------
    phot : astropy.table.Table
        Table with photometry data
    """
    phot = table.Table.read(os.path.join(data_dir, 'supernovae', object_name, f'{object_name}.txt'), format='ascii')
    return phot


def read_bolo(object_name):
    """
    Get the bolometric parameters of a supernova.

    Parameters
    ----------
    object_name : str
        Name of the supernova.

    Returns
    -------
    bolo : astropy.table.Table
        Table with the bolometric luminosity,
        radius, and temperature.
    """
    bolo = table.Table.read(os.path.join(data_dir, 'supernovae', object_name, f'{object_name}_bol.txt'), format='ascii')
    return bolo


def calc_flux_lum(phot, redshift):
    """
    Calculate F_lambda and L_lambda for a photometry file,
    given a redshift.

    Parameters
    ----------
    phot : astropy.table.Table
        Table with photometry data
    redshift : float
        Redshift of the object

    Returns
    -------
    F_lambda : array
        Array of flux values in erg/s/cm^2/A
    L_lambda : array
        Array of luminosity values in erg/s/A
    """

    # If 'zeropoint' or 'cenwave' not in phot, calculate them using quick_cenwave_zeropoint
    if not all([i in phot.keys() for i in ['zeropoint', 'cenwave']]):
        cenwaves, zeropoints = quick_cenwave_zeropoint(phot)
        phot['zeropoint'] = zeropoints
        phot['cenwave'] = cenwaves

    # Make sure Mag is a key in phot
    if not all([i in phot.keys() for i in ['Mag']]):
        raise ValueError('Photometry table must have keys "Mag"')

    # Calculate the luminosity distance
    DL = calc_DL(redshift) * u.pc

    # Convert to F_nu using zeropoints in table
    F_nu_phot = 10 ** (-0.4 * phot['Mag']) * (phot['zeropoint'] * u.Jy)

    # Calculate flux
    F_lambda = F_nu_phot.to(u.erg / u.s / u.cm / u.cm / u.AA, equivalencies=u.spectral_density(phot['cenwave'] * u.AA))

    # Calculate luminosity
    L_lambda = F_lambda * 4 * np.pi * DL.to(u.cm) ** 2 * (1 + redshift)

    return F_lambda, L_lambda
