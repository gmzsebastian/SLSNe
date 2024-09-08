"""
This file contains general utilities to process the data from the slsne package.

Written by Sebastian Gomez, 2024.
"""

# Import necessary packages
import os
import numpy as np
from astropy import table
import re
import json
from matplotlib.pyplot import cm
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo

# Get directory with reference data
current_file_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_file_dir, 'ref_data')

# Color-blind friendly red and green colors
cb_g = [0.288921, 0.758394, 0.428426, 1.]
cb_r = [0.862745, 0.078431, 0.235294, 1.]

# Map of references
cite_map = {'CPCS': '2019CoSka..49..125Z',
            'Gaia': '2016pas..conf...65W',
            'MDS': '2020ApJ...905...94V',
            'ThisWork': '2024arXiv240707946G',
            'ZTF': '2019PASP..131a8002B'}


def get_data_table(data_dir=data_dir):
    """
    Read in the data table with the names of the supernovae,
    their redshifts, and other key parameters.

    Parameters
    ----------
    data_dir : str
        Name of directory that contains the data table.

    Returns
    -------
    data_table : astropy.table.table.Table
        Astropy table with the the SLSNe and all their parameters.
    """
    data_table = table.Table.read(os.path.join(data_dir, 'sne_data.txt'), format='ascii')
    return data_table


def get_use_names(data_dir=data_dir):
    """
    Read in the list of SLSNe that are to be used in the
    reference database.

    Parameters
    ----------
    data_dir : str
        Name of directory that contains the 'use_names.txt' file.

    Returns
    -------
    use_names : np.array
        Array with the names of the SLSNe to be used.
    """
    use_names = np.genfromtxt(os.path.join(data_dir, 'use_names.txt'), dtype='str')
    return use_names


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


def calc_flux_lum(phot, redshift, return_lambda=False):
    """
    Calculate F_lambda and L_lambda for a photometry file,
    given a redshift.

    Parameters
    ----------
    phot : astropy.table.Table
        Table with photometry data
    redshift : float
        Redshift of the object
    return_lambda : bool, default False
        If True, return the wavelength of the filter
        in angstroms.

    Returns
    -------
    F_lambda : array
        Array of flux values in erg/s/cm^2/A
    L_lambda : array
        Array of luminosity values in erg/s/A
    lambda_AA : array, optional
        Array of central wavelengths in angstroms.
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

    if return_lambda:
        lambda_AA = np.array(phot['cenwave']) * u.AA
        return F_lambda, L_lambda, lambda_AA
    else:
        return F_lambda, L_lambda


def create_json(object_name, output_dir, default_err=0.1):
    """
    Create the JSON file that MOSFiT needs to fit the photometry
    from a photometry file of a known SLSN in the reference database.

    Parameters
    ----------
    object_name : str
        Name of the SLSN.
    output_dir : str
        Name of the directory where the JSON file will be
    default_err : float, default 0.1
        Default error to use for the photometry when it
        is missing or equal to 0.

    Returns
    -------
    Nothing, it just saves the JSON file.
    """

    # Get photometry table
    phot = get_lc(object_name)

    # Rename columns into MOSFiT format
    use = phot['Ignore'] == 'False'
    time = phot[use]['MJD'].astype(float)
    magnitude = phot[use]['Mag'].astype(float)
    e_magnitude = phot[use]['MagErr'].astype(float)
    upperlimit = np.array([False] * len(time))
    upperlimit[phot[use]['UL'] == 'True'] = True
    band = np.array(phot[use]['Filter']).astype(str)
    telescope = np.array(phot[use]['Telescope']).astype(str)
    instrument = np.array(phot[use]['Instrument']).astype(str)
    system = np.array(phot[use]['System']).astype(str)
    u_time = ['MJD'] * len(time)
    source = ['1'] * len(time)

    # Replace value of e_magnitude when there is none
    e_magnitude[e_magnitude == 0.0] = default_err
    e_magnitude[np.isnan(e_magnitude)] = default_err

    # Create output table
    output = table.Table([time, magnitude, e_magnitude, upperlimit, band, telescope, instrument, system,
                          u_time, source],
                         names=['time', 'magnitude', 'e_magnitude', 'upperlimit', 'band', 'telescope',
                                'instrument', 'system', 'u_time', 'source'])

    # Convert to Pandas
    photometry = output.to_pandas().to_dict(orient='records')

    # Create data dictionary for MOSFiT data
    template = {
        object_name: {
            "name": object_name,
            "sources": [
                {
                    "name": "Gomez et al. 2024",
                    "alias": "1"
                }
            ],
            "alias": [
                {
                    "value": object_name,
                    "source": "1"
                }
            ],
            "photometry": []
        }
    }
    template[object_name]['photometry'] = photometry

    # Save output
    print(f'\nSaving JSON file for {object_name}...')
    file_path = os.path.join(output_dir, f'{object_name}.json')
    with open(file_path, 'w') as file:
        json.dump(template, file, indent=4)


def calc_percentile(samples):
    """
    This function calculates the 1-sigma uncertainties
    for the samples.

    Parameters
    ----------
    samples : np.ndarray
        The samples to calculate the uncertainties.

    Returns
    -------
    output : tuple
        The median, upper uncertainty, and lower uncertainty
        of the samples.
    """
    values = np.percentile(samples, [15.87, 50, 84.13])
    output = values[1], values[2] - values[1], values[1] - values[0]
    return output


def get_params(object_name=None, param_names=None, local_dir=None):
    """
    Get the parameters of a supernova from the reference database.

    Parameters
    ----------
    object_name : str, default None
        Name of the supernova to get the light curve from.
        If None, all parameters will be returned
    param_names : list, optional
        List of parameter names to get from the reference table.
    local_dir : bool, default False
        If True, the parameters will be read from the local
        directory.

    Returns
    -------
    params : astropy.table.Table
        Table with the parameters of the supernova.
    """

    # Make sure param_names is a list
    if type(param_names) is str:
        param_names = [param_names]

    # Import light curve data
    if object_name is None:
        params = table.Table.read(os.path.join(data_dir, 'all_parameters.txt'), format='ascii')

        # Get requested parameters
        if param_names is not None:
            format_names = [f"{param}_{suffix}" for param in param_names for suffix in ['med', 'up', 'lo']]
            params = params[format_names]

        return params
    else:
        if local_dir is None:
            params = table.Table.read(os.path.join(data_dir, 'supernovae',
                                                   object_name, f'{object_name}_params.txt'), format='ascii')
            # Read reference data from sne_data file
            data_table = get_data_table()
            ref_data = data_table[data_table['Name'] == object_name]

            # Append ref_data to the params table metadata
            # For all keys in ref_data
            for key in ref_data.keys():
                params.meta[key] = ref_data[key][0]

        else:
            params = table.Table.read(os.path.join(local_dir,
                                                   object_name, 'jupyter', 'output_parameters.txt'), format='ascii')
        return params


def create_parameters(output_dir=None, use_names=None, local_dir=None):
    """
    This function creates a table with the parameters of all
    the SLSNe in the reference database.

    Parameters
    ----------
    output_dir : str, default None
        Name of the directory where the output table will
        be saved. If None, the table will not be saved.
    use_names : np.array, default None
        List with the names of the SLSNe to be used.
        If None, all SLSNe will be used.
    local_dir : str, default None
        Name of the directory where the parameters are stored.
        If None, the parameters will be read from the reference
        database.

    Returns
    -------
    output : astropy.table.Table
        Table with the parameters of all the SLSNe.
    """

    # Get data table and supernovae names
    data_table = get_data_table()
    if use_names is None:
        use_names = get_use_names()

    # Get parameter names from a default object
    params = get_params(use_names[0], local_dir=local_dir)
    colnames = np.array(params.colnames)
    colnames[colnames == 'kenergy'] = 'log(kenergy)'
    colnames[colnames == 'TSD'] = 'log(TSD)'
    use_columns = ['redshift'] + list(colnames) + ['1frac', 'efficiency']
    colnames = ['name'] + [item for sublist in [[f'{i}_lo', f'{i}_med', f'{i}_up']
                                                for i in use_columns] for item in sublist]

    # Create output table
    data = np.zeros((len(use_names), len(colnames)))
    output = table.Table(data=data, names=colnames)

    # Make the first column the names for use_names
    output['name'] = use_names

    for i in range(len(use_names)):
        object_name = use_names[i]
        print(i + 1, '/', len(use_names), object_name)

        # Read parameter file
        params = get_params(object_name, local_dir=local_dir)

        # Find the corresponding object in data_table
        match = data_table['Name'] == object_name
        redshift = data_table['Redshift'][match][0]
        output[i]['redshift_med'] = redshift

        for column in params.colnames:
            # Calculate mean and 1-sigma errors
            if column == 'kenergy':
                med, up, lo = calc_percentile(np.log10(1.0e51 * params[column]))
                output[i]['log(kenergy)_lo'] = lo
                output[i]['log(kenergy)_med'] = med
                output[i]['log(kenergy)_up'] = up
            elif column == 'TSD':
                med, up, lo = calc_percentile(np.log10(params[column]))
                output[i]['log(TSD)_lo'] = lo
                output[i]['log(TSD)_med'] = med
                output[i]['log(TSD)_up'] = up
            else:
                med, up, lo = calc_percentile(params[column])
                output[i][f'{column}_lo'] = lo
                output[i][f'{column}_med'] = med
                output[i][f'{column}_up'] = up

        # Calculate 1 - frac
        frac1 = 1 - params['frac']
        med, up, lo = calc_percentile(frac1)
        output[i]['1frac_lo'] = lo
        output[i]['1frac_med'] = med
        output[i]['1frac_up'] = up

        # Calculate efficiency
        E_rad = 10 ** params['log(E_rad)']
        E_kin = 1.0e51 * params['kenergy']
        efficiency = E_rad / E_kin
        med, up, lo = calc_percentile(efficiency)
        output[i]['efficiency_lo'] = lo
        output[i]['efficiency_med'] = med
        output[i]['efficiency_up'] = up

    # Return output
    if output_dir is not None:
        # Round off all columns to 5 decimal places, except the first column
        for column in output.colnames[1:]:
            output[column] = np.round(output[column], 5)
        output.write(os.path.join(output_dir, 'all_parameters.txt'), format='ascii.fixed_width',
                     delimiter=None, overwrite=True)
    else:
        return output


def get_references(object_names=None):
    """
    Print out the ADS bibcodes for the references used for either all
    the SLSNe in the reference database or a specific list of SLSNe.

    Parameters
    ----------
    object_names : np.array, default None
        List with the names of the SLSNe to get the references from.
        If None, all SLSNe will be used.

    Returns
    -------
    Nothing, it just prints out the references.
    """

    # Open Bibtex file
    bibtex_file_path = os.path.join(data_dir, 'references.bib')
    with open(bibtex_file_path, 'r') as file:
        bibtex_content = file.read()

    if object_names is None:
        object_names = get_use_names()

    # Make sure object_names is a list
    if type(object_names) is str:
        object_names = [object_names]

    # Get all the bibcodes for all objects in object_names
    bibcodes = np.array([])
    for object_name in object_names:
        source = get_lc(object_name)['Source']
        bibcodes = np.append(bibcodes, source)

    # Replace values in cite_map with their bibcodes
    for key in cite_map.keys():
        bibcodes[bibcodes == key] = cite_map[key]

    # Append Gomez et al. 2024 paper
    bibcodes = np.append(bibcodes, '2024arXiv240707946G')

    # Unique bibcodes
    bibcodes = np.unique(bibcodes)
    used_codes = []

    # Print out the references
    for bibcode in bibcodes:
        # Regex to find the entry with the given bibcode
        pattern = re.compile(
            r'@(ARTICLE|INPROCEEDINGS|PHDTHESIS){' + re.escape(bibcode) + r',.*?^}',
            re.DOTALL | re.MULTILINE
        )
        match = pattern.search(bibtex_content)

        if match:
            print(match.group(0), '\n')
            used_codes.append(bibcode)

    # Print out the bibcodes that were used
    print('Bibcodes used in this work:')
    print("\\citep{%s}" % (', '.join(np.sort(used_codes))))
