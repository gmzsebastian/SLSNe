"""
This file contains utilities to import, plot, and process the data from MOSFiT.

Written by Sebastian Gomez, 2024.
"""

# Import necessary packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import json
import os
from .utils import calc_percentile, quick_cenwave_zeropoint, plot_colors, get_data_table, get_cenwave, get_lc
from astropy import units as u
from .models import slsnni, nickelcobalt, magnetar
from astropy import table
import scipy.integrate as itg
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.family': 'serif'})


def import_mosfit(object_name, mosfit_dir, import_extras=False):
    """
    This function imports the data from MOSFiT and returns
    the necessary data.

    Parameters
    ----------
    object_name : str
        The name of the object to import.
    mosfit_dir : str
        The path to the file containing the data.
        It will assume there is a folder named object_name
        with a products directory with:
        object_name_chain.json
        object_name_walkers.json
        object_name_extras.json (optional)
    import_extras : bool, default False
        Whether to import the extras file.

    Returns
    -------
    all_chain : np.ndarray
        The chain of all walkers with shape (nwalkers, nsteps, nparameters).
    chain_names : np.ndarray
        The names of the parameter for each chain.
    data : dict
        The dictionary with all the MOSFiT data.
    extras : dict
        The dictionary with all the MOSFiT extra parameters.
    """
    # Get file paths
    chain_path = os.path.join(mosfit_dir, object_name, 'products', f'{object_name}_chain.json')
    walkers_path = os.path.join(mosfit_dir, object_name, 'products', f'{object_name}_walkers.json')
    extras_path = os.path.join(mosfit_dir, object_name, 'products', f'{object_name}_extras.json')

    # Make sure chain and walkers files exist
    if not os.path.exists(chain_path):
        raise FileNotFoundError(f'File {chain_path} not found.')
    if not os.path.exists(walkers_path):
        raise FileNotFoundError(f'File {walkers_path} not found.')

    # Import Chains
    print('Importing chain file...')
    with open(chain_path, 'r', encoding='utf-8') as f:
        chain_data = json.load(f)
        all_chain = np.array(chain_data[0])
        chain_names = np.array(chain_data[1])

    # Import Model Realizations
    print('Importing walkers file...')
    with open(walkers_path, 'r', encoding='utf-8') as f:
        data = json.loads(f.read())
        if 'name' not in data:
            data = data[list(data.keys())[0]]

    # Import Extras
    if import_extras:
        # Make sure extras file exists
        if not os.path.exists(extras_path):
            raise FileNotFoundError(f'File {extras_path} not found.')
        print('Importing extras file...')
        with open(extras_path, 'r', encoding='utf-8') as f:
            extras = json.loads(f.read())

        return all_chain, chain_names, data, extras
    else:
        return all_chain, chain_names, data


def plot_trace(param_chain, param_values, param_values_log, min_val, max_val,
               title_name, param, log, n_steps, burn_in, output_dir):
    '''
    This function plots the trace of a parameter chain.

    Parameters
    ----------
    param_chain : np.ndarray
        The chain of the parameter with shape (nwalkers, nsteps).
    param_values : np.ndarray
        The median, upper and lower limits of the parameter.
    param_values_log : np.ndarray
        The median, upper and lower limits of the log of the parameter.
    min_val : float
        The minimum value of the parameter.
    max_val : float
        The maximum value of the parameter.
    title_name : str
        The name of the parameter.
    param : str
        The name of the parameter.
    log : bool
        Whether the parameter is in log scale.
    n_steps : int
        The number of steps in the chain.
    burn_in : float
        The fraction of steps to burn in.
    output_dir : str
        The directory to save the plot.
    '''

    # Average walker position
    Averageline = np.average(param_chain.T, axis=1)

    # Plot Trace
    plt.subplots_adjust(wspace=0)
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax0 = plt.subplot(gs[0])
    ax0.axhline(param_values[0], color='r', lw=2.0, linestyle='--', alpha=0.75)
    ax0.axhline(param_values[0] - param_values[2], color='r', lw=1.0, linestyle='--', alpha=0.50)
    ax0.axhline(param_values[0] + param_values[1], color='r', lw=1.0, linestyle='--', alpha=0.50)
    ax0.plot(Averageline, lw=1.0, color='b', alpha=0.75)
    ax0.plot(param_chain.T, '-', color='k', alpha=0.2, lw=0.5)
    plt.xlim(0, n_steps - 1)
    if log:
        plt.ylim(np.log10(min_val), np.log10(max_val))
    else:
        plt.ylim(min_val, max_val)

    title_string = r"$%s^{+%s}_{-%s}$" % (np.round(param_values[0], 5), np.round(param_values[1], 5),
                                          np.round(param_values[2], 5))
    if log:
        title_string += '  = log(' + r"$%s^{+%s}_{-%s}$" % (np.round(param_values_log[0], 5),
                                                            np.round(param_values_log[1], 5),
                                                            np.round(param_values_log[2], 5)) + ')'
    plt.title(title_string)
    plt.ylabel(title_name)
    plt.xlabel("Step")

    # Plot Histogram
    ax1 = plt.subplot(gs[1])
    plt.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
                    labeltop=False, labelright=False, labelbottom=False)
    if log:
        plt.ylim(np.log10(min_val), np.log10(max_val))
    else:
        plt.ylim(min_val, max_val)
    ax1.hist(np.ndarray.flatten(param_chain[:, int(n_steps*burn_in):]), bins='auto',
             orientation="horizontal", color='k')
    ax1.axhline(Averageline[-1], color='b', lw=1.0, linestyle='-', alpha=0.75)
    ax1.axhline(param_values[0], color='r', lw=2.0, linestyle='--', alpha=0.75)
    ax1.axhline(param_values[0] - param_values[2], color='r', lw=1.0, linestyle='--', alpha=0.50)
    ax1.axhline(param_values[0] + param_values[1], color='r', lw=1.0, linestyle='--', alpha=0.50)

    output_path = os.path.join(output_dir, param + "_Trace.jpg")
    plt.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.clf()
    plt.close('all')


def plot_params(all_chain, chain_names, data, output_dir, plot_corner=True,
                exclude=['efficiency', 'lphoto', 'variance'], plot_derived=True,
                append_derived=False, burn_in=0.9, reference_band='r', is_slsn=True):
    '''
    This function plots the trace of all parameters in the chain, and
    optionally the corner plot. The all_chain, chain_names, and data
    parameters are obtained from the import_mosfit function.

    Parameters
    ----------
    all_chain : np.ndarray
        The chain of all walkers with shape (nwalkers, nsteps, nparameters).
    chain_names : np.ndarray
        The names of the parameter for each chain.
    data : dict
        The dictionary with all the MOSFiT data.
    output_dir : str
        The directory to save the plots.
    plot_corner : bool, default False
        Whether to plot the corner plot.
    exclude : list, default ['efficiency', 'lphoto', 'variance']
        The parameters to exclude from the plot.
    plot_derived : bool, default True
        Whether to also plot the derived parameters.
    append_derived : bool, default False
        Whether to append the derived parameters to the corner plot.
    burn_in : float, default 0.9
        The fraction of steps to burn in.
    reference_band : str, default 'r'
        The reference band to use for peak calculations.
    is_slsn : bool, default True
        Is the object being processed a SLSN?
    '''
    # Number of walkers
    n_steps = all_chain.shape[1]

    # Get MOSFiT parameters
    setup = data['models'][0]['setup']
    photo = data['photometry']
    model = data['models'][0]

    # Initialize corner chain
    if plot_corner:
        names = []
        truths = []

    # Empty arrays for saving parameters
    output_parameters = []
    output_names = []

    # Add derived parameters
    if plot_derived:
        if 'fnickel' in chain_names:
            derived = ['A_V', 'MJD0', 'kenergy', 'mnickel', 'TSD', 'L0']
        else:
            derived = ['A_V', 'MJD0', 'kenergy', 'TSD', 'L0']
        chain_names = np.append(chain_names, derived)

        # Get MJD of first datapoint
        MJD0 = data['models'][0]['realizations'][0]['parameters']['reference_texplosion']['value']

        setup['A_V'] = {'latex': 'A_V',
                        'log': False,
                        'min_value': 0.0,
                        'max_value': 2.0}

        setup['kenergy'] = {'latex': 'KE / 10^{51} erg\\ s^{-1}',
                            'log': False,
                            'min_value': 0.1,
                            'max_value': 40.0}

        setup['mnickel'] = {'latex': 'M_{Ni}',
                            'log': False,
                            'min_value': 1e-3,
                            'max_value': 50.0}

        setup['MJD0'] = {'latex': 'MJD_0',
                         'log': False,
                         'min_value': MJD0 - 100,
                         'max_value': MJD0 + 100}

        setup['L0'] = {'latex': 'L_0',
                       'log': True,
                       'min_value': 1.0e38,
                       'max_value': 1.0e50}

        setup['TSD'] = {'latex': 't_{\\rm SD}',
                        'log': True,
                        'min_value': 1,
                        'max_value': 1.0e10}

        nhhost_index = np.where(chain_names == 'nhhost')[0][0]
        if 'fnickel' in chain_names:
            fnickel_index = np.where(chain_names == 'fnickel')[0][0]
        else:
            fnickel_index = None
        mejecta_index = np.where(chain_names == 'mejecta')[0][0]
        vejecta_index = np.where(chain_names == 'vejecta')[0][0]
        texplosion_index = np.where(chain_names == 'texplosion')[0][0]
        pspin_index = np.where(chain_names == 'Pspin')[0][0]
        bfield_index = np.where(chain_names == 'Bfield')[0][0]
        Mns_index = np.where(chain_names == 'Mns')[0][0]
    else:
        derived = []

    for i in range(len(chain_names)):
        param = chain_names[i]
        if param not in exclude:
            print('Plotting ' + param)

            # Extract Name and Log
            if 'latex' in setup[param]:
                param_latex = setup[param]['latex']
            else:
                param_latex = param
            if 'log' in setup[param]:
                param_log = setup[param]['log']
            else:
                param_log = False

            # Minimum and Maximum values
            min_val = setup[param]['min_value']
            max_val = setup[param]['max_value']

            # Modify the chain if necessary
            if param in derived:
                if param == 'A_V':
                    param_chain = all_chain[:, :, nhhost_index] / 1.8e21
                elif param == 'kenergy':
                    mod_mass = all_chain[:, :, mejecta_index]
                    mod_velocity = all_chain[:, :, vejecta_index]
                    # Convert Units
                    M_SUN_to_GRAMS = u.Msun.to(u.g)
                    KM_to_CM = u.km.to(u.cm)
                    param_chain = (3/10) * (mod_mass * M_SUN_to_GRAMS) * (mod_velocity * KM_to_CM) ** 2 / 1E51
                elif param == 'mnickel':
                    mod_nickel = all_chain[:, :, fnickel_index]
                    mod_mass = all_chain[:, :, mejecta_index]
                    param_chain = mod_nickel * mod_mass
                elif param == 'MJD0':
                    param_chain = all_chain[:, :, texplosion_index] + MJD0
                elif param == 'TSD':
                    mod_pspin = all_chain[:, :, pspin_index]
                    mod_bfield = all_chain[:, :, bfield_index]
                    mod_Mns = all_chain[:, :, Mns_index]
                    param_chain = np.log10(1.3E5 * (mod_pspin ** 2) * (mod_bfield ** -2) * (mod_Mns / 1.4))
                elif param == 'L0':
                    mod_pspin = all_chain[:, :, pspin_index]
                    mod_bfield = all_chain[:, :, bfield_index]
                    param_chain = np.log10(2E47 * (mod_pspin ** -4) * (mod_bfield ** 2))

            # Unless it is not a derived parameter
            else:
                chain_index = i
                if param_log:
                    param_chain = np.log10(all_chain[:, :, chain_index])
                else:
                    param_chain = all_chain[:, :, chain_index]

            # Calculate the best estimate and error bars, only using the second half of walkers
            param_values = calc_percentile(param_chain[:, int(n_steps*burn_in):])
            param_values_log = calc_percentile(10 ** param_chain[:, int(n_steps*burn_in):])

            # Append Names and best values
            if param_log:
                title_name = r'$\log{(%s)}$' % param_latex
            else:
                title_name = r'$%s$' % param_latex
            if plot_corner:
                if (param not in derived) or append_derived:
                    truths.append(param_values[0])
                    names.append(title_name)

                # Append the chains for the corner plot
                reshaped = param_chain.reshape(all_chain.shape[0],
                                               all_chain.shape[1]).T[int(n_steps*burn_in):].flatten()
                if (param not in derived) or append_derived:
                    if 'corner_chain' not in locals():
                        corner_chain = reshaped
                    else:
                        corner_chain = np.vstack((corner_chain, reshaped))

            # Save the best values
            param_dir = os.path.join(output_dir, param + '.txt')
            if param_log:
                np.savetxt(param_dir, param_values_log)
            else:
                np.savetxt(param_dir, param_values)

            # Plot a trace plot
            plot_trace(param_chain, param_values, param_values_log, min_val, max_val,
                       title_name, param, param_log, n_steps, burn_in, output_dir)

            # If it's log, convert to linear value
            if param_log:
                parameter_array = 10 ** param_chain[:, -1]
            else:
                parameter_array = param_chain[:, -1]

            if param == 'Bfield':
                parameter_array *= 1E14

            if param == 'L0':
                parameter_array = np.log10(parameter_array)

            # Append to array if it exists
            if len(output_parameters) == 0:
                output_parameters = parameter_array
            else:
                output_parameters = np.vstack([output_parameters, parameter_array])
            output_names = np.append(output_names, param)

    # Get Model Realizations
    realizations = [[] for x in range(len(model['realizations']))]
    for ph in photo:
        rn = ph.get('realization', None)
        si = ph.get('simulated', False)
        if (rn and not si):
            fi = ph.get('band', False)
            if fi == reference_band:
                realizations[int(rn) - 1].append((float(ph['time']), float(ph['magnitude'])))

    # Generate grid of times and magnitudes
    max_length = np.max([len(i) for i in realizations])
    max_width = len(realizations)

    if max_length > 0:
        # Fill them with empty values
        times = np.ones((max_width, max_length)) * np.nan
        mags = np.ones((max_width, max_length)) * 99

        # Fill them with real values
        for i in range(len(realizations)):
            xs, ys = zip(*realizations[i])
            times[i][-len(xs):], mags[i][-len(xs):] = xs, ys
        # Set the cells with no times to a magnitude of 99
        mags[mags == 99] = np.nanmax(mags)

        # Collapse the times to one array
        time = np.nanmean(times, axis=0)

        # Get peaks
        peaks = np.nanargmin(mags, axis=1)

        brightest_mag = np.nanmin(mags, axis=1)
        brightest_day = time[peaks]

        bright_mag = calc_percentile(brightest_mag)
        print('%s ± %s' % (np.around(bright_mag[0], 1), np.around(0.5 * (bright_mag[1] + bright_mag[2]), 1)))

        bright_phase = calc_percentile(brightest_day)
        print('%s ± %s' % (np.around(bright_phase[0], 1), np.around(0.5 * (bright_phase[1] + bright_phase[2]), 1)))

        mag_dir = os.path.join(output_dir, 'peak_mag.txt')
        day_dir = os.path.join(output_dir, 'peak_MJD.txt')
        np.savetxt(mag_dir, bright_mag)
        np.savetxt(day_dir, bright_phase)

    # Save output parameters
    if is_slsn:
        output_parameters[output_names == 'Bfield'] = np.log10(output_parameters[output_names == 'Bfield'])
        output_parameters[output_names == 'nhhost'] = np.log10(output_parameters[output_names == 'nhhost'])
        output_parameters[output_names == 'KE'] = np.log10(output_parameters[output_names == 'KE'])

        # And modify their names
        output_names[output_names == 'Bfield'] = 'log(Bfield)'
        output_names[output_names == 'nhhost'] = 'log(nhhost)'
        output_names[output_names == 'KE'] = 'log(KE)'
        output_names[output_names == 'L0'] = 'log(L0)'

        # Append light curve parameters
        output_parameters = np.vstack([output_parameters, brightest_mag])
        output_names = np.append(output_names, 'Peak_mag')
        output_parameters = np.vstack([output_parameters, brightest_day])
        output_names = np.append(output_names, 'Peak_MJD')

        # Write final data
        final_data = np.round(output_parameters.T, 5)
        output_table = table.Table(final_data, names=output_names)
        output_table_dir = os.path.join(output_dir, 'output_parameters.txt')
        output_table.write(output_table_dir, format='ascii.fixed_width', delimiter=None, overwrite=True)

    # Plot the corner plot
    if plot_corner:
        print('Plotting corner plot...')
        import corner
        fig = corner.corner(corner_chain.T, labels=names, truths=truths, show_titles=True,
                            quantiles=[0.1587, 0.50, 0.8413], title_kwargs={"fontsize": 20},
                            label_kwargs={"fontsize": 20}, use_math_text=True, smooth=2)
        corner_path = os.path.join(output_dir, 'corner.pdf')
        fig.savefig(corner_path)
        plt.clf()
        plt.close('all')


def plot_mosfit_lc(data, object_name, explosion_time, redshift, output_dir, plot_ignored=False):
    """
    This function plots the MOSFiT light curve and photometry
    of a supernova.

    Parameters
    ----------
    data : dict
        The dictionary with all the MOSFiT data.
    object_name : str
        The name of the object to import.
    explosion_time : float
        The time of explosion in MJD.
    redshift : float
        The redshift of the object.
    output_dir : str
        The directory to save the plot.
    plot_ignored : bool, default False
        Whether to plot the ignored photometry.
    """

    # Get MOSFiT model
    model = data['models'][0]
    # Get photometry
    phot = get_lc(object_name, lc_type='phot')
    # Calculate observed phase
    observed_phase = (phot['MJD'] - explosion_time) / (1 + redshift)

    # Get MOSFiT data
    photo = data['photometry']

    # Find a valid realization
    run = True
    ind = 0
    while run:
        try:
            r0 = photo[ind]['realization']
            run = False
        except Exception:
            ind += 1
    # Get the time of the first datapoint
    t0 = photo[ind]['time']

    # Get number of walkers
    walkers = np.array([i['realization'] if 'realization' in i else None for i in photo]).astype(float)
    n_walkers = int(np.nanmax(walkers))

    # Empty arrays for MOSFiT data
    band_array = np.array([])
    instrument_array = np.array([])
    telescope_array = np.array([])
    system_array = np.array([])
    magnitude_array = np.array([])

    # Get Model Realizations
    for ph in photo[:int(len(photo)/n_walkers)]:
        rn = ph.get('realization', None)
        si = ph.get('simulated', False)
        if rn and not si:
            if (ph.get('time') == t0) & (ph.get('realization') == r0):
                band = ph.get('band', None)
                instrument = ph.get('instrument', None)
                telescope = ph.get('telescope', None)
                system = ph.get('system', None)
                magnitude = ph.get('magnitude', None)
                if band and system:
                    band_array = np.append(band_array, band)
                    instrument_array = np.append(instrument_array, instrument)
                    telescope_array = np.append(telescope_array, telescope)
                    system_array = np.append(system_array, system)
                    magnitude_array = np.append(magnitude_array, magnitude)
    # Only use the sets that have unique magnitudes
    observed_mosfit_magnitude = np.unique(magnitude_array)

    # Empty array to specify model
    phot['Model'] = [np.nan] * len(phot)

    # Ad-hoc solution for Gaia
    if 'Gaia' in phot['Instrument']:
        phot['Instrument'] = phot['Instrument'].astype('U30')
        phot['Instrument'][phot['Instrument'] == 'Gaia'] = 'Astrometric'

    # Assign Corresponding model and marker index
    for j in range(len(observed_mosfit_magnitude)):
        match = magnitude_array == observed_mosfit_magnitude[j]
        # Get parameters that use this magnitude
        match_band = band_array[match]
        match_telescope = telescope_array[match]
        match_instrument = instrument_array[match]
        match_system = system_array[match]
        print(j, '=', match_band, match_telescope, match_instrument, match_system)
        for k in range(len(match_band)):
            k_match = (phot['Filter'] == match_band[k]) & \
                      (phot['Telescope'] == match_telescope[k]) & \
                      (phot['Instrument'] == match_instrument[k]) & \
                      (phot['System'] == match_system[k])
            phot['Model'][k_match] = j

    # Get central wavelengths and zeropoints
    cenwaves, zeropoints = quick_cenwave_zeropoint(phot)
    phot['cenwave'] = cenwaves
    phot['zeropoint'] = zeropoints

    # Get the corresponding model and wavelength for each model
    models_used = np.sort(np.unique(phot['Model']))
    band_waves = np.array([phot[phot['Model'] == mod]['cenwave'][0] for mod in models_used[np.isfinite(models_used)]])
    models_used[np.argsort(band_waves)]

    # Get bands used
    band_filts, band_ind = np.unique(phot['Filter'], return_index=True)
    bands_used = band_filts[np.argsort(phot['cenwave'][band_ind])][::-1]

    # Marker list for plotting
    markers = np.array(['o', 's', '*', 'p', 'P', '<', '>'])

    # If only part of a band is ignored, don't replot it at a different offset
    n_offset = 0
    for band in bands_used:
        n_band = phot['Filter'] == band
        unique_models = np.unique(phot['Model'][n_band])

        if (len(unique_models) > 1) & np.any(np.isnan(unique_models)):
            unique_models = unique_models[np.isfinite(unique_models)]
            phot['Model'][np.isnan(phot['Model']) & n_band] = unique_models[0]

        n_offset += len(np.unique(phot['Model'][n_band]))

    # Plot limits
    xmin = round(np.min([-50, np.min(observed_phase) - 10]))
    xmax = round(np.max([200, np.max(observed_phase) + 10]) / 10) * 10
    ymin = np.round(np.min([20.0, np.min(phot['Mag']) - 0.5]), 1)
    ymax = np.round(np.max([26.0, np.max(phot['Mag']) + 0.5 + n_offset]), 1)
    delta_time = xmax - xmin
    delta_mag = ymax - ymin

    # Plot Parameters
    plt.gca().invert_yaxis()
    plt.gca().set_xlim(xmin, xmax)
    plt.gca().set_ylim(bottom=ymax, top=ymin)
    if delta_time > 800:
        plt.gca().set_xticks(np.arange(xmin, xmax, 125))
    elif delta_time > 400:
        plt.gca().set_xticks(np.arange(xmin, xmax, 50))
    else:
        plt.gca().set_xticks(np.arange(xmin, xmax, 25))
    if delta_mag > 20:
        plt.gca().set_yticks(np.arange(ymin, ymax+1, 2))
    else:
        plt.gca().set_yticks(np.arange(ymin, ymax+1, 1))
    plt.gca().set_xlabel('Phase [rest days]')
    plt.gca().set_ylabel('Apparent Magnitude + Constant')

    # Empty arrays for plotting
    shift = 0.0
    final_times = np.array([])
    final_upper = np.array([])
    final_mean = np.array([])
    final_lower = np.array([])
    final_cenwav = np.array([])
    final_zerop = np.array([])
    final_names = np.array([])

    # Plot each photometry band
    for band in bands_used:
        n_band = phot['Filter'] == band
        unique_models = np.unique(phot['Model'][n_band])

        # Plot each MOSFiT model
        for mod in range(len(unique_models)):
            if np.isnan(unique_models[mod]):
                n_mod = np.isnan(phot['Model'])
            else:
                n_mod = phot['Model'] == unique_models[mod]

            # Get corresponding central wavelength and zeropoint
            cenwave = phot['cenwave'][n_mod][0]
            zeropoint = phot['zeropoint'][n_mod][0]

            # Get parameters that use this magnitude
            unique_table = table.unique(phot['Filter', 'Telescope', 'Instrument', 'System'][n_mod])
            used_filters = list(unique_table['Filter'])
            used_telescopes = list(unique_table['Telescope'])
            used_instruments = list(unique_table['Instrument'])
            used_systems = list(unique_table['System'])

            # Produce Model Realizations
            realizations = [[] for x in range(len(model['realizations']))]
            for ph in photo:
                rn = ph.get('realization', None)
                si = ph.get('simulated', False)
                if (rn and not si):
                    fi = ph.get('band', False)
                    te = ph.get('telescope', False)
                    it = ph.get('instrument', False)
                    ab = ph.get('system', False)
                    if (fi in used_filters) & (te in used_telescopes) & (it in used_instruments) & (ab in used_systems):
                        sucess = np.any([(fi == f) & (te == t) & (it == i) & (ab == s)
                                         for f, t, i, s in zip(used_filters, used_telescopes, used_instruments,
                                                               used_systems)])
                        if sucess:
                            realizations[int(rn) - 1].append((float(ph['time']), float(ph['magnitude'])))

            # Generate grid of times and magnitudes
            max_length = np.max([len(i) for i in realizations])
            max_width = len(realizations)

            if max_length > 0:
                # Fill them with empty values
                times = np.ones((max_width, max_length)) * np.nan
                mags = np.ones((max_width, max_length)) * 99

                # Fill them with real values
                for i in range(len(realizations)):
                    xs, ys = zip(*realizations[i])
                    times[i][-len(xs):], mags[i][-len(xs):] = xs, ys
                # Set the cells with no times to a magnitude of 99
                mags[mags == 99] = np.nanmax(mags)

                # Collapse the times to one array
                time = np.nanmean(times, axis=0)

                # Get 1 sigma errors
                lower_mags, mean_mags, upper_mags = np.nanpercentile(mags, [15.87, 50, 84.13], axis=0)

                # Calculate phase
                phase = (time - explosion_time) / (1 + redshift)

                # Plot Model
                plt.plot(phase, mean_mags + shift, color=plot_colors(band))
                plt.fill_between(phase, lower_mags + shift, upper_mags + shift,
                                 color=plot_colors(band), alpha=0.5, linewidth=0)

                if time[0] == time[1]:
                    length = len(time[::2])
                    final_times = np.append(final_times, time[::2])
                    final_upper = np.append(final_upper, upper_mags[::2])
                    final_mean = np.append(final_mean, mean_mags[::2])
                    final_lower = np.append(final_lower, lower_mags[::2])
                    final_cenwav = np.append(final_cenwav, np.array([cenwave] * length))
                    final_zerop = np.append(final_zerop, np.array([zeropoint] * length))
                else:
                    length = len(time)
                    final_times = np.append(final_times, time)
                    final_upper = np.append(final_upper, upper_mags)
                    final_mean = np.append(final_mean, mean_mags)
                    final_lower = np.append(final_lower, lower_mags)
                    final_cenwav = np.append(final_cenwav, np.array([cenwave] * length))
                    final_zerop = np.append(final_zerop, np.array([zeropoint] * length))

            # Plot Photoemtry
            match_det = phot['UL'] == 'False'
            match_UL = phot['UL'] == 'True'
            ignore = phot['Ignore'] == 'True'

            # Pick Marker
            marker = markers[mod]

            # Make label
            label = band

            if len(unique_models) > 1:
                telescopes_model = np.unique(np.array([phot['Telescope'][phot['Model'] == m][0] for
                                                       m in unique_models[np.isfinite(unique_models)]]))
                instruments_model = np.unique(np.array([phot['Instrument'][phot['Model'] == m][0] for
                                                        m in unique_models[np.isfinite(unique_models)]]))
                systems_model = np.unique(np.array([phot['System'][phot['Model'] == m][0] for
                                                    m in unique_models[np.isfinite(unique_models)]]))
                if len(telescopes_model) > 1:
                    if len(used_telescopes) == 1:
                        label += f'-{used_telescopes[0]}'
                elif len(instruments_model) > 1:
                    if used_instruments[0] != '--':
                        label += f'-{used_instruments[0]}'
                elif len(systems_model) > 1:
                    label += f'-{used_systems[0]}'

            if max_length > 0:
                final_names = np.append(final_names, np.array([label] * length))

            if shift > 0:
                label += ' + %s' % np.around(shift, decimals=1)

            plt.errorbar(observed_phase[match_det & ~ignore & n_mod], phot[match_det & ~ignore & n_mod]['Mag'] + shift,
                         yerr=phot[match_det & ~ignore & n_mod]['MagErr'], color=plot_colors(band), markersize=8,
                         alpha=1.0, fmt=marker, markeredgecolor='black', markeredgewidth=1, elinewidth=1, label=label)
            plt.errorbar(observed_phase[match_UL & ~ignore & n_mod], phot[match_UL & ~ignore & n_mod]['Mag'] + shift,
                         color=plot_colors(band), markersize=8, alpha=1.0, fmt='v', markeredgecolor='black',
                         markeredgewidth=1, elinewidth=1)
            if plot_ignored:
                plt.errorbar(observed_phase[match_det & ignore & n_mod],
                             phot[match_det & ignore & n_mod]['Mag'] + shift,
                             yerr=phot[match_det & ignore & n_mod]['MagErr'],
                             color=plot_colors(band), markersize=8, alpha=0.3, fmt=marker)
                plt.errorbar(observed_phase[match_UL & ignore & n_mod],
                             phot[match_UL & ignore & n_mod]['Mag'] + shift,
                             color=plot_colors(band), markersize=8, alpha=0.3, fmt='v')

            print(f'Plotted {label} band')
            shift += 1.0

    plt.margins(0.02, 0.1)
    plt.legend(ncol=3, bbox_to_anchor=(0.0, 1.02), loc='lower left', frameon=True)
    plot_dir = os.path.join(output_dir, f'{object_name}_mosfit.pdf')
    plt.savefig(plot_dir, bbox_inches='tight')
    plt.clf()
    plt.close('all')

    # Create output array of light curves
    final_times = np.round(final_times, 3)
    final_upper = np.round(final_upper, 3)
    final_mean = np.round(final_mean, 3)
    final_lower = np.round(final_lower, 3)
    final_cenwav = np.round(final_cenwav, 3)
    final_zerop = np.round(final_zerop, 3)

    stacked_data = np.array([final_times, final_upper, final_mean, final_lower, final_cenwav, final_zerop, final_names])
    stacked_names = ['MJD', 'Upper', 'Mean', 'Lower', 'Cenwave', 'Zeropoint', 'Filter']

    # Sort, clean, and save.
    stacked_table = table.Table(stacked_data.T, names=stacked_names)
    stacked_table = table.unique(stacked_table)
    lightcurve_dir = os.path.join(output_dir, f'{object_name}_model.txt')
    stacked_table.write(lightcurve_dir, format='ascii.fixed_width',
                        delimiter=None, overwrite=True)


def get_mosfit_bolometric(extras, data, object_name, redshift, output_dir):
    """
    This function calculates the bolometric light curve from the MOSFiT output.

    Parameters
    ----------
    extras : dict
        The dictionary with all the MOSFiT extra parameters.
    data : dict
        The dictionary with all the MOSFiT data.
    object_name : str
        The name of the object to import.
    redshift : float
        The redshift of the object.
    output_dir : str
        The directory to save the plot
    """

    # Get reference time from MOSFiT photometry
    t_ref = data['models'][0]['realizations'][0]['parameters']['reference_texplosion']['value']

    # Read MOSFiT Output
    radiusphot = np.array(extras['radiusphot'])
    temperaturephot = np.array(extras['temperaturephot'])
    luminosities = np.array(extras['luminosities_out'])
    times_in = np.array(extras['times_out']) * (1 + redshift) + t_ref

    # Extract Parameters
    times_out, time_ind = np.unique(times_in, return_index=True)
    rad_out = radiusphot.T[time_ind].T
    tem_out = temperaturephot.T[time_ind].T
    lum_out = luminosities.T[time_ind].T

    # Clean pre-explosion data
    pre = (rad_out == 0) | np.isinf(tem_out)
    rad_out[pre] = 0
    tem_out[pre] = 0
    lum_out[pre] = 0

    # Get 1-sigma ranges
    rad_low, rad_med, rad_high = np.nanpercentile(rad_out, [15.87, 50, 84.13], axis=0)
    tem_low, tem_med, tem_high = np.nanpercentile(tem_out, [15.87, 50, 84.13], axis=0)
    lum_low, lum_med, lum_high = np.nanpercentile(lum_out, [15.87, 50, 84.13], axis=0)

    # Save output
    output = (times_out, rad_low, rad_med, rad_high, tem_low, tem_med, tem_high, lum_low, lum_med, lum_high)
    colnames = ('MJD', 'R_low', 'R_med', 'R_high', 'T_low', 'T_med', 'T_high', 'L_low', 'L_med', 'L_high')
    table_out = table.Table(output, names=colnames)
    bol_dir = os.path.join(output_dir, f'{object_name}_bol.txt')
    table_out.write(bol_dir, format='ascii.fixed_width', delimiter=None, overwrite=True)

    # Boom day and Peak day indeces
    peak_ind = np.argmax(lum_out, axis=1)
    boom_ind = np.array([np.min(np.where(i > 0)) for i in lum_out])
    # And values
    peak = times_out[peak_ind]
    boom = times_out[boom_ind]

    # Rise time
    rise_time_array = (peak - boom) / (1 + redshift)
    rise_time = calc_percentile(rise_time_array)
    rise_time_dir = os.path.join(output_dir, 'rise_time.txt')
    np.savetxt(rise_time_dir, rise_time)

    # Peak Luminosity
    brightest_lum = np.nanmax(lum_out, axis=1)
    peak_lum = calc_percentile(np.log10(brightest_lum))
    peak_lum_dir = os.path.join(output_dir, 'peak_lum.txt')
    np.savetxt(peak_lum_dir, peak_lum)

    # Calculate Total luminosity
    total_lum = itg.trapz(lum_out, times_out*3600*24) / (1 + redshift)
    bol_out = calc_percentile(np.log10(total_lum))
    e_rad_dir = os.path.join(output_dir, 'E_rad.txt')
    np.savetxt(e_rad_dir, bol_out)

    # 1/e of the maximum
    elum = brightest_lum / np.e

    # Calculate e-fold decline time
    e_fold_array = np.array([])
    for i in range(len(lum_out)):
        # Only consider phases after peak
        late = peak_ind[i]
        late_l = lum_out[i][late:]
        late_t = times_out[late:]

        # Index of e-fold time
        e_ind = np.argmin(np.abs(late_l - elum[i]))

        # Append answer
        out = (late_t[e_ind] - peak[i]) / (1 + redshift)
        e_fold_array = np.append(e_fold_array, out)
    e_out = calc_percentile(e_fold_array)
    e_fold_dir = os.path.join(output_dir, 'e_fold.txt')
    np.savetxt(e_fold_dir, e_out)

    # If output_parameters.txt exists, read it in and append four columns for e_rad, peak_lum, rise_time, and e_fold
    output_table_dir = os.path.join(output_dir, 'output_parameters.txt')
    if os.path.exists(output_table_dir):
        output_table = table.Table.read(output_table_dir, format='ascii')
        # Only do this if the columns don't already exist
        if 'log(E_rad)' not in output_table.colnames:
            output_table['log(E_rad)'] = np.round(np.log10(total_lum), 5)
        if 'log(Peak_lum)' not in output_table.colnames:
            output_table['log(Peak_lum)'] = np.round(np.log10(brightest_lum), 5)
        if 'Rise_Time' not in output_table.colnames:
            output_table['Rise_Time'] = np.round(rise_time_array, 5)
        if 'E_fold' not in output_table.colnames:
            output_table['E_fold'] = np.round(e_fold_array, 5)
        output_table.write(output_table_dir, format='ascii.fixed_width', delimiter=None, overwrite=True)


def process_rest_frame(object_name, output_dir, redshift, save_rest_frame=True):
    """
    This function calculates the rest-frame parameters for a given object.

    Parameters
    ----------
    object_name : str
        The name of the object to import.
    output_dir : str
        The directory with the output_parameters.txt file.
    redshift : float
        The redshift of the object.
    save_rest_frame : bool, default True
        Whether to create and save the rest-frame light curve.
    """

    # Read output parameter table
    output_table_dir = os.path.join(output_dir, 'output_parameters.txt')
    output_table = table.Table.read(output_table_dir, format='ascii')

    # Bands needed for rest-frame calculations
    if save_rest_frame:
        bands = ['swift_UVW2', 'swift_UVM2', 'swift_UVW1', 'swift_U', 'U', 'u',
                 'swift_B', 'B', 'g', 'swift_V', 'V', 'r', 'R', 'i', 'I', 'z', 'y',
                 'J', 'H', 'Ks', 'W1', 'W2']
        bandwaves = {i: get_cenwave(i, verbose=False) for i in bands}
    else:
        bands = ['B', 'r']

    # Phases to use for calculations
    phases = np.append(np.linspace(0, 100, 201), np.linspace(100, 800, 351))

    # Always assumed to be extinction corrected
    ebv = 0

    # Empty list for models
    model_obs = []
    fractions = np.array([])

    # Calculate SEDs for each walker
    for j in range(len(output_table)):
        print('\t', j + 1, '/', len(output_table))
        param = output_table[j]
        Pspin = param['Pspin']
        Bfield = 10 ** param['log(Bfield)'] / 1e14
        Mns = param['Mns']
        thetaPB = param['thetaPB']
        # texplosion = param['texplosion']
        kappa = param['kappa']
        log_kappa_gamma = np.log10(param['kappagamma'])
        mejecta = param['mejecta']
        if 'fnickel' in param.colnames:
            fnickel = param['fnickel']
        else:
            fnickel = 0.0
        v_ejecta = param['vejecta']
        temperature = param['temperature']
        cut_wave = param['cutoff_wavelength']
        alpha = param['alpha']

        # Modify parameters for rest-frame models
        texplosion_rest = param['texplosion'] - np.median(output_table['texplosion'])
        redshift_rest = 0
        log_nh_host_rest = 16

        # Get magnitudes in each band
        output_mags = slsnni(phases, Pspin, Bfield, Mns, thetaPB, texplosion_rest, kappa,
                             log_kappa_gamma, mejecta, fnickel, v_ejecta, temperature,
                             cut_wave, alpha, redshift_rest, log_nh_host_rest, ebv, bands)

        # Calculate corresponding nickel and magnetar luminosities
        nickel_lum = nickelcobalt(phases, fnickel, mejecta, rest_t_explosion=0)
        magnetar_lum = magnetar(phases, Pspin, Bfield, Mns, thetaPB, rest_t_explosion=0)

        # Integrate to get total luminosity
        nickel_total = np.trapz(nickel_lum, phases * 24 * 3600)
        magnetar_total = np.trapz(magnetar_lum, phases * 24 * 3600)

        # Calculate the fraction of magnetar luminosity
        fraction = magnetar_total / (nickel_total + magnetar_total)

        # Append to list
        fractions = np.append(fractions, fraction)
        model_obs.append(output_mags)

    # Get rest-frame r and B-bands
    r_bands = np.array([k['r'] for k in model_obs])
    B_bands = np.array([k['B'] for k in model_obs])

    # Get the index of peak
    r_peak_ind = np.nanargmin(r_bands, axis=1)

    # Get the value of the peak
    r_peak = np.nanmin(r_bands, axis=1)

    # Get the time of peak
    r_peak_phase = phases[r_peak_ind]

    # Get the time at which r-band has declined by 1-mag
    dim_phase = np.array([phases[k:][np.argmin(np.abs(m[k:] - (b + 1)))]
                         for m, k, b in zip(r_bands, r_peak_ind, r_peak)])
    tau1_phase = dim_phase - r_peak_phase

    # Calculate Delta_m15
    B_peak_ind = np.nanargmin(B_bands, axis=1)

    # Get brightest Mag
    B_peak = np.nanmin(B_bands, axis=1)
    B_peak_phase = phases[B_peak_ind]

    # Calculate Delta m15
    ind_15 = np.array([np.argmin(np.abs(phases - (i + 15))) for i in B_peak_phase])
    late_mags = np.array([i[o] for i, o in zip(B_bands, ind_15)])
    Delta_m15 = late_mags - B_peak

    # Only do this if the columns don't already exist
    if 'tau_1' not in output_table.colnames:
        output_table['tau_1'] = np.round(tau1_phase, 5)
    if 'delta_m15' not in output_table.colnames:
        output_table['delta_m15'] = np.round(Delta_m15, 5)
    if 'r_peak' not in output_table.colnames:
        output_table['r_peak'] = np.round(r_peak, 5)
    if 'frac' not in output_table.colnames:
        output_table['frac'] = np.round(fractions, 5)
    output_table.write(output_table_dir, format='ascii.fixed_width', delimiter=None, overwrite=True)

    # Save the rest frame light curve
    if save_rest_frame:
        # Empty arrays for creating output table
        final_MJD = np.array([])
        final_phase = np.array([])
        final_upper = np.array([])
        final_mean = np.array([])
        final_lower = np.array([])
        final_bands = np.array([])
        final_cenwav = np.array([])

        # Calculate the phase for this SN
        times = phases * (1 + redshift) + np.median(output_table['MJD0'])

        # Calculate the mean and ± 1 sigma of each band
        for band in bands:
            mags = np.array([k[band] for k in model_obs])
            lower_mags, mean_mags, upper_mags = np.nanpercentile(mags, [15.87, 50, 84.13], axis=0)
            cenwave = bandwaves[band]

            final_MJD = np.append(final_MJD, times)
            final_phase = np.append(final_phase, phases)
            final_upper = np.append(final_upper, upper_mags)
            final_mean = np.append(final_mean, mean_mags)
            final_lower = np.append(final_lower, lower_mags)
            final_bands = np.append(final_bands, np.array([band] * len(times)))
            final_cenwav = np.append(final_cenwav, np.array([cenwave] * len(times))).astype(float)

        # Create output array of light curves
        final_MJD = np.round(final_MJD, 3)
        final_phase = np.round(final_phase, 3)
        final_upper = np.round(final_upper, 3)
        final_mean = np.round(final_mean, 3)
        final_lower = np.round(final_lower, 3)
        final_cenwav = np.round(final_cenwav, 3)

        stacked_data = np.array([final_MJD, final_phase, final_upper, final_mean,
                                 final_lower, final_cenwav, final_bands])
        stacked_names = ['MJD', 'Phase', 'Upper', 'Mean', 'Lower', 'Cenwave', 'Filter']
        stacked_table = table.Table(stacked_data.T, names=stacked_names)
        output_name = os.path.join(output_dir, f'{object_name}_rest.txt')
        stacked_table.write(output_name, format='ascii.fixed_width', delimiter=None, overwrite=True)


def process_mosfit(object_name, mosfit_dir, output_dir, data_table=None, redshift=None, plot_parameters=True,
                   plot_corner=True, plot_lc=True, plot_bol=True, calc_rest=True, save_rest_frame=True):
    """
    This function processes the MOSFiT output for a given object. Combining the other
    functions in this module.

    Parameters
    ----------
    object_name : str
        The name of the object to import.
    mosfit_dir : str
        The directory where the MOSFiT output is located.
    output_dir : str
        The directory to save the plots.
    data_table : table.Table, default None
        The table with all the data.
    redshift : float, default None
        The redshift of the object.
    plot_parameters : bool, default True
        Whether to plot the parameters.
    plot_corner : bool, default True
        Whether to plot the corner plot.
    plot_lc : bool, default True
        Whether to plot the light curve.
    plot_bol : bool, default True
        Whether to plot the bolometric light curve.
    calc_rest : bool, default False
        Whether to calculate the rest frame parameters.
    save_rest_frame : bool, default True
        Whether to save the rest frame light curve.
    """

    # Import MOSFiT data
    print('\nProcessing', object_name)
    all_chain, chain_names, data, extras = import_mosfit(object_name, mosfit_dir, import_extras=True)

    # Get redshift
    if redshift is not None:
        pass
    else:
        if data_table is None:
            data_table = get_data_table()
        redshift = data_table['Redshift'][data_table['Name'] == object_name][0]

    # Plot the trace and corner plot and save output parameters
    if plot_parameters:
        plot_params(all_chain, chain_names, data, output_dir, plot_corner=plot_corner)

    # Get explosion time
    mjd_dir = os.path.join(output_dir, 'MJD0.txt')
    explosion_time = np.genfromtxt(mjd_dir)[0]

    # Plot the MOSFiT light curve and save model light curve
    if plot_lc:
        plot_mosfit_lc(data, object_name, explosion_time, redshift, output_dir)

    # Create the bolometric evolution
    if plot_bol:
        get_mosfit_bolometric(extras, data, object_name, redshift, output_dir)

    # Calculate rest frame parameters
    if calc_rest:
        process_rest_frame(object_name, output_dir, redshift, save_rest_frame)
