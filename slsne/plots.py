"""
This file contains utilities to create plots for SLSN data.

Written by Sebastian Gomez, 2024.
"""

# Import necessary packages
import numpy as np
import matplotlib.pyplot as plt
from .utils import get_params, get_data_table, cb_g, cb_r
from matplotlib import gridspec
import os
from scipy.optimize import curve_fit
import matplotlib.patheffects as pe
from astropy import table
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.family': 'serif'})

# Get directory with reference data
current_file_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_file_dir, 'ref_data')

# Create dictionary with parameter limits, whether it is log, and the name
param_dict = {'redshift': [0.0, 2.2, False, r'$z$'],
              'texplosion': [-200, 0, False, r'$t_{\rm exp}$ [Days]'],
              'fnickel': [0.001, 0.5, True, r'f$_{\rm Ni}$'],
              'Pspin': [0.2, 20.0, False, r'P$_{\rm spin}$ [ms]'],
              'log(Bfield)': [12.5, 15.2, False, r'log(B$_{\perp}$ / G)'],
              'Mns': [1.0, 2.2, False, r'M$_{\rm NS}\ [{\rm M}_\odot]$'],
              'thetaPB': [0.0, np.pi / 2, False, r'$\theta_{\rm PB}$ [rad]'],
              'mejecta': [0.5, 120.0, True, r'M$_{\rm ej}\ [{\rm M}_\odot]$'],
              'kappa': [0.01, 0.34, False, r'$\kappa$ [cm$^2$g$^{-1}$]'],
              'kappagamma': [0.009, 0.5, True, r'$\kappa_\gamma$ [cm$^2$g$^{-1}$]'],
              'vejecta': [1.8, 33.0, True, r'V$_{\rm ej}$ [1000 km s$^{-1}$]'],
              'temperature': [3000.0, 10000.0, False, r'T [K]'],
              'alpha': [0.0, 5.0, False, r'P$_{\rm cutoff}$'],
              'cutoff_wavelength': [2000, 6000, False, r'$\lambda_{\rm cutoff}$' + ' [\u212b]'],
              'log(nhhost)': [17, 22, False, r'$\log{n_{\rm H,host}}$ [cm$^{-2}$]'],
              'A_V': [0.0, 1.0, False, r'A$_{\rm V}$ [mag]'],
              'MJD0': [48000, 60000, False, r'MJD$_0$'],
              'log(kenergy)': [50.6, 52.7, False, r'log(E$_K$ / erg)'],
              'mnickel': [0.0015, 50.0, True, r'M$_{\rm Ni}\ [{\rm M}_\odot]$'],
              'log(TSD)': [3, 10, False, r'$\log(t_{\rm SD}\ /\ s)$'],
              'log(L0)': [40, 50, False, r'$\log(L_0\ /\ {\rm erg\ s}^{-1})$'],
              'Peak_mag': [14, 25, False, r'$m_{\rm r}$ [mag]'],
              'Peak_MJD': [48000, 60000, False, r'MJD$_{\rm peak}$'],
              'log(E_rad)': [49.8, 51.9, False, r'$\log(E\ /\ erg)$'],
              'log(Peak_lum)': [43.0, 45.5, False, r'$\log(L_{\rm max}$ / erg s$^{-1}$)'],
              'Rise_Time': [6, 200, True, r'$\tau_{\rm rise}$ [Days]'],
              'E_fold': [10, 300, True, r'$\tau_{e}$ [Days]',],
              'tau_1': [0, 220, False, r'$\tau_{\rm 1}$ [Days]',],
              'delta_m15': [0.0, 0.7, False, r'$\Delta m_{15}$ [mag]'],
              'r_peak': [-19, -23, False, r'$M_{\rm r, peak}$ [mag]'],
              'frac': [0, 1.02, False, r'$f_{\rm mag}$'],
              '1frac': [1e-3, 1, True, r'1-$f_{\rm mag}$'],
              'efficiency': [5e-3, 2.5, True, r'$\epsilon$']}


def make_plot(param_x, param_y, param_z=None, output_dir='.', remove_bronze=True, plot_tmag=False,
              plot_menergy=False, plot_fnickel=False, plot_mean=False, plot_1_1=False,
              include_frac=False, include_others=False, individual_name=None):
    """
    This function creates a correlation plot between two parameters.

    Parameters
    ----------
    param_x : str
        The parameter to be plotted on the x-axis.
    param_y : str
        The parameter to be plotted on the y-axis.
    param_z : str, optional
        The parameter to be used as the color scale.
    output_dir : str, default='.'
        The directory where the plot will be saved.
    remove_bronze : bool, default=True
        Whether to remove the bronze SLSNe from the plot.
    plot_tmag : bool, default=False
        Whether to plot the magnetar diffusion timescale.
    plot_menergy : bool, default=False
        Whether to plot the magnetar energy.
    plot_fnickel : bool, default=False
        Whether to plot the lines of constant nickel fraction.
    plot_mean : bool, default=False
        Whether to plot lines show the mean and 1-sigma range
        for the parameters.
    plot_1_1 : bool, default=False
        Wheather to plot a 1-1 correlation line.
    include_frac : bool, default=False
        Whether to include markers for the SNe with low fraction
        of magnetar energy.
    include_others : bool, default=False
        Whether to include markers for comparsions to other works.
    individual_name : str, default=None
        The name of the individual object to be plotted.
    """

    # Import parameters data
    params = get_params()

    # Remove Bronze objects by default
    if remove_bronze:
        data_table = get_data_table()
        good_names = data_table['Name'][data_table['Quality'] != 'Bronze']
        use_names = [i in good_names for i in params['name']]
        params = params[use_names]

    # Make sure param_x and param_y are valid
    if param_x not in param_dict.keys():
        raise ValueError(f'{param_x} is not a valid parameter.')
    if param_y not in param_dict.keys():
        raise ValueError(f'{param_y} is not a valid parameter.')
    if param_z is not None:
        if param_z not in param_dict.keys():
            raise ValueError(f'{param_z} is not a valid parameter.')

    # Modify the ejecta velocity parameter
    params['vejecta_med'] = params['vejecta_med'] / 1e3
    params['vejecta_up'] = params['vejecta_up'] / 1e3
    params['vejecta_lo'] = params['vejecta_lo'] / 1e3

    # Get requested parameter values
    xparam_med = params[f'{param_x}_med']
    xparam_up = params[f'{param_x}_up']
    xparam_lo = params[f'{param_x}_lo']

    yparam_med = params[f'{param_y}_med']
    yparam_up = params[f'{param_y}_up']
    yparam_lo = params[f'{param_y}_lo']

    # Set up figure
    f = plt.figure()
    f.set_size_inches(6.4, 4.8)
    plt.subplots_adjust(hspace=0.0, wspace=0.0)
    alpha = 0.5
    bins = 40

    # Get axis limits
    xmin, xmax = param_dict[param_x][0], param_dict[param_x][1]
    ymin, ymax = param_dict[param_y][0], param_dict[param_y][1]

    # Get plot labels and log values
    xlabel = param_dict[param_x][3]
    ylabel = param_dict[param_y][3]
    xlog = param_dict[param_x][2]
    ylog = param_dict[param_y][2]

    if include_others & (param_x == 'Pspin'):
        xlog = True
        xmin = 0.6
        xmax = 50.0

    # Reverse the limits if necessary for the histogram
    if ymax < ymin:
        ymin_h, ymax_h = ymax, ymin
    else:
        ymin_h, ymax_h = ymin, ymax

    if xmax < xmin:
        xmin_h, xmax_h = xmax, xmin
    else:
        xmin_h, xmax_h = xmin, xmax

    if param_z is None:
        gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1.1], height_ratios=[1.1, 3])
        ax1 = plt.subplot(gs[2])
    else:
        gs = gridspec.GridSpec(2, 3, width_ratios=[3, 1, 0.2], height_ratios=[1, 3])
        ax1 = plt.subplot(gs[1, 0])

        # Get z parameter values
        zlog = param_dict[param_z][2]
        zlabel = param_dict[param_z][3]

    # Plot the data with additional markers
    if include_frac:
        blue = ((params['frac_med'] + params['frac_up']) <= 0.90) & (((params['frac_med'] + params['frac_up']) > 0.50))
        red = (params['frac_med'] + params['frac_up']) <= 0.50

        ax1.errorbar(xparam_med[~red & ~blue], yparam_med[~red & ~blue],
                     xerr=[xparam_lo[~red & ~blue], xparam_up[~red & ~blue]],
                     yerr=[yparam_lo[~red & ~blue], yparam_up[~red & ~blue]], fmt='o',
                     color=cb_g, markersize=10, alpha=alpha, markeredgecolor='k', zorder=1000)

        ax1.errorbar(xparam_med[blue], yparam_med[blue],
                     xerr=[xparam_lo[blue], xparam_up[blue]],
                     yerr=[yparam_lo[blue], yparam_up[blue]], fmt='o',
                     color='b', markersize=10, alpha=alpha, markeredgecolor='k', zorder=1000)

        ax1.errorbar(xparam_med[red], yparam_med[red],
                     xerr=[xparam_lo[red], xparam_up[red]],
                     yerr=[yparam_lo[red], yparam_up[red]], fmt='o',
                     color=cb_r, markersize=10, alpha=alpha, markeredgecolor='k', zorder=1000)

    # Plot individual objects
    if individual_name is not None:
        if individual_name not in params['name']:
            raise ValueError(f'{individual_name} is not a valid name.')

        # Get the index of the individual object
        ind = np.where(params['name'] == individual_name)[0][0]

        # Plot the individual object
        ax1.errorbar(xparam_med[ind], yparam_med[ind], xerr=[[xparam_lo[ind]], [xparam_up[ind]]],
                     yerr=[[yparam_lo[ind]], [yparam_up[ind]]], fmt='o', color='b', markersize=10, alpha=0.9,
                     markeredgecolor='k', zorder=2000)

    # Plot the data for comparisons to other works
    if include_others:
        if (param_x == 'Pspin') & (param_y == 'mejecta'):
            # Import data from Aguilera-Dena and Kumar using astropy tables
            aguilera = table.Table.read(os.path.join(data_dir, 'aguilera-dena.txt'), format='ascii')
            kumar = table.Table.read(os.path.join(data_dir, 'kumar.txt'), format='ascii')

            # Calculate Pspin from Aguilera-Dena's data
            # Assuming MOSFiT model from 1971ApJ...164L..95O
            aguilera['Pspin'] = np.sqrt((2.6e52 * (aguilera['Mns'].value / 1.4) ** (3 / 2)) /
                                        (aguilera['ErotNS'].value * 1E51))

            # Plot data
            ax1.errorbar(xparam_med, yparam_med, xerr=[xparam_lo, xparam_up], yerr=[yparam_lo, yparam_up], fmt='o',
                         color=cb_g, markersize=10, alpha=alpha, markeredgecolor='k', zorder=500)
            ax1.errorbar(aguilera[param_x], aguilera[param_y], fmt='o', color='b', markersize=10,
                         alpha=alpha + 0.1, markeredgecolor='k', zorder=1000)
            ax1.errorbar(kumar[param_x], kumar[param_y], xerr=kumar[param_x + '_err'],
                         yerr=kumar[param_y + '_err'], fmt='o', color=cb_r, markersize=10,
                         alpha=alpha + 0.1, markeredgecolor='k', zorder=1100)
        else:
            print('Comparisons to other works only available for Pspin and mejecta.')

    # Plot the data normally
    elif param_z is None:
        ax1.errorbar(xparam_med, yparam_med, xerr=[xparam_lo, xparam_up], yerr=[yparam_lo, yparam_up], fmt='o',
                     color=cb_g, markersize=10, alpha=alpha, markeredgecolor='k', zorder=500)
    else:
        if zlog:
            colors = np.log10(params[f'{param_z}_med'])
        else:
            colors = params[f'{param_z}_med']

        # Color mapping
        norm = plt.Normalize(vmin=min(colors), vmax=max(colors))
        cmap = plt.cm.plasma_r
        scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar_map.set_array(colors)

        for i in range(len(xparam_med)):
            ax1.errorbar(xparam_med[i], yparam_med[i], xerr=[[xparam_lo[i]], [xparam_up[i]]],
                         yerr=[[yparam_lo[i]], [yparam_up[i]]], fmt='o', color=scalar_map.to_rgba(colors[i]),
                         markersize=10, alpha=0.4, markeredgecolor='k', zorder=300)

    # Plot optional functions
    if plot_tmag:
        if (param_x == 'Pspin') & (param_y == 'log(Bfield)'):

            # Mean ejecta mass and kinetic energy
            Mejecta = np.median(params['mejecta_med'])
            KE = 10 ** np.median(params['log(kenergy)_med'])

            # Plot constant diffusion timescales
            xarray = np.linspace(xmin, xmax, 100)
            yarray1 = np.log10(1E14 * np.sqrt(((xarray) ** 2) /
                               (0.1 * 20.7) * (Mejecta) ** (-3/4) * (KE / 1E51) ** (1/4)))
            yarray2 = np.log10(1E14 * np.sqrt(((xarray) ** 2) /
                               (1.0 * 20.7) * (Mejecta) ** (-3/4) * (KE / 1E51) ** (1/4)))
            yarray3 = np.log10(1E14 * np.sqrt(((xarray) ** 2) /
                               (10. * 20.7) * (Mejecta) ** (-3/4) * (KE / 1E51) ** (1/4)))
            ax1.plot(xarray, yarray1, color='k', linestyle='-', label=r't$_{\rm mag}$ = 0.1 t$_{\rm diff}$', zorder=900)
            ax1.plot(xarray, yarray2, color='k', linestyle='--', label=r't$_{\rm mag}$ = t$_{\rm diff}$', zorder=900)
            ax1.plot(xarray, yarray3, color='k', linestyle=':', label=r't$_{\rm mag}$ = 10 t$_{\rm diff}$', zorder=900)
            leg = ax1.legend(loc='lower right')
            leg.set_zorder(1000)
        else:
            print('Magnetar diffusion timescale plot only available for Pspin and log(Bfield).')

    # Plot magnetar energy
    if plot_menergy:
        if (param_x == 'Pspin') & (param_y == 'log(kenergy)'):

            # Mean neutron star mass
            M_NS = 1.7

            # Plot constant magnetar energy
            xarray = np.linspace(0.01, 20, 300)
            yarray = np.log10(2.6E52 * (M_NS / 1.4) ** (3. / 2.) * xarray ** (-2))
            ax1.plot(xarray, yarray, color='k', label=r'$M_{\rm NS} = {%s} M_\odot$' % M_NS, zorder=900)
            leg = ax1.legend(loc='upper right')
            leg.set_zorder(1000)
        else:
            print('Magnetar energy plot only available for Pspin and log(kenergy).')

    # Plot the mean values of the parameter
    if plot_mean:
        # Get the mean and 1-sigma range
        xparam_minus, xparam_cen, xparam_plus = np.percentile(xparam_med, [15.87, 50, 84.13], axis=0)
        yparam_minus, yparam_cen, yparam_plus = np.percentile(yparam_med, [15.87, 50, 84.13], axis=0)

        # Plot lines
        ax1.axvline(x=xparam_minus, color='k', linestyle='--', linewidth=0.5)
        ax1.axvline(x=xparam_cen, color='k', linestyle='-', linewidth=1)
        ax1.axvline(x=xparam_plus, color='k', linestyle='--', linewidth=0.5)
        ax1.axhline(y=yparam_minus, color='k', linestyle='--', linewidth=0.5)
        ax1.axhline(y=yparam_cen, color='k', linestyle='-', linewidth=1)
        ax1.axhline(y=yparam_plus, color='k', linestyle='--', linewidth=0.5)

    # Plot the 1-to-1 correlation line
    if plot_1_1:
        tot_min = np.min(np.append(xparam_med, yparam_med))
        tot_max = np.max(np.append(xparam_med, yparam_med))
        plt.plot([tot_min, tot_max], [tot_min, tot_max], color='magenta',
                 linestyle='--', linewidth=1, zorder=900)
        plt.annotate('1:1', xy=(tot_max * 0.55, tot_max * 0.45), color='magenta')

    # Plot lines of constant nickel fraction
    if plot_fnickel:
        if (param_x == 'mejecta') & (param_y == 'mnickel'):

            # Plot lines of constant nickel fraction
            xarray = np.linspace(0.1, 120)
            frac1 = 0.5 * xarray
            frac2 = 0.1 * xarray
            frac3 = 0.01 * xarray
            ax1.plot(xarray, frac1, color='k', linewidth=1, linestyle='--', zorder=900)
            ax1.plot(xarray, frac2, color='k', linewidth=1, linestyle='--', zorder=900)
            ax1.plot(xarray, frac3, color='k', linewidth=1, linestyle='--', zorder=900)
            ax1.annotate(r'f$_{\rm Ni} = 0.5$', xy=(0.60, 0.425), rotation=21, zorder=1e4,
                         path_effects=[pe.withStroke(linewidth=1, foreground="white")])
            ax1.annotate(r'f$_{\rm Ni} = 0.1$', xy=(0.60, 0.080), rotation=21, zorder=1e4,
                         path_effects=[pe.withStroke(linewidth=1, foreground="white")])
            ax1.annotate(r'f$_{\rm Ni} = 0.01$', xy=(0.60, 0.008), rotation=21, zorder=1e4,
                         path_effects=[pe.withStroke(linewidth=1, foreground="white")])
        else:
            print('Lines of constant nickel fraction only available for mejecta and mnickel.')

    # Set Plot limits
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    # Set plot log scale
    if xlog:
        ax1.set_xscale('log')
    else:
        ax1.set_xscale('linear')
    if ylog:
        ax1.set_yscale('log')
    else:
        ax1.set_yscale('linear')
    # Set plot names
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)

    if param_z is None:
        ax2 = plt.subplot(gs[3])
        ax3 = plt.subplot(gs[0])
        ax4 = plt.subplot(gs[1])
        use_color = cb_g
    else:
        ax2 = plt.subplot(gs[1, 1])
        ax3 = plt.subplot(gs[0, 0])
        ax4 = plt.subplot(gs[:, 2])
        use_color = scalar_map.to_rgba(max(colors))

    # Plot histograms
    # Y-axis histogram
    if ylog:
        ax2.hist(np.log10(yparam_med), density=False, orientation="horizontal", color=use_color,
                 bins=bins, range=(np.log10(ymin_h), np.log10(ymax_h)), alpha=0.5)
        ax2.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
                        labeltop=False, labelright=False, labelbottom=False)
        ax2.set_ylim(np.log10(ymin), np.log10(ymax))
        if include_frac:
            ax2.hist(np.log10(yparam_med[blue]), density=False, orientation="horizontal", color='b',
                     bins=bins, range=(np.log10(ymin_h), np.log10(ymax_h)), alpha=0.5)
            ax2.hist(np.log10(yparam_med[red]), density=False, orientation="horizontal", color=cb_r,
                     bins=bins, range=(np.log10(ymin_h), np.log10(ymax_h)), alpha=0.5)
        if include_others:
            ax2.hist(np.log10(aguilera[param_y]), density=False, orientation="horizontal", color='b',
                     bins=bins, range=(np.log10(ymin_h), np.log10(ymax_h)), alpha=0.5)
            ax2.hist(np.log10(kumar[param_y]), density=False, orientation="horizontal", color=cb_r,
                     bins=bins, range=(np.log10(ymin_h), np.log10(ymax_h)), alpha=0.5)
    else:
        ax2.hist(yparam_med, density=False, orientation="horizontal", color=use_color,
                 bins=bins, range=(ymin_h, ymax_h), alpha=0.5)
        ax2.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
                        labeltop=False, labelright=False, labelbottom=False)
        ax2.set_ylim(ymin, ymax)
        if include_frac:
            ax2.hist(yparam_med[blue], density=False, orientation="horizontal", color='b',
                     bins=bins, range=(ymin_h, ymax_h), alpha=0.5)
            ax2.hist(yparam_med[red], density=False, orientation="horizontal", color=cb_r,
                     bins=bins, range=(ymin_h, ymax_h), alpha=0.5)
        if include_others:
            ax2.hist(aguilera[param_y], density=False, orientation="horizontal", color='b',
                     bins=bins, range=(ymin_h, ymax_h), alpha=0.5)
            ax2.hist(kumar[param_y], density=False, orientation="horizontal", color=cb_r,
                     bins=bins, range=(ymin_h, ymax_h), alpha=0.5)
    # X-axis histogram
    if xlog:
        ax3.hist(np.log10(xparam_med), density=False, color=use_color, bins=bins,
                 range=(np.log10(xmin_h), np.log10(xmax_h)), alpha=0.5)
        ax3.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
                        labeltop=False, labelright=False, labelbottom=False)
        ax3.set_xlim(np.log10(xmin), np.log10(xmax))
        if include_frac:
            ax3.hist(np.log10(xparam_med[blue]), density=False, color='b', bins=bins,
                     range=(np.log10(xmin_h), np.log10(xmax_h)), alpha=0.5)
            ax3.hist(np.log10(xparam_med[red]), density=False, color=cb_r, bins=bins,
                     range=(np.log10(xmin_h), np.log10(xmax_h)), alpha=0.5)
        if include_others:
            ax3.hist(np.log10(aguilera[param_x]), density=False, color='b', bins=bins,
                     range=(np.log10(xmin_h), np.log10(xmax_h)), alpha=0.5)
            ax3.hist(np.log10(kumar[param_x]), density=False, color=cb_r, bins=bins,
                     range=(np.log10(xmin_h), np.log10(xmax_h)), alpha=0.5)
    else:
        ax3.hist(xparam_med, density=False, color=use_color, bins=bins,
                 range=(xmin_h, xmax_h), alpha=0.5)
        ax3.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
                        labeltop=False, labelright=False, labelbottom=False)
        ax3.set_xlim(xmin, xmax)
        if include_frac:
            ax3.hist(xparam_med[blue], density=False, color='b', bins=bins,
                     range=(xmin_h, xmax_h), alpha=0.5)
            ax3.hist(xparam_med[red], density=False, color=cb_r, bins=bins,
                     range=(xmin_h, xmax_h), alpha=0.5)
        if include_others:
            ax3.hist(aguilera[param_x], density=False, color='b', bins=bins,
                     range=(xmin_h, xmax_h), alpha=0.5)
            ax3.hist(kumar[param_x], density=False, color=cb_r, bins=bins,
                     range=(xmin_h, xmax_h), alpha=0.5)

    # Plot legend if necessary
    if include_frac:
        ax4.errorbar([], [], fmt='o', color=cb_g, markersize=10, alpha=1.0, markeredgecolor='k', label='All SLSNe')
        ax4.errorbar([], [], fmt='o', color='b', markersize=10, alpha=1.0, markeredgecolor='k',
                     label=r'$f_{\rm mag, max} < 0.90$')
        ax4.errorbar([], [], fmt='o', color=cb_r, markersize=10, alpha=1.0, markeredgecolor='k',
                     label=r'$f_{\rm mag, max} < 0.50$')
        ax4.legend(loc='best', fontsize=11, frameon=False)
    if include_others:
        ax4.errorbar([], [], fmt='o', color=cb_g, markersize=10, alpha=1.0, markeredgecolor='k', label='All SLSNe')
        ax4.errorbar([], [], fmt='o', color='b', markersize=10, alpha=1.0, markeredgecolor='k',
                     label='Aguilera-Dena 2020')
        ax4.errorbar([], [], fmt='o', color=cb_r, markersize=10, alpha=1.0, markeredgecolor='k',
                     label='GRB SNe (Kumar 2024)')
        ax4.legend(loc='best', fontsize=11, frameon=False)
    if individual_name is not None:
        ax4.errorbar([], [], fmt='o', color='b', markersize=10, alpha=1.0, markeredgecolor='k',
                     label=individual_name)
        ax4.legend(loc='best', fontsize=11, frameon=False)

    # Clear the upper corner
    if param_z is not None:
        cbar = plt.colorbar(scalar_map, cax=ax4, orientation='vertical')
        if zlog:
            cbar.set_label(f'log({zlabel})')
        else:
            cbar.set_label(zlabel)
        if param_z in ['Peak_mag', 'r_peak']:
            cbar.ax.invert_yaxis()
        ax5 = plt.subplot(gs[0, 1])
        ax5.spines['left'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax5.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
                        labeltop=False, labelright=False, labelbottom=False)
    else:
        ax4.spines['left'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax4.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
                        labeltop=False, labelright=False, labelbottom=False)

    # Save figure
    if param_z is None:
        plot_name = f'{param_x}_{param_y}.pdf'
    else:
        plot_name = f'{param_x}_{param_y}_{param_z}.pdf'
    plot_dir = os.path.join(output_dir, plot_name)
    plt.savefig(plot_dir, bbox_inches='tight')
    plt.clf()
    plt.close('all')


def z_curve(mag, redshift):
    """
    This function calculates the limiting absolute magnitude
    as a function of redshift for a giving apparent magnitude
    limit of a survey.

    Parameters
    ----------
    mag : float
        The limiting magnitude of the survey.
    redshift : np.array
        A redshift array.

    Returns
    -------
    M: float
        The absolute magnitude of the source.
    """

    # Luminosity Distance in parsecs
    DL = cosmo.luminosity_distance(z=redshift).to(u.pc).value

    # Distance Modulus
    DM = 5 * np.log10(DL / 10)

    # Absolute Magnitude
    M = mag - DM + 2.5 * np.log10(1 + redshift)

    return M


def magnitude_plot(output_dir='.', remove_bronze=True):
    """
    This function creates a plot of the limiting magnitude
    as a function of redshift for a survey.

    Parameters
    ----------
    output_dir : str, default='.'
        The directory where the plot will be saved.
    remove_bronze : bool, default=True
        Whether to remove the bronze SLSNe from the plot.
    """

    # Import parameters data
    params = get_params()

    # Remove Bronze objects by default
    if remove_bronze:
        data_table = get_data_table()
        good_names = data_table['Name'][data_table['Quality'] != 'Bronze']
        use_names = [i in good_names for i in params['name']]
        params = params[use_names]

    # Import FLEET data
    data_table = get_data_table()

    # Import parameters data
    params = get_params()

    if remove_bronze:
        good_names = data_table['Name'][data_table['Quality'] != 'Bronze']
        use_names = [i in good_names for i in params['name']]
        params = params[use_names]

    # Get redshifts and surveys
    redshifts = params['redshift_med']
    surveys = np.array([data_table['Survey'][data_table['Name'] == i][0] for i in params['name']])
    ZTF = surveys == 'ZTF'
    DES = surveys == 'DES'
    PS1 = surveys == 'PS1'

    # Array for plotting constant lines
    z_array = np.linspace(0.01, 3, 100)
    # For ZTF
    shallow = 20.5
    Mag_shallow = z_curve(shallow, z_array)
    # For DES and MDS
    deep = 23.5
    Mag_deep = z_curve(deep, z_array)

    # Set up lot
    f = plt.figure()
    f.set_size_inches(22, 5)
    plt.subplots_adjust(hspace=0.0, wspace=0.0)

    gs = gridspec.GridSpec(2, 8,
                           width_ratios=[3, 0.7, 1.1, 3, 0.7, 1.1, 3, 0.7],
                           height_ratios=[1.1, 3]
                           )
    xmin, xmax = 0, 2.1
    ymin, ymax = -23.3, -18.8
    ymin_E, ymax_E = 49.5, 52.0

    ax1 = plt.subplot(gs[1, 0])
    ax1.errorbar(redshifts, params['r_peak_med'], yerr=[params['r_peak_lo'], params['r_peak_up']],
                 color=cb_g, alpha=0.4, fmt='o')
    ax1.errorbar(redshifts[ZTF], params['r_peak_med'][ZTF], color=cb_r, alpha=0.4, fmt='*', label='ZTF')
    ax1.errorbar(redshifts[DES], params['r_peak_med'][DES], color='b', alpha=0.4, fmt='*', label='DES')
    ax1.errorbar(redshifts[PS1], params['r_peak_med'][PS1], color='magenta', alpha=0.4, fmt='*', label='PS1')

    ax1.plot(z_array, Mag_shallow, color=cb_r, linestyle='--', linewidth=1)
    ax1.annotate(r'$m_r = %s$' % shallow, xy=(0.45, -22.8), color=cb_r)

    ax1.plot(z_array, Mag_deep, color='k', linestyle='--', linewidth=1)
    ax1.annotate(r'$m_r = %s$' % deep, xy=(0.85, -20.55), color='k')

    ax1.legend(loc='lower right')
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymax, ymin)
    ax1.set_xlabel('Redshift')
    ax1.set_ylabel(r'Peak Absolute Magnitude [$r$]')

    ax1b = plt.subplot(gs[1, 3])
    ax1b.errorbar(redshifts, params['log(E_rad)_med'], yerr=[params['log(E_rad)_lo'], params['log(E_rad)_up']],
                  color=cb_g, alpha=0.4, fmt='o')
    ax1b.errorbar(redshifts[ZTF], params['log(E_rad)_med'][ZTF], color=cb_r, alpha=0.4, fmt='*', label='ZTF')
    ax1b.errorbar(redshifts[DES], params['log(E_rad)_med'][DES], color='b', alpha=0.4, fmt='*', label='DES')
    ax1b.errorbar(redshifts[PS1], params['log(E_rad)_med'][PS1], color='magenta', alpha=0.4, fmt='*', label='PS1')

    ax1b.legend(loc='lower right')
    ax1b.set_xlim(xmin, xmax)
    ax1b.set_ylim(ymin_E, ymax_E)
    ax1b.set_xlabel('Redshift')
    ax1b.set_ylabel(r'$\log(E_{\rm rad}\ /\ {\rm erg})$')

    ax2b = plt.subplot(gs[1, 4])
    ax2b.hist(params['log(E_rad)_med'], density=False, orientation="horizontal", color=cb_g, bins=30,
              range=(ymin_E, ymax_E), alpha=0.75)
    ax2b.tick_params(axis='both', bottom=False, labelbottom=False)
    ax2b.set_ylim(ymin_E, ymax_E)
    ax2b.yaxis.set_label_position("right")
    ax2b.yaxis.tick_right()

    ax3b = plt.subplot(gs[0, 3])
    ax3b.hist(redshifts, density=False, orientation="vertical", color=cb_g, bins=40, range=(xmin, xmax), alpha=0.75)
    ax3b.tick_params(axis='both', left=False, labelleft=False)
    ax3b.set_xlim(xmin, xmax)
    ax3b.xaxis.set_label_position("top")
    ax3b.xaxis.tick_top()

    ax2 = plt.subplot(gs[1, 1])
    ax2.hist(params['r_peak_med'], density=False, orientation="horizontal", color=cb_g, bins=30,
             range=(ymin, ymax), alpha=0.75)
    ax2.tick_params(axis='both', bottom=False, labelbottom=False)
    ax2.set_ylim(ymax, ymin)
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()

    ax3 = plt.subplot(gs[0, 0])
    ax3.hist(redshifts, density=False, orientation="vertical", color=cb_g, bins=40, range=(xmin, xmax), alpha=0.75)
    ax3.tick_params(axis='both', left=False, labelleft=False)
    ax3.set_xlim(xmin, xmax)
    ax3.xaxis.set_label_position("top")
    ax3.xaxis.tick_top()

    # Save plot
    plot_name = 'absmag_redshift.pdf'
    plot_dir = os.path.join(output_dir, plot_name)
    plt.savefig(plot_dir, bbox_inches='tight')
    plt.clf()
    plt.close('all')


def broken_law(x, x_break, m1, m2, b):
    """
    Function to fit a broken power law to a distribution

    Parameters
    ----------
    x : np.array
        The x values to fit.
    x_break : float
        The break point of the power law.
    m1 : float
        The slope of the first power law.
    m2 : float
        The slope of the second power law.
    b : float
        The y-intercept of the power law.

    Returns
    -------
    out : np.array
        The y values of the power law.
    """
    one = m1 * (x - x_break) + b
    two = m2 * (x - x_break) + b
    out = one
    out[x > x_break] = two[x > x_break]
    return out


def samples(x, m, m_err, b, b_err, N=1000):
    """
    Create samples of a linear model for plotting.

    Parameters
    ----------
    x : np.array
        The x values to model.
    m : float
        The slope of the model.
    m_err : float
        The error on the slope.
    b : float
        The y-intercept of the model.
    b_err : float
        The error on the y-intercept.
    N : int, default=1000
        The number of samples to create.

    Returns
    -------
    dn : np.array
        The lower bound of the model.
    cen : np.array
        The central value of the model.
    up : np.array
        The upper bound of the model
    """

    # Samples of slopes and intercepts
    m_array = np.random.normal(m, m_err, N)
    b_array = np.random.normal(b, b_err, N)

    # Calculate model
    y = np.array([m * x + b for m, b in zip(m_array, b_array)])

    # Get mean
    dn, cen, up = np.percentile(y, [15.87, 50, 84.13], axis=0)

    return dn, cen, up


def plot_mass_distribution(M_max=100, bins=50, n_walkers=150, use_Blanchard=False, remove_bronze=True, output_dir='.'):
    """
    This function creates a plot of the progenitor mass distribution of SLSNe,
    and fits the data with both a single and a broken power law.

    Parameters
    ----------
    M_max : float, default=100
        The maximum mass to plot.
    bins : int, default=50
        The number of bins to use in the histogram.
    n_walkers : int, default=150
        The number of walkers used in the MCMC fit.
    use_Blanchard : bool, default=False
        Whether to use the Blanchard et al. (2020) sample.
    remove_bronze : bool, default=True
        Whether to remove the bronze SLSNe from the plot.
    output_dir : str, default='.'
        The directory where the plot will be saved.
    """

    # Import SLSN data
    data_table = get_data_table()

    # Select which objects to use
    if use_Blanchard:
        use_names = ['DES14X3taz', 'iPTF13ajg', 'iPTF13dcc', 'iPTF13ehe', '2016wi', 'iPTF16bad', 'LSQ12dlf',
                     'LSQ14bdq', 'LSQ14mo', 'PS110ahf', 'PS110awh', 'PS110bzj', 'PS110ky', 'PS110pm', 'PS111afv',
                     'PS111aib', 'PS111ap', 'PS111bam', 'PS111bdn', 'PS111tt', 'PS112bmy', 'PS112bqf', 'PS113gt',
                     'PS113or', 'PS114bj', '2016ard', '2016inl', '2017dwh', 'PTF09atu', 'PTF09cnd', 'PTF10aagc',
                     'PTF10bfz', 'PTF10nmn', 'PTF10uhf', 'PTF10vqv', 'PTF12dam', 'PTF12gty', 'PTF12hni', 'PTF12mxx',
                     'iPTF13bjz', 'iPTF13cjq', 'SCP06F6', '2005ap', '2006oz', '2007bi', '2009cb', '2009jh', '2010gx',
                     '2010hy', '2010md', '2011ke', '2011kf', '2011kg', '2012il', '2013dg', '2013hy', '2015bn',
                     '2016eay', 'SNLS06D4eu', 'SNLS07D2bv', 'SSS120810']
        # Crop data_table to only include objects in use_names
        data_table = data_table[[i in use_names for i in data_table['Name']]]
    else:
        if remove_bronze:
            data_table = data_table[data_table['Quality'] != 'Bronze']

    # Empty arrays to store the total mass
    N_SLSN = len(data_table)
    all_SLSN = np.array([])

    for i in range(N_SLSN):
        object_name = data_table[i]['Name']
        print(i + 1, '/', N_SLSN, object_name)

        # Get object data
        params = get_params(object_name)

        # Get total progenitor mass
        total_mass = np.array(params['mejecta']) + np.array(params['Mns'])

        # Increase counter
        all_SLSN = np.append(all_SLSN, total_mass)

    # Generate mass histogram
    hist_SLSN_in, bins_SLSN = np.histogram(all_SLSN, range=(1, M_max),
                                           bins=np.logspace(np.log10(1), np.log10(M_max), bins),
                                           density=True)
    hist_SLSN_in = np.append(0, hist_SLSN_in)
    hist_SLSN = hist_SLSN_in * n_walkers

    # Find peak and maximum of the distribution
    min_SLSN = bins_SLSN[np.argmax(hist_SLSN)]
    max_SLSN = bins_SLSN[25:][np.argmin(np.abs((hist_SLSN[25:] - 1)))]

    # Select SLSNe that lie within the acceptable range
    good_SLSN = (bins_SLSN > min_SLSN) & (bins_SLSN < max_SLSN)
    x_data = np.log10(bins_SLSN[good_SLSN])
    y_data = np.log10(hist_SLSN[good_SLSN])
    bin_half = np.diff(x_data)[0] / 2

    # Fit a line to these
    pars_SLSN, cov_SLSN = np.polyfit(x_data - bin_half, y_data, 1, cov=True)
    m_SLSN, b_SLSN = pars_SLSN
    Err_m_SLSN = np.sqrt(np.diag(cov_SLSN))[0]
    Err_b_SLSN = np.sqrt(np.diag(cov_SLSN))[1]

    # Fit the SLSN distribution with two power laws
    initial_guess = [np.log10(10), -0.84, -1.47, b_SLSN]
    if use_Blanchard:
        popt, pcov = curve_fit(broken_law, x_data - bin_half, y_data, p0=initial_guess,
                               bounds=([0.8, -1, -2, 0], [1.4, -0.3, -1, 10]))
    else:
        popt, pcov = curve_fit(broken_law, x_data - bin_half, y_data, p0=initial_guess,
                               bounds=([0.8, -1, -2, 0], [1.4, 0, -1, 10]))

    # Calculate uncertainties
    perr = np.sqrt(np.diag(pcov))

    # Extract the slope parameters and their uncertainties
    x_break, x_break_err = popt[0], perr[0]
    m1, m1_err = popt[1], perr[1]
    m2, m2_err = popt[2], perr[2]
    b, b_err = popt[3], perr[3]

    # Create samples for plotting
    xarray = np.linspace(0.0, M_max, 100)

    # For a single powerlaw
    dn, cen, up = samples(xarray, m_SLSN, Err_m_SLSN, b_SLSN, Err_b_SLSN)

    # And for a broken powerlaw
    x_break_samples = np.random.normal(x_break, x_break_err, 1000)
    m1_samples = np.random.normal(m1, m1_err, 1000)
    m2_samples = np.random.normal(m2, m2_err, 1000)
    b_samples = np.random.normal(b, b_err, 1000)
    beta, beta_err = np.mean(10 ** x_break_samples), np.std(10 ** x_break_samples)
    broken_samples = np.array([broken_law(np.log10(xarray), o, p, q, r) for o, p, q, r in
                               zip(x_break_samples, m1_samples, m2_samples, b_samples)])
    brokn_dn, brokn_cen, brokn_up = np.percentile(broken_samples, [15.87, 50, 84.13], axis=0)

    # Set up figure
    fig, ax1 = plt.subplots()
    ticks = np.array([1, 2, 5, 10, 25, 100])

    # Plot Best Fit
    ax1.plot(10 ** xarray, 10 ** cen, color='k', linestyle='-', linewidth=1,
             label=r'$\alpha = %s \pm %s$' % (np.round(m_SLSN, 2), np.round(Err_m_SLSN, 2)))
    ax1.fill_between(10 ** xarray, 10 ** dn, 10 ** up, color='k', linewidth=0, alpha=0.2)

    ax1.plot(xarray, 10 ** brokn_cen, color='b', linestyle='-', linewidth=1,
             label=(r'$\alpha_1 = %s \pm %s$' % (np.round(m1, 2), np.round(m1_err, 2)) + '\n' +
                    r'$\alpha_2 = %s \pm %s$' % (np.round(m2, 2), np.round(m2_err, 2)) + '\n' +
                    r'M$_{\rm B} = %s \pm %s$ M$_\odot$' % (np.round(beta, 1), np.round(beta_err, 1))), zorder=10)
    ax1.fill_between(xarray, 10 ** brokn_dn, 10 ** brokn_up, color='b', linewidth=0, alpha=0.2)

    # Plot N = 1 line
    ax1.annotate('N = 1', xy=(5, 1.1))
    ax1.axhline(y=1, color='k', zorder=101)
    ax1.fill_between([1, M_max], y1=0, y2=1, color='white', alpha=0.6, zorder=100)

    # Plot actual histogram data
    ax1.step(bins_SLSN, hist_SLSN, alpha=1.0, color=cb_g, label='SLSNe', zorder=20)
    ax1.legend(loc='upper right', frameon=True)
    ax1.set_xlabel(r'M$_{\rm NS}$ + M$_{\rm ej}$ [M$_\odot$]')
    ax1.set_xlim(1, M_max)
    ax1.set_ylim(0.1, 60)
    ax1.set_ylabel('N SNe')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(ticks)

    # Slope and intercept of ZAMS relation from
    # Sukhbold et al. 2018 2018ApJ...860...93S
    m_suk, b_suk = 2.278, 8.571
    ax2 = ax1.twiny()
    ax2.set_xscale('log')
    ax2.set_xlabel(r'M$_{\rm ZAMS}$ [M$_\odot$]')
    ax2.set_xlim(1, M_max)
    ax2.set_xticks(ax1.get_xticks()[1:-1])
    ax2.set_xticklabels(np.round(m_suk * ticks + b_suk, 1)[1:-1])

    # Save plot
    if use_Blanchard:
        plot_name = f'distribution_mass_{M_max}_Blanchard.pdf'
    else:
        plot_name = f'distribution_mass_{M_max}.pdf'
    plot_dir = os.path.join(output_dir, plot_name)
    plt.savefig(plot_dir, bbox_inches='tight')
    plt.clf()
    plt.close('all')
