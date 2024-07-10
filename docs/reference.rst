.. _reference:

Reference Data
==============

The ``slsne`` package is heavily dependent on reference data used to create all the plots and parameter tables.
Here, we describe some of the reference data, its format, and how to use it.

Filter Parameters
-----------------

The central wavelength in angstroms and the zeropoint in Jansky for each filter are required quantities. These
values are stored in two files in the ``slsne/ref_data`` directory. The ``filter_reference.txt`` file contains the
central wavelength and zeropoint for a wide range of telescopes and instruments. Wherever the central
wavelength or zeropoint is called ``Generic`` in this file, this value is replaced with the values stored 
in ``generic_reference.txt``. The function that does this operation is ``slsne.utils.define_filters``.

.. code-block:: python

    from slsne.utils import define_filters
    filters = define_filters()  

Once the big table of filters ``filters`` is created, it can be used to access the central wavelength and zeropoint
using the other built in functions.

.. code-block:: python

    from slsne.utils import get_cenwave, quick_cenwave_zeropoint, check_filters
    from astropy.table import Table

    # Create a mock photometry table
    names = ['Telescope', 'Instrument', 'System', 'Filter']
    data = [['Swift', 'P48', 'Generic'],
            ['UVOT', 'ZTF', 'Generic'],
            ['Vega', 'AB', 'AB'],
            ['UVW1', 'g', 'r']]
    phot = Table(data, names=names)

    # get_cenwave returns the central wavelength,
    # and optionally the zeropoint of a given filter
    cenwave, zeropoint = get_cenwave('g', return_zp=True, verbose=True)

    # quick_cenwave_zeropoint returns the central wavelength
    # and zeropoint of a table of photometry that has already
    # been processed and verified to be clean.
    cenwaves, zeropoints = quick_cenwave_zeropoint(phot)

    # If the previous function failed, you can check which filter
    # in `phot`caused the problem by running check_filters
    check_filters(phot)

Supernova Parameters
--------------------

Each supernova has a set of parameters that are either produced directly MOSFiT models, derived from these models,
or measured from observational features. Here we list each parameter, its label, and a description of what it is.

.. list-table::
    :widths: 20 20 60
    :header-rows: 1

    * - Parameter Name
      - Symbol
      - Description
    * - redshift         
      - .. math:: z                               
      - Redshift of the SN.
    * - texplosion       
      - .. math:: t_{\rm exp} [Days]              
      - Explosion time relative to first data point.
    * - fnickel          
      - .. math:: f_{\rm Ni}                      
      - Fraction of the ejecta that is Nickel.
    * - Pspin            
      - .. math:: P_{\rm spin} [ms]               
      - Spin period of the magnetar.
    * - log(Bfield)      
      - .. math:: log(B_{\perp} / G)              
      - Magnetic field of the magnetar.
    * - Mns              
      - .. math:: M_{\rm NS}\ [{\rm M}_\odot]     
      - Mass of the neutron star magnetar.
    * - thetaPB          
      - .. math:: \theta_{\rm PB} [rad]           
      - Angle of the dipole moment.
    * - mejecta          
      - .. math:: M_{\rm ej}\ [{\rm M}_\odot]     
      - Ejecta mass.
    * - kappa            
      - .. math:: \kappa [cm^2g^{-1}]         
      - Optical opacity.
    * - kappagamma       
      - .. math:: \kappa_\gamma [cm^2g^{-1}]  
      - Gamma-ray opacity.
    * - vejecta          
      - .. math:: V_{\rm ej} [1000 km s^{-1}]   
      - Velocity of the ejecta.
    * - temperature      
      - .. math:: T [K]                             
      - Minimum photosphere temperature floor.
    * - alpha            
      - .. math:: P_{\rm cutoff}                  
      - Slope of the wavelength suppression.
    * - cutoff_wavelength
      - .. math:: \lambda_{\rm cutoff} [\mathrm{\mathring{A}}]
      - Flux below this wavelength is suppressed.
    * - log(nhhost)      
      - .. math:: \log{n_{\rm H,host}} [cm^{-2}]
      - Column density in the host galaxy.
    * - A_V              
      - .. math:: A_{\rm V} [mag]                 
      - Intrinsic host galaxy extinction in V-band.
    * - MJD0             
      - .. math:: MJD_0                           
      - Explosion date in MJD.
    * - log(kenergy)     
      - .. math:: log(E_K / erg)                  
      - Kinetic energy of the SN.
    * - mnickel          
      - .. math:: M_{\rm Ni}\ [{\rm M}_\odot]     
      - Nickel mass.
    * - log(TSD)         
      - .. math:: \log(t_{\rm SD}\ /\ s)          
      - Spin-down time.
    * - log(L0)          
      - .. math:: \log(L_0\ /\ {\rm erg\ s}^{-1}) 
      - Initial magnetar spin-down luminosity.
    * - Peak_mag         
      - .. math:: m_{\rm r} [mag]                 
      - Peak observed r-band magnitude.
    * - Peak_MJD         
      - .. math:: MJD_{\rm peak}                  
      - Date of peak in r-band.
    * - log(E_rad)       
      - .. math:: \log(E\ /\ erg)                 
      - Total radiated energy of the SN.
    * - log(Peak_lum)    
      - .. math:: \log(L_{\rm max}\ /\ {\rm erg\ s}^{-1}) 
      - The peak bolometric luminosity.
    * - Rise_Time        
      - .. math:: \tau_{\rm rise} [Days]          
      - Time from explosion to peak.
    * - E_fold           
      - .. math:: \tau_{e} [Days],               
      - Time it takes the SN to decline by a factor of e.
    * - tau_1            
      - .. math:: \tau_{\rm 1} [Days],           
      - Time it takes the SN to decline by 1 magnitude.
    * - delta_m15        
      - .. math:: \Delta m_{15} [mag]             
      - Magnitudes by which the SN fades 15 days after maximum in B-band.
    * - r_peak           
      - .. math:: M_{\rm r, peak} [mag]           
      - Peak absolute r-band magnitude.
    * - frac             
      - .. math:: f_{\rm mag}                     
      - Fraction of the total luminosity due to the magnetar contribution.
    * - 1frac            
      - .. math:: 1-f_{\rm mag}                   
      - Fraction of the total luminosity due to radioactive decay.
    * - efficiency       
      - .. math:: \epsilon                        
      - Radiative efficiency.

There's a number of way of accessing these parameters using the built in functions.

.. code-block:: python

    from slsne.utils import get_params

    # Get all the parameters for the full sample of SNe
    params = get_params()

    # Get a specfic set of parameters
    params_some = get_params(param_names=["Pspin","mejecta"])

    # Get all parameters for a single SN
    params_2018lfe = get_params('2018lfe')