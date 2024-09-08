.. _reference:

Reference Data
==============

The ``slsne`` package is heavily dependent on reference data used to create all the plots and parameter tables.
Here, we describe some of the reference data used, its format, and how to use it.

Filter Parameters
-----------------

The central wavelength in angstroms and the zeropoint in Janskys for each filter are required quantities. These
values are stored in two files in the ``slsne/ref_data`` directory. The ``filter_reference.txt`` file contains the
central wavelength and zeropoint for a wide range of telescopes and instruments. Some filter listed here have values
set to ``Generic``. These values are replaced with the values stored in ``generic_reference.txt``. The function that
does this operation is ``slsne.utils.define_filters``.

.. code-block:: python

    from slsne.utils import define_filters
    filters = define_filters()  

The ``filters`` table has the following columns:


.. code-block:: text

    Filter Telescope Instrument System Cenwave Zeropoint
     str6    str12     str11     str4  float64  float64 
    ------ --------- ---------- ------ ------- ---------
         B        --         --   Vega 4369.53   4323.91
         B        LT         --     AB 4369.53    3631.0
         B        LT         --   Vega 4369.53   4323.91
         B        LT     RATCam   Vega  4332.7   4210.22

Once the big table ``filters`` is created, it can be used to access the central wavelength and zeropoint by the
user, or using some built in functions. For example, the ``get_cenwave`` function returns the central wavelength
and optionally the zeropoint of a given filter. By default this function is very verbose and will print out details
about the choices made for the wavelength and zeropoint.

.. code-block:: python

    from slsne.utils import get_cenwave

    cenwave, zeropoint = get_cenwave('g', system='AB', return_zp=True, verbose=True)

Alternatively, the ``quick_cenwave_zeropoint`` function can be used to quickly get the central wavelength and zeropoint
of a table of photometry that has already been processed and verified to be clean.

.. code-block:: python

    from slsne.utils import quick_cenwave_zeropoint
    from astropy.table import Table

    # Create a mock photometry table
    names = ['Telescope', 'Instrument', 'System', 'Filter']
    data = [['Swift', 'P48', 'Generic'],
            ['UVOT', 'ZTF', 'Generic'],
            ['Vega', 'AB', 'AB'],
            ['UVW1', 'g', 'r']]
    phot = Table(data, names=names)

    # Get the central wavelength and zeropoints of all filters in the table.
    cenwaves, zeropoints = quick_cenwave_zeropoint(phot)

If the ``quick_cenwave_zeropoint`` function fails, it is likely that the filter names in the photometry table are not
correct or missing from the database. In this case, the ``check_filters`` function can be used to identify which
filter is causing the problem.

.. code-block:: python

    from slsne.utils import check_filters
    check_filters(phot)


Supernova Parameters
--------------------

Each supernova in the database has a set of parameters that are either produced directly from ``MOSFiT`` models, derived from these
models, or measured from observational features. Here we list each parameter, its label, and a description of what it is.

.. list-table::
    :widths: 20 20 60
    :header-rows: 1

    * - Parameter Name
      - Symbol [Units]
      - Description
    * - redshift         
      - .. math:: z                               
      - Redshift of the supernova.
    * - texplosion       
      - .. math:: t_{\rm exp}\ [{\rm Days}]
      - Explosion time in observer-frame days relative to first data point.
    * - fnickel          
      - .. math:: f_{\rm Ni}                      
      - Fraction of the ejecta mass that is Nickel.
    * - Pspin            
      - .. math:: P_{\rm spin}\ [{\rm ms}]               
      - Spin period of the magnetar.
    * - log(Bfield)      
      - .. math:: \log(B_{\perp} / G)              
      - Magnetic field of the magnetar.
    * - Mns              
      - .. math:: M_{\rm NS}\ [{\rm M}_\odot]     
      - Mass of the neutron star magnetar.
    * - thetaPB          
      - .. math:: \theta_{\rm PB}\ [{\rm rad}]           
      - Angle of the dipole moment.
    * - mejecta          
      - .. math:: M_{\rm ej}\ [{\rm M}_\odot]     
      - Ejecta mass.
    * - kappa            
      - .. math:: \kappa\ [cm^2g^{-1}]         
      - Optical opacity.
    * - kappagamma       
      - .. math:: \kappa_\gamma\ [cm^2g^{-1}]  
      - Gamma-ray opacity.
    * - vejecta          
      - .. math:: V_{\rm ej}\ [1000 {\rm km} {\rm s}^{-1}]   
      - Velocity of the ejecta.
    * - temperature      
      - .. math:: T\ [K]                             
      - Minimum photosphere temperature floor.
    * - alpha            
      - .. math:: P_{\rm cutoff}                  
      - Slope of the wavelength suppression.
    * - cutoff_wavelength
      - .. math:: \lambda_{\rm cutoff}\ [\mathrm{\mathring{A}}]
      - Flux below this wavelength is suppressed.
    * - log(nhhost)      
      - .. math:: \log{(n_{\rm H,host})}\ [{\rm cm}^{-2}]
      - Column density in the host galaxy.
    * - A_V              
      - .. math:: A_{\rm V}\ [{\rm mag}]                 
      - Intrinsic host galaxy extinction in V-band.
    * - MJD0             
      - .. math:: {\rm MJD}_0                           
      - Explosion date in MJD.
    * - log(kenergy)     
      - .. math:: \log(E_K / {\rm erg})
      - Total kinetic energy of the SN equal to :math:`\frac{3}{10} M_{\rm ej} V_{\rm ej}^2`.
    * - mnickel          
      - .. math:: M_{\rm Ni}\ [{\rm M}_\odot]
      - Nickel mass.
    * - log(TSD)         
      - .. math:: \log(t_{\rm SD}\ /\ s)
      - Magnetar spin-down time.
    * - log(L0)          
      - .. math:: \log(L_0\ /\ {\rm erg\ s}^{-1}) 
      - Initial magnetar spin-down luminosity.
    * - Peak_mag         
      - .. math:: m_{\rm r}\ [{\rm mag}]
      - Peak observed r-band magnitude.
    * - Peak_MJD         
      - .. math:: {\rm MJD}_{\rm peak}                  
      - Date of peak in r-band.
    * - log(E_rad)       
      - .. math:: \log(E_{\rm rad}\ /\ {\rm erg})
      - Total radiated energy of the SN.
    * - log(Peak_lum)    
      - .. math:: \log(L_{\rm max}\ /\ {\rm erg\ s}^{-1}) 
      - Peak bolometric luminosity.
    * - Rise_Time        
      - .. math:: \tau_{\rm rise}\ [{\rm Days}]
      - Number of frame days from explosion to bolometirc peak.
    * - E_fold           
      - .. math:: \tau_{e}\ [{\rm Days}]
      - Time it takes the SN to decline by a factor of e.
    * - tau_1            
      - .. math:: \tau_{\rm 1}\ [{\rm Days}]
      - Time it takes the SN to decline by 1 magnitude.
    * - delta_m15        
      - .. math:: \Delta m_{15}\ [{\rm mag}]
      - Magnitudes by which the SN fades 15 days after maximum in B-band.
    * - r_peak           
      - .. math:: M_{\rm r, peak}\ [{\rm mag}]
      - Peak absolute r-band magnitude.
    * - frac             
      - .. math:: f_{\rm mag}
      - Fraction of the total luminosity due to the magnetar contribution.
    * - 1frac            
      - .. math:: 1-f_{\rm mag}
      - Fraction of the total luminosity due to radioactive decay.
    * - efficiency       
      - .. math:: \epsilon
      - Radiative efficiency, equal to :math:`{E_{\rm rad}} / {E_{\rm K}}`.

You can access these parameters using the ``get_params`` function.

.. code-block:: python

    from slsne.utils import get_params

    # Get all the parameters for the full sample of SNe
    params = get_params()

    # Get a specfic set of parameters
    params_select = get_params(param_names=["Pspin","mejecta"])

    # Get all parameters for a single SN
    params_2018lfe = get_params('2018lfe')