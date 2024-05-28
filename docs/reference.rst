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

One the big table of filters ``filters`` is created, it can be used to access the central wavelength and zeropoint
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
