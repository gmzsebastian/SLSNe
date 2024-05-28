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
