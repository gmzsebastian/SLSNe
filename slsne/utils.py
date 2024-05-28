"""
This file contains general utilities to process the data from the SLSNe package.
Written by Sebastian Gomez, 2024.
"""

# Import necessary packages
import os
import numpy as np
from astropy import table

# Get directory with reference data
current_file_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_file_dir, 'ref_data')


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
    filters: astropy.table.table.Table
        Astropy table with filters, their telescope, instrument, system (AB or Vega),
        their central wavelength, and zeropoint
    """

    # Import filter parameters
    generics = table.Table.read(f'{data_dir}/generic_reference.txt', format='ascii')
    filters_in = table.Table.read(f'{data_dir}/filter_reference.txt', format='ascii')

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
