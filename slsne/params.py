"""
This file contains general utilities to process the observational and
physical parameters of SLSNe.

Written by Sebastian Gomez, 2024.
"""

# Import necessary packages
from astropy import table
import os

# Get directory with reference data
current_file_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_file_dir, 'ref_data')


def get_params(param_names=None):
    """
    Get the parameters of the SLSN sample from the reference
    table.

    Parameters
    ----------
    param_names : list, optional
        List of parameter names to get from the reference table.

    Returns
    -------
    params : astropy.table.Table
        Table with the requested parameters.
    """

    # Read table
    params = table.Table.read(os.path.join(data_dir, 'all_parameters.txt'), format='ascii')

    # Make sure param_names is a list
    if type(param_names) is str:
        param_names = [param_names]

    # Get requested parameters
    if param_names is not None:
        format_names = [f"{prefix}_{param}" for param in param_names for prefix in ['cen', 'up', 'dn']]
        params = params[format_names]

    return params
