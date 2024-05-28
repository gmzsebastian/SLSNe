from ..utils import define_filters
from astropy.table import Table
import os
import pytest


@pytest.fixture
def data_dir():
    # Get directory with reference data
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_file_dir, '..', 'ref_data')


def test_define_filters_first_row(data_dir):

    # Call the function
    result = define_filters(data_dir)

    # Check that the result is a Table
    assert isinstance(result, Table)

    # Read the input filters
    filters_in = Table.read(f'{data_dir}/filter_reference.txt', format='ascii')

    # Check that the ZTF filter is in the output table and is exactly 3
    ZTF = filters_in[filters_in['Instrument'] == 'ZTF']
    assert len(ZTF) == 3

    # Check that all the values for 'Instrument' in filters_in are in the resulting table
    for i in filters_in['Instrument']:
        assert i in result['Instrument']

    # Make sure there are no instances for the word 'Generic' in the resulting table
    assert 'Generic' not in list(result['Cenwave'])
    assert 'Generic' not in list(result['Zeropoint'])
