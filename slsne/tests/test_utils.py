from ..utils import (define_filters, get_cenwave, quick_cenwave_zeropoint,
                     check_filters, plot_colors, read_phot)
from astropy.table import Table
import os
import pytest
import numpy as np


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


def test_get_cenwave_swift_filter():
    cenwave, zeropoint = get_cenwave('swift_UVW1', system='AB', return_zp=True, verbose=False)
    assert cenwave == 2681.67
    assert zeropoint == 3631.0


def test_get_cenwave_non_swift_filter():
    cenwave = get_cenwave('g', instrument='ZTF', return_zp=False, verbose=False)
    assert cenwave == 4746.48


def test_get_cenwave_generic_filter():
    cenwave, zeropoint = get_cenwave('z', return_zp=True, system='Vega', verbose=False)
    assert cenwave == 8922.78
    assert zeropoint == 2238.99


def test_get_cenwave_unknown_filter():
    with pytest.raises(KeyError):
        get_cenwave('potato', verbose=False)


def test_get_cenwave_unknown_system():
    with pytest.raises(KeyError):
        get_cenwave('g', system='penguin', return_zp=True, verbose=False)


# Create a mock phot table
names = ['Telescope', 'Instrument', 'System', 'Filter']
data = [['Swift', 'P48', 'Generic'],
        ['UVOT', 'ZTF', 'Generic'],
        ['Vega', 'AB', 'AB'],
        ['UVW1', 'g', 'r']]
phot = Table(data, names=names)


def test_quick_cenwave_zeropoint():
    cenwaves, zeropoints = quick_cenwave_zeropoint(phot)
    assert np.all(cenwaves == np.array([2681.67, 4746.48, 6141.12]))
    assert np.all(zeropoints == np.array([921.0, 3631.0, 3631.0]))


def test_quick_cenwave_zeropoint_missing_column():
    phot_missing_column = phot.copy()
    phot_missing_column.remove_column('Telescope')
    with pytest.raises(KeyError):
        quick_cenwave_zeropoint(phot_missing_column)


def test_check_filters(capsys):
    # This should pass without raising an exception
    # and without printing anything
    check_filters(phot)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_check_filters_missing_column():
    # Remove the 'Telescope' column
    phot_missing_column = phot.copy()
    phot_missing_column.remove_column('Telescope')
    with pytest.raises(KeyError):
        check_filters(phot_missing_column)


def test_check_filters_prints_something(capsys):
    # Add a row with a filter that is not in the reference data
    phot.add_row(['pink', 'penguin', 'AB', 'g'])
    check_filters(phot)
    captured = capsys.readouterr()
    assert captured.out != ""


def test_plot_colors_known_band():
    assert plot_colors('u') == 'navy'
    assert plot_colors('r') == 'r'
    assert plot_colors('i') == 'maroon'


def test_plot_colors_swift():
    color = plot_colors('UVW1')
    assert isinstance(color, np.ndarray)
    assert color.shape == (4,)
    assert all(isinstance(num, (int, float)) for num in color)


def test_plot_colors_unknown_band():
    assert plot_colors('potato') == 'k'


def test_read_phot(mocker):
    # Call the function with a test object name
    phot = read_phot('2018lfe')

    # Check that the returned table is correct
    assert isinstance(phot, Table)
    assert len(phot) >= 1
    assert 'MJD' in phot.colnames
