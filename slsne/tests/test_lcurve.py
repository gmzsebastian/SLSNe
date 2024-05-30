from ..lcurve import interpolate_1D, interpolate_2D, get_all_lcs, get_kcorr, fit_map
import numpy as np
from astropy.table import Table
import pytest


def test_interpolate_1D():
    time = np.array([1, 2, 3, 4, 5])
    flux = np.array([10, 20, 30, 40, 50])
    samples = np.array([1.5, 2.5, 3.5, 4.5])

    expected_output = np.array([15, 25, 35, 45])  # Expected interpolated flux values

    output = interpolate_1D(time, flux, samples)

    assert np.allclose(output, expected_output)

    # Test with samples outside the time range
    samples_outside = np.array([-1, 6])
    output = interpolate_1D(time, flux, samples_outside)
    assert np.isnan(output).all(), "The function should return np.nan for samples outside the time range"


def test_interpolate_2D():
    time = np.array([-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0])
    wavelength = np.array([2083.95, 3551.05, 4671.78, 8922.78, 12399.85, 16152.77])
    flux = np.array([[1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
                     [2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
                     [3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
                     [4.5, 5.5, 6.5, 7.5, 8.5, 9.5],
                     [5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
                     [6.5, 7.5, 8.5, 9.5, 10.5, 11.5],
                     [7.5, 8.5, 9.5, 10.5, 11.5, 12.5]])

    # Test that ValueError is raised when both out_wave and out_phase are None
    with pytest.raises(ValueError):
        interpolate_2D(time, wavelength, flux, None, None)

    # Test that the function correctly interpolates the flux values
    out_phase = np.array([-30.0, -5.0, 20.5])
    out_wave = np.array([3000.0, 2200.2, 9421.2])
    expected_flux = np.array([2.01545483, 4.03524949, 9.62960542])
    assert np.allclose(interpolate_2D(time, wavelength, flux, out_wave, out_phase), expected_flux)


def test_get_all_lcs_sigma():
    # Select only a few sne names
    names = ['2018lfe', '2017jan', '2018ffj', '2018hpq', '2018lzw', '2019aamv', '2019cwu']

    # Test with valid sigma values
    for sigma in [1, 2, 3]:
        try:
            get_all_lcs('g', names=names, sigma=sigma)
        except Exception as e:
            pytest.fail(f"Unexpected Exception with sigma={sigma}: {e}")

    # Test with invalid sigma value
    with pytest.raises(ValueError):
        get_all_lcs('g', names=names, sigma=4)

    # Test with non-integer sigma value
    with pytest.raises(ValueError):
        get_all_lcs('g', names=names, sigma=1.5)

    # Test with an invalid filter name
    with pytest.raises(ValueError):
        get_all_lcs('quokka')


def test_get_all_lcs_names():
    # Test with valid names
    try:
        get_all_lcs('r', names=['2018lfe'])
    except Exception as e:
        pytest.fail(f"Unexpected Exception with names=['2018lfe']: {e}")

    # Test with single name as string, making sure
    # the output in a numpy array of length equal to the
    # length of time_samples
    time_samples, mag_mean = get_all_lcs('r', names='2018lfe', time_samples=200)
    assert isinstance(mag_mean, np.ndarray)
    assert len(mag_mean) == 200


def test_get_kcorr_missing_keys():
    phot = Table({'MJD': [50000], 'Telescope': ['HST'], 'Instrument': ['ACS'], 'System': ['Vega']})
    with pytest.raises(ValueError):
        get_kcorr(phot, 0.5)


def test_get_kcorr_output_band():
    phot = Table({'MJD': [50000, 50001], 'Telescope': ['P48', 'Swift'],
                  'Instrument': ['ZTF', 'UVOT'], 'System': ['AB', 'Vega'], 'Filter': ['r', 'UVW1']})
    phot_kcorr = get_kcorr(phot, 0.5, output_band='r', peak=50000)
    assert len(phot_kcorr) == 2


def test_get_kcorr_calculation(mocker):
    phot = Table({'MJD': [55000], 'Telescope': ['P48'], 'Instrument': ['ZTF'], 'System': ['AB'], 'Filter': ['r']})
    mocker.patch('slsne.utils.quick_cenwave_zeropoint', return_value=(6366.38, 3631.0))
    kcorr = get_kcorr(phot, 0.5, peak=55000)
    assert np.isclose(kcorr[0], -0.75398754)


def test_missing_keys():
    phot = Table({'Telescope': ['HST'], 'Instrument': ['ACS'], 'System': ['Vega']})
    with pytest.raises(ValueError):
        fit_map(phot, 0.5)


def test_peak_boom():
    phot = Table({'MJD': [55000], 'Mag': [25.0], 'Telescope': ['P48'],
                  'Instrument': ['ZTF'], 'System': ['AB'], 'Filter': ['r']})
    with pytest.raises(ValueError):
        fit_map(phot, 0.5)
    stretch, amplitude, offset = fit_map(phot, 0.5, peak=55000)
    assert stretch is not None
    assert amplitude is not None
    assert offset is not None


def test_fit_map_calculation(mocker):
    phot = Table({'Mag': [25], 'MJD': [55000], 'Telescope': ['P48'],
                  'Instrument': ['ZTF'], 'System': ['AB'], 'Filter': ['r']})
    mocker.patch('slsne.utils.quick_cenwave_zeropoint', return_value=(6366.38, 3631.0))
    stretch, amplitude, offset = fit_map(phot, 0.5, peak=55000)
    assert np.isclose(stretch, 1.0000000001681757)
    assert np.isclose(amplitude, 4.417201985290917)
    assert np.isclose(offset, 0.29981190993624574)
