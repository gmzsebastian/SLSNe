from ..lcurve import interpolate, get_all_lcs
import numpy as np
import pytest


def test_interpolate():
    time = np.array([1, 2, 3, 4, 5])
    flux = np.array([10, 20, 30, 40, 50])
    samples = np.array([1.5, 2.5, 3.5, 4.5])

    expected_output = np.array([15, 25, 35, 45])  # Expected interpolated flux values

    output = interpolate(time, flux, samples)

    assert np.allclose(output, expected_output)

    # Test with samples outside the time range
    samples_outside = np.array([-1, 6])
    output = interpolate(time, flux, samples_outside)
    assert np.isnan(output).all(), "The function should return np.nan for samples outside the time range"


def test_get_all_lcs_sigma():
    # Test with valid sigma values
    for sigma in [1, 2, 3]:
        try:
            get_all_lcs('g', sigma=sigma)
        except Exception as e:
            pytest.fail(f"Unexpected Exception with sigma={sigma}: {e}")

    # Test with invalid sigma value
    with pytest.raises(ValueError):
        get_all_lcs('g', sigma=4)

    # Test with non-integer sigma value
    with pytest.raises(ValueError):
        get_all_lcs('g', sigma=1.5)

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
