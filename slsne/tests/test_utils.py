import numpy as np
import pytest

from ..utils import fahrenheit_to_celsius

# List of (fahrenheit, expected_celsius) pairs
test_cases = [
    (1, 2),
    (2, 4),
    (3, 6),
    (4, 8),
    (5, 10),
]

# Parameterize test values
@pytest.mark.parametrize("fahrenheit, expected_celsius", test_cases)

# Test that fahrenheit_to_celsius gives the result it is supposed to
def test_fahrenheit_to_celsius(fahrenheit, expected_celsius):
    result = fahrenheit_to_celsius(fahrenheit)
    assert pytest.approx(result, 0.0001) == expected_celsius
