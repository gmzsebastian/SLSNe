"""
This file contains general utilities to process the data
from the SLSNe package.
"""

def fahrenheit_to_celsius(fahrenheit):
    """
    Convert Fahrenheit to Celsius.

    Parameters
    ----------
    fahrenheit : float
        Temperature in Fahrenheit.

    Returns
    -------
    celsius: float
        Temperature in Celsius.
    """
    celsius = (fahrenheit - 32) * 5.0/9.0
    return celsius
