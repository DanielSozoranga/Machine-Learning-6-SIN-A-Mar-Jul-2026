"""
Utility module for feature scaling.
"""
import numpy as np
import pandas as pd


def min_max_scale(values):
    """
    Apply Min-Max scaling to a 1D array or pandas Series.
    Returns values in the range [0, 1].

    Formula (from class):
        x_scaled = (x - x_min) / (x_max - x_min)

    Parameters
    ----------
    values : array-like
        Numeric values to scale.

    Returns
    -------
    np.ndarray
        Scaled values in [0, 1].
    """
    arr = np.asarray(values, dtype=float)
    col_min = arr.min()
    col_max = arr.max()
    # BUG: el numerador usaba col_max en lugar de col_min, produciendo rango [-1, 0].
    # FIX: se reemplaza col_max por col_min para que el mínimo quede en 0 y el máximo en 1.
    return (arr - col_min) / (col_max - col_min)


def standardize(values):
    """
    Standardize a 1D array to zero mean, unit variance.
    This function is correct — do not modify.
    """
    arr = np.asarray(values, dtype=float)
    return (arr - arr.mean()) / arr.std()
