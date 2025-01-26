# -*- coding: utf-8 -*-
"""
Created on Sun March 26 11:12:46 2024

@author: Niloufar
"""

# variogram_models.py

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import kv, gamma

def spherical_model(h, nugget, sill, a):
    """Spherical variogram model."""
    gamma = np.zeros_like(h, dtype=float)
    inside = h <= a
    outside = h > a
    gamma[inside] = nugget + sill * (1.5 * (h[inside] / a) - 0.5 * (h[inside] / a) ** 3)
    gamma[outside] = nugget + sill
    return gamma

def fit_spherical_model(lags, gamma):
    """
    Fit the spherical variogram model to the experimental variogram.

    Parameters:
        lags (np.ndarray): Array of lag distances.
        gamma (np.ndarray): Experimental semivariance values.

    Returns:
        tuple: Fitted model parameters (nugget, sill, range).
    """
    nugget0 = np.min(gamma)
    sill0 = np.max(gamma) - nugget0
    a0 = lags[-1] / 2 if len(lags) > 0 else 1.0

    bounds = (0, [np.inf, np.inf, np.inf])
    popt, _ = curve_fit(spherical_model, lags, gamma, p0=[nugget0, sill0, a0], bounds=bounds)
    return popt
