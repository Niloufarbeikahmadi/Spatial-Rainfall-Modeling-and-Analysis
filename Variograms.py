#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
variograms.py
------------

Contains variogram models, model fitting functions, and routines to compute
experimental variograms.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import kv, gamma as sp_gamma

# ----- Candidate Variogram Models -----

def spherical_model(h, nugget, sill, a):
    gamma = np.zeros_like(h, dtype=float)
    inside = h <= a
    outside = h > a
    gamma[inside] = nugget + sill * (1.5 * (h[inside] / a) - 0.5 * (h[inside] / a) ** 3)
    gamma[outside] = nugget + sill
    return gamma

def exponential_model(h, nugget, sill, a):
    return nugget + sill * (1 - np.exp(-h / a))

def gaussian_model(h, nugget, sill, a):
    return nugget + sill * (1 - np.exp(-((h / a) ** 2)))

def matern_model(h, nugget, sill, a, nu):
    """
    Matérn variogram model.
    """
    epsilon = 1e-10  # Prevent division by zero
    h = np.maximum(h, epsilon)
    part1 = (2 ** (1 - nu)) / sp_gamma(nu)
    part2 = (np.sqrt(2 * nu) * h / a) ** nu
    part3 = kv(nu, np.sqrt(2 * nu) * h / a)
    return nugget + sill * (1 - part1 * part2 * part3)

def wave_effect_model(h, nugget, sill, a):
    """
    Wave effect model.
    """
    epsilon = 1e-10
    h_safe = np.maximum(h, epsilon)
    wave_effect = np.sin(h_safe / a) / (h_safe / a)
    wave_effect[h == 0] = 1  # Limit: sin(x)/x -> 1 as x->0
    return nugget + sill * (1 - wave_effect)

# ----- Model Fitting Helpers -----

def fit_model(model_func, lags, gamma, p0, bounds):
    """
    Fit a given model function to the experimental variogram.
    """
    try:
        popt, _ = curve_fit(model_func, lags, gamma, p0=p0, bounds=bounds)
    except Exception as e:
        print(f"Fitting error for model {model_func.__name__}: {e}")
        popt = None
    return popt

def fit_matern_model(lags, gamma):
    """
    Fit the Matérn variogram model including parameter nu.
    """
    nugget0 = np.min(gamma)
    sill0 = np.max(gamma) - nugget0
    a0 = lags[-1] / 2 if len(lags) > 0 else 1.0
    nu0 = 1.5
    bounds = (0, [np.inf, np.inf, np.inf, 5])
    
    def matern_with_nu(h, nugget, sill, a, nu):
        return matern_model(h, nugget, sill, a, nu)
    
    try:
        popt, _ = curve_fit(matern_with_nu, lags, gamma, p0=[nugget0, sill0, a0, nu0], bounds=bounds)
    except Exception as e:
        print(f"Matérn fitting failed: {e}")
        popt = [nugget0, sill0, a0, nu0]
    return popt

def calculate_aic_bic(gamma, gamma_fit, num_params):
    """
    Calculate AIC and BIC for model comparison.
    """
    n = len(gamma)
    residual_sum_squares = np.sum((gamma - gamma_fit) ** 2)
    aic = n * np.log(residual_sum_squares / n) + 2 * num_params
    bic = n * np.log(residual_sum_squares / n) + num_params * np.log(n)
    return aic, bic

# ----- Experimental Variogram Computation -----

def compute_indicator_variogram(daily_df, day_list, class_label: str, maxlag: float = None,
                                n_lags: int = 10, chunk_size: int = 5000):
    """
    Compute experimental variograms for rainfall occurrence (binary indicator) for a list of days.
    Returns averaged lag distances and semivariance.
    """
    all_day_bin_sum = []
    all_day_bin_count = []
    valid_day_count = 0

    for day in day_list:
        sub = daily_df[daily_df['day'] == day]
        if sub.empty:
            continue
        coords = sub[['Longitude', 'Latitude']].values
        vals = sub['indicator'].values.astype(float)
        N = len(coords)
        if N < 2:
            continue

        # Determine maxlag if not provided.
        if maxlag is None:
            # Use a rough bounding box
            lat_min, lat_max = 36.5, 38.5
            lon_min, lon_max = 12, 16
            current_maxlag = np.sqrt((lat_max - lat_min) ** 2 + (lon_max - lon_min) ** 2)
        else:
            current_maxlag = maxlag

        bins = np.linspace(0, current_maxlag, n_lags + 1)
        bin_sum = np.zeros(n_lags, dtype=float)
        bin_count = np.zeros(n_lags, dtype=int)

        # Compute semivariance in chunks.
        for start_i in range(0, N, chunk_size):
            end_i = min(start_i + chunk_size, N)
            coords_chunk = coords[start_i:end_i]
            vals_chunk = vals[start_i:end_i]

            for i in range(len(coords_chunk)):
                dx = coords[:, 0] - coords_chunk[i, 0]
                dy = coords[:, 1] - coords_chunk[i, 1]
                dists = np.sqrt(dx ** 2 + dy ** 2)
                gamma_values = 0.5 * (vals - vals_chunk[i]) ** 2
                digit = np.digitize(dists, bins) - 1
                for k in range(n_lags):
                    mask = (digit == k)
                    if np.any(mask):
                        bin_sum[k] += gamma_values[mask].sum()
                        bin_count[k] += mask.sum()

        with np.errstate(divide='ignore', invalid='ignore'):
            day_semivariance = bin_sum / bin_count
        day_semivariance[np.isnan(day_semivariance)] = 0
        all_day_bin_sum.append(day_semivariance)
        all_day_bin_count.append(np.where(bin_count > 0, 1, 0))
        valid_day_count += 1

    if valid_day_count == 0:
        return None, None

    all_day_bin_sum = np.array(all_day_bin_sum)
    all_day_bin_count = np.array(all_day_bin_count)
    avg_semivariance = np.sum(all_day_bin_sum, axis=0) / np.sum(all_day_bin_count, axis=0)
    avg_lags = bins[:-1]
    return avg_lags, avg_semivariance

def compute_magnitude_variogram(daily_df, day_list, maxlag: float = None,
                                n_lags: int = 10, chunk_size: int = 5000):
    """
    Compute the experimental variogram for rainfall magnitude (using the 'Rain' field)
    for the specified list of days.
    """
    all_day_bin_sum = []
    all_day_bin_count = []
    valid_day_count = 0

    for day in day_list:
        sub = daily_df[daily_df['day'] == day]
        if sub.empty:
            continue
        coords = sub[['Longitude', 'Latitude']].values
        vals = sub['Rain'].values.astype(float)
        N = len(coords)
        if N < 2:
            continue

        if maxlag is None:
            lat_min, lat_max = coords[:, 1].min(), coords[:, 1].max()
            lon_min, lon_max = coords[:, 0].min(), coords[:, 0].max()
            current_maxlag = np.sqrt((lat_max - lat_min) ** 2 + (lon_max - lon_min) ** 2)
        else:
            current_maxlag = maxlag

        bins = np.linspace(0, current_maxlag, n_lags + 1)
        bin_sum = np.zeros(n_lags, dtype=float)
        bin_count = np.zeros(n_lags, dtype=int)

        for start_i in range(0, N, chunk_size):
            end_i = min(start_i + chunk_size, N)
            coords_chunk = coords[start_i:end_i]
            vals_chunk = vals[start_i:end_i]
            for i in range(len(coords_chunk)):
                dx = coords[:, 0] - coords_chunk[i, 0]
                dy = coords[:, 1] - coords_chunk[i, 1]
                dists = np.sqrt(dx ** 2 + dy ** 2)
                semivariance_values = 0.5 * (vals - vals_chunk[i]) ** 2
                digit = np.digitize(dists, bins) - 1
                for k in range(n_lags):
                    mask = (digit == k)
                    if np.any(mask):
                        bin_sum[k] += semivariance_values[mask].sum()
                        bin_count[k] += mask.sum()

        with np.errstate(divide='ignore', invalid='ignore'):
            day_semivariance = bin_sum / bin_count
        day_semivariance[np.isnan(day_semivariance)] = 0
        all_day_bin_sum.append(day_semivariance)
        all_day_bin_count.append(np.where(bin_count > 0, 1, 0))
        valid_day_count += 1

    if valid_day_count == 0:
        return None, None
    all_day_bin_sum = np.array(all_day_bin_sum)
    all_day_bin_count = np.array(all_day_bin_count)
    exp_gamma = np.sum(all_day_bin_sum, axis=0) / np.sum(all_day_bin_count, axis=0)
    avg_lags = bins[:-1]
    return avg_lags, exp_gamma

def select_best_variogram_model(lags: np.ndarray, exp_gamma: np.ndarray, maxlag: float):
    """
    Fit candidate models to the experimental variogram and select the best model based on AIC.
    Returns a tuple: (model_name, parameters, AIC, BIC)
    """
    models = {
        'Spherical': spherical_model,
        'Exponential': exponential_model,
        'Gaussian': gaussian_model,
        'Matérn': matern_model,
        'Wave Effect': wave_effect_model
    }
    model_results = []
    for model_name, model_func in models.items():
        if model_name == 'Matérn':
            popt = fit_matern_model(lags, exp_gamma)
        else:
            p0 = [np.min(exp_gamma), np.max(exp_gamma) - np.min(exp_gamma), maxlag/2]
            bounds = (0, [np.inf, np.inf, np.inf])
            popt = fit_model(model_func, lags, exp_gamma, p0, bounds)
        if popt is None:
            continue
        try:
            fitted_gamma = model_func(lags, *popt)
        except Exception as e:
            print(f"Error in model {model_name}: {e}")
            continue
        aic, bic = calculate_aic_bic(exp_gamma, fitted_gamma, len(popt))
        model_results.append((model_name, popt, aic, bic))
    if not model_results:
        return None
    best_model = min(model_results, key=lambda x: x[2])
    return best_model
