#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
kriging.py
----------

Contains kriging functions for both rainfall occurrence and magnitude interpolation.
"""

import numpy as np
from pykrige.ok import OrdinaryKriging
from tqdm import tqdm
import pandas as pd

def krige_with_best_variogram(daily_df, class_dict, variogram_params, grid_res: float = 0.02):
    """
    Perform kriging for rainfall occurrence using the best-fit variogram parameters per class.
    Returns a dictionary with grid definitions and probability maps.
    """
    day_to_class = {}
    for lbl, days in class_dict.items():
        for d in days:
            day_to_class[pd.Timestamp(d)] = lbl

    lat_min, lat_max = 36.5, 38.5
    lon_min, lon_max = 12, 16
    grid_lats = np.arange(lat_min, lat_max + grid_res, grid_res)
    grid_lons = np.arange(lon_min, lon_max + grid_res, grid_res)
    results = {}

    for day in tqdm(daily_df['day'].unique(), desc="Kriging all days"):
        sub = daily_df[daily_df['day'] == day].drop_duplicates(subset=['Latitude', 'Longitude'])
        if sub['indicator'].nunique() <= 1 or len(sub) < 3:
            unique_vals = sub['indicator'].unique()
            if len(unique_vals) == 1 and unique_vals[0] == 0:
                results[day] = {'grid_lons': grid_lons, 'grid_lats': grid_lats,
                                'prob_map': np.zeros((len(grid_lats), len(grid_lons)))}
            continue

        class_label = day_to_class.get(pd.Timestamp(day))
        if not class_label:
            continue

        params = variogram_params[class_label]
        variogram_model = 'exponential'
        variogram_parameters = {'sill': params[1], 'range': params[2], 'nugget': params[0]}

        try:
            OK = OrdinaryKriging(
                x=sub['Longitude'].values,
                y=sub['Latitude'].values,
                z=sub['indicator'].values.astype(float),
                variogram_model=variogram_model,
                variogram_parameters=variogram_parameters
            )
            z, ss = OK.execute('grid', grid_lons, grid_lats)
            results[day] = {'grid_lons': grid_lons, 'grid_lats': grid_lats, 'prob_map': z}
        except Exception as e:
            print(f"Skipping day {day} due to error: {e}")
            continue
    return results

def compute_cutoff_threshold(prob_map: np.ndarray) -> float:
    """
    Compute the cutoff threshold to convert a probability map to a binary map.
    """
    p = prob_map.flatten()
    m = np.mean(p)
    p_sorted = np.sort(p)
    N = len(p_sorted)
    if m <= 0:
        return 1.0
    if m >= 1:
        return 0.0
    quantile_index = int(np.floor((1.0 - m) * N))
    quantile_index = max(0, min(quantile_index, N - 1))
    return p_sorted[quantile_index]

def calculate_class_cutoffs(results: dict, class_dict: dict) -> dict:
    """
    Calculate cutoff thresholds for each class by aggregating probability maps.
    """
    class_probs = {label: [] for label in class_dict}
    for day, data in results.items():
        prob_map = data['prob_map']
        for class_label, dates in class_dict.items():
            if day in dates:
                class_probs[class_label].append(prob_map)
    class_thresholds = {}
    for class_label, prob_maps in class_probs.items():
        if prob_maps:
            all_probs = np.concatenate([pm.flatten() for pm in prob_maps])
            class_thresholds[class_label] = compute_cutoff_threshold(all_probs)
    return class_thresholds

def create_binary_maps(occurrence_results: dict, class_thresholds: dict, class_dict: dict) -> dict:
    """
    For each day in the occurrence_results, create a binary map based on its class threshold.
    """
    for day_key, data in occurrence_results.items():
        day_ts = pd.Timestamp(day_key)
        found_class = None
        for class_label, days in class_dict.items():
            day_list = [pd.Timestamp(d) for d in days]
            if day_ts in day_list:
                found_class = class_label
                break
        if found_class is None:
            print(f"No class found for day {day_key}. Skipping binary map creation.")
            continue
        threshold = class_thresholds.get(found_class, None)
        if threshold is None:
            print(f"No threshold defined for class {found_class} for day {day_key}.")
            continue
        prob_map = data.get('prob_map')
        if prob_map is None:
            print(f"Probability map not found for day {day_key}.")
            continue
        binary_map = (prob_map > threshold).astype(int)
        occurrence_results[day_key]['binary_map'] = binary_map
        occurrence_results[day_key]['cutoff'] = threshold
    return occurrence_results

def krige_final_rainfall_maps(daily_df, occurrence_results, magnitude_variogram_results, group_days, grid_res: float = 0.02) -> dict:
    """
    For each day, use the binary occurrence map to mask the grid and perform ordinary kriging for rainfall magnitude.
    Returns a dictionary with the final interpolated rainfall maps.
    """
    final_rainfall_maps = {}
    for day_key, occ_data in occurrence_results.items():
        grid_lons = occ_data.get('grid_lons')
        grid_lats = occ_data.get('grid_lats')
        binary_map = occ_data.get('binary_map')
        if binary_map is None:
            print(f"Binary map not found for day {day_key}; skipping.")
            continue

        day_obj = pd.to_datetime(day_key).normalize()
        group_found = None
        for grp, day_list in group_days.items():
            day_list_conv = pd.to_datetime(day_list).normalize()
            if day_obj in day_list_conv.values:
                group_found = grp
                break
        if group_found is None:
            print(f"No group found for day {day_key}; skipping magnitude interpolation.")
            continue

        mag_var_result = magnitude_variogram_results.get(group_found)
        if mag_var_result is None:
            print(f"No magnitude variogram result for group {group_found} on day {day_key}; skipping.")
            continue

        best_model = mag_var_result['Best Model']
        popt = mag_var_result['Parameters']
        model_map = {
            'Spherical': 'spherical',
            'Exponential': 'exponential',
            'Gaussian': 'gaussian',
            'Matérn': 'spherical',      # Fallback if custom model is not provided.
            'Wave Effect': 'spherical'
        }
        variogram_model = model_map.get(best_model, 'spherical')
        variogram_parameters = {'nugget': popt[0], 'sill': popt[1], 'range': popt[2]}

        day_df = daily_df[pd.to_datetime(daily_df['day']).dt.normalize() == day_obj]
        if day_df.empty:
            print(f"No point data for day {day_key}; skipping.")
            continue
        lons = day_df['Longitude'].values
        lats = day_df['Latitude'].values
        rains = day_df['Rain'].values.astype(float)

        try:
            OK = OrdinaryKriging(
                x=lons,
                y=lats,
                z=rains,
                variogram_model=variogram_model,
                variogram_parameters=variogram_parameters,
                verbose=False,
                enable_plotting=False,
                exact_values=True
            )
            z, ss = OK.execute('grid', grid_lons, grid_lats)
            z_masked = np.where(binary_map == 1, z, 0)
            final_rainfall_maps[day_key] = {
                'grid_lons': grid_lons,
                'grid_lats': grid_lats,
                'rainfall_map': z_masked,
                'used_group': group_found,
                'variogram_model': best_model,
                'variogram_parameters': variogram_parameters
            }
            print(f"Day {day_key} processed: group {group_found}, model {best_model}.")
        except Exception as e:
            print(f"Error kriging day {day_key}: {e}")
            continue
    return final_rainfall_maps

def replace_negative_with_zero(final_rain_maps: dict) -> dict:
    """
    Replace all negative rainfall values with zero in the final rainfall maps.
    """
    for day, data in final_rain_maps.items():
        if 'rainfall_map' in data:
            rainfall_map = data['rainfall_map']
            rainfall_map[rainfall_map < 0] = 0
            final_rain_maps[day]['rainfall_map'] = rainfall_map
    return final_rain_maps
