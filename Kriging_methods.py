# -*- coding: utf-8 -*-
"""
Created on Sun March 26 12:14:34 2024

@author: niloufar
"""

# kriging_methods.py

import numpy as np
from pykrige.ok import OrdinaryKriging
from tqdm import tqdm

def krige_with_best_variogram(daily_df, class_dict, variogram_params, grid_res=0.02):
    """
    Perform kriging using the best variogram parameters.

    Parameters:
        daily_df (pd.DataFrame): DataFrame with daily rainfall data.
        class_dict (dict): Classification of days by frequency.
        variogram_params (dict): Variogram parameters for each class.
        grid_res (float): Resolution of the output grid.

    Returns:
        dict: Kriging results for each day.
    """
    results = {}
    for day in tqdm(daily_df['day'].unique(), desc="Kriging all days"):
        sub = daily_df[daily_df['day'] == day]

        if sub.empty or len(sub) < 3:
            continue

        grid_lats = np.arange(sub['Latitude'].min(), sub['Latitude'].max() + grid_res, grid_res)
        grid_lons = np.arange(sub['Longitude'].min(), sub['Longitude'].max() + grid_res, grid_res)

        OK = OrdinaryKriging(
            x=sub['Longitude'].values,
            y=sub['Latitude'].values,
            z=sub['Rain'].values,
            variogram_model='spherical',
            nlags=6
        )
        z, ss = OK.execute('grid', grid_lons, grid_lats)

        results[day] = {'grid_lons': grid_lons, 'grid_lats': grid_lats, 'rainfall_map': z}

    return results
