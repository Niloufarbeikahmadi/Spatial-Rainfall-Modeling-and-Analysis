#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
visualization.py
-----------------

Contains functions to plot variograms, daily probability maps, and overview figures.
"""

import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import pandas as pd
from matplotlib.patches import Patch

def plot_variograms_with_models(daily_df, class_dict, maxlag: float = 2, n_lags: int = 40, chunk_size: int = 5000, variogram_module=None):
    """
    Plot the experimental variogram and fitted curves for various candidate models.
    Returns a dictionary with model comparison data per class.
    """
    # Import candidate models and fitting functions from variogram_module if provided.
    if variogram_module is None:
        raise ValueError("Please provide the variogram module with model functions.")
    
    models = {
        'Spherical': variogram_module.spherical_model,
        'Exponential': variogram_module.exponential_model,
        'Gaussian': variogram_module.gaussian_model,
        'Matérn': variogram_module.matern_model,
        'Wave Effect': variogram_module.wave_effect_model
    }
    labels = ['Total'] + list(class_dict.keys())
    day_sets = [daily_df['day'].unique()] + list(class_dict.values())
    model_comparison = {label: [] for label in labels}

    num_labels = len(labels)
    fig, axes = plt.subplots(1, num_labels, figsize=(5 * num_labels, 5), sharey=True)
    if num_labels == 1:
        axes = [axes]

    for ax, label, days in zip(axes, labels, day_sets):
        lags, gamma = variogram_module.compute_indicator_variogram(daily_df, days, label, maxlag=maxlag, n_lags=n_lags, chunk_size=chunk_size)
        if lags is not None and gamma is not None and not np.all(np.isnan(gamma)):
            ax.plot(lags, gamma, 'o', color='black', label='Experimental Bins', markersize=3, alpha=0.6)
            for model_name, model_func in models.items():
                if model_name == 'Matérn':
                    popt = variogram_module.fit_matern_model(lags, gamma)
                else:
                    p0 = [np.min(gamma), np.max(gamma) - np.min(gamma), lags[-1] / 2]
                    bounds = (0, [np.inf, np.inf, np.inf])
                    popt = variogram_module.fit_model(model_func, lags, gamma, p0, bounds)
                if popt is not None:
                    gamma_fit = model_func(lags, *popt)
                    aic, bic = variogram_module.calculate_aic_bic(gamma, gamma_fit, len(popt))
                    h_fit = np.linspace(0, maxlag, 100)
                    gamma_model = model_func(h_fit, *popt)
                    ax.plot(h_fit, gamma_model, '-', label=model_name, linewidth=1.5)
                    model_comparison[label].append((model_name, popt, aic, bic))
            ax.set_title(f'{label} Variogram')
            ax.set_xlabel('Lag distance (degrees)')
            ax.set_ylabel('Semivariance')
            ax.grid(True)
            ax.legend()
        else:
            ax.set_title(f'{label} Variogram')
            ax.text(0.5, 0.5, 'No valid variogram', ha='center', va='center', transform=ax.transAxes)
    plt.tight_layout()
    plt.show()
    return model_comparison

def plot_daily_probability_map(prob_map, grid_lons, grid_lats, day, daily_df):
    """
    Plot the daily probability map and the gauge observations.
    """
    daily_df['day'] = pd.to_datetime(daily_df['day'], errors='coerce')
    sub = daily_df[daily_df['day'].dt.date == day]
    lons = sub['Longitude'].values
    lats = sub['Latitude'].values
    vals = sub['indicator'].values.astype(float)

    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    axes[0].scatter(lons[vals == 0], lats[vals == 0], c='red', label='No Rain', alpha=0.7, edgecolor='k', s=50)
    axes[0].scatter(lons[vals == 1], lats[vals == 1], c='blue', label='Rain', alpha=0.7, edgecolor='k', s=50)
    axes[0].set_title(f'Point Distribution of Rainfall - Day {day}')
    axes[0].set_ylabel('Latitude')
    axes[0].legend()
    axes[0].grid(True)

    im = axes[1].imshow(prob_map, extent=(lons.min(), lons.max(), lats.min(), lats.max()),
                          origin='lower', cmap='RdYlBu', vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=axes[1], label='Probability of Rain')
    axes[1].set_title(f'Interpolated Probability of Rain - Day {day}')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

def plot_day_overview(day: 'datetime.date', daily_df: pd.DataFrame,
                      occurrence_results: dict, magnitude_results: dict,
                      title_prefix="Rainfall Overview"):
    """
    Create an overview plot for a given day that includes gauge observations,
    occurrence binary map, gauge rainfall amount, and interpolated rainfall magnitude map.
    """
    day_key = day
    sub = daily_df[daily_df['day'].dt.date == day_key]
    if sub.empty:
        print(f"No gauge data found for day {day_key}.")
        return None, None
    if day_key not in occurrence_results:
        print(f"No occurrence results found for day {day_key}.")
        return None, None

    occ_data = occurrence_results[day_key]
    prob_map = occ_data.get('binary_map')
    grid_lons_occ = occ_data.get('grid_lons')
    grid_lats_occ = occ_data.get('grid_lats')

    if day_key not in magnitude_results:
        print(f"No magnitude results found for day {day_key}.")
        return None, None

    mag_data = magnitude_results[day_key]
    mag_map = mag_data.get('rainfall_map')
    grid_lons_mag = mag_data.get('grid_lons')
    grid_lats_mag = mag_data.get('grid_lats')

    sub_no_rain = sub[sub['Rain'] == 0]
    sub_rain = sub[sub['Rain'] > 0]

    # Load Sicilian shapefile for context.
    sicily = gpd.read_file("C:/Users/nilou/OneDrive/Desktop/UNIPA PHD/obs reconstrution/data/sicily.shp")
    sicily = sicily.to_crs(epsg=4326)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(f"{title_prefix} - {day_key}", fontsize=16, y=0.95)

    def plot_shapefile(ax):
        sicily.boundary.plot(ax=ax, edgecolor='black', linewidth=1)

    ax1 = axes[0, 0]
    ax1.scatter(sub_no_rain['Longitude'], sub_no_rain['Latitude'], c='red', alpha=0.7, edgecolor='k', s=30, label='No Rain')
    ax1.scatter(sub_rain['Longitude'], sub_rain['Latitude'], c='blue', alpha=0.7, edgecolor='k', s=30, label='Rain')
    plot_shapefile(ax1)
    ax1.set_title("Gauge Occurrence Distribution (Rain/No Rain)")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.grid(True)
    ax1.legend()

    ax2 = axes[1, 0]
    if prob_map is not None:
        im2 = ax2.imshow(prob_map, origin='lower', extent=(grid_lons_occ.min(), grid_lons_occ.max(),
                                                             grid_lats_occ.min(), grid_lats_occ.max()),
                         cmap='RdYlBu', vmin=0, vmax=1, aspect='auto')
        legend_elements = [Patch(facecolor='red', edgecolor='k', label='No Rain (0)'),
                           Patch(facecolor='blue', edgecolor='k', label='Rain (1)')]
        ax2.legend(handles=legend_elements, loc='lower left')
    plot_shapefile(ax2)
    ax2.set_title("Occurrence Binary Map")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.grid(True)

    ax3 = axes[0, 1]
    sc3 = ax3.scatter(sub['Longitude'], sub['Latitude'], c=sub['Rain'], cmap='RdYlBu', edgecolor='k', s=30)
    cbar3 = fig.colorbar(sc3, ax=ax3, shrink=0.8)
    cbar3.set_label("Gauge Rainfall (mm)")
    plot_shapefile(ax3)
    ax3.set_title("Gauge Rainfall Amount Distribution")
    ax3.set_xlabel("Longitude")
    ax3.set_ylabel("Latitude")
    ax3.grid(True)

    ax4 = axes[1, 1]
    if mag_map is not None:
        im4 = ax4.imshow(mag_map, origin='lower', extent=(grid_lons_mag.min(), grid_lons_mag.max(),
                                                            grid_lats_mag.min(), grid_lats_mag.max()),
                         cmap='RdYlBu', aspect='auto')
        cbar4 = fig.colorbar(im4, ax=ax4, shrink=0.8)
        cbar4.set_label("Interpolated Rainfall (mm)")
    plot_shapefile(ax4)
    ax4.set_title("Interpolated Rainfall Magnitude Map")
    ax4.set_xlabel("Longitude")
    ax4.set_ylabel("Latitude")
    ax4.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    return fig, axes
