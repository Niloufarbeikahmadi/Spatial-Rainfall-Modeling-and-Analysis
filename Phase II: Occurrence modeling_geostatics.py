# -*- coding: utf-8 -*-
"""
Rainfall Interpolation Pipeline - Phase II: Magnitude Interpolation
Created on Wed Jul  9 14:05:48 2025
@author: Utente
"""

# Standard library imports
import os
import pickle
import warnings
from datetime import date
from typing import Any, Callable, Dict, List, Tuple

# Third-party imports
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import dill
import geopandas as gpd
import gstools as gs
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
import xarray as xr
from matplotlib.collections import PatchCollection
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from matplotlib.ticker import FuncFormatter
from pykrige.ok import OrdinaryKriging
from pyproj import Geod, Transformer
from scipy.stats import probplot, shapiro
from shapely.geometry import Point, box
from shapely.ops import unary_union
from sklearn.preprocessing import (FunctionTransformer, PowerTransformer,
                                  QuantileTransformer)
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Constants and Configuration
# -----------------------------------------------------------------------------
SHAPEFILE_PATH = "C:/Users/Utente/.../sicily.shp"
RESOLUTION_DEG = 0.02  # ~2km resolution
LON_MIN, LON_MAX = 12.0, 16.0
LAT_MIN, LAT_MAX = 36.5, 38.5

# Variogram models to evaluate
GS_MODELS = {
    "Gaussian": gs.Gaussian,
    "Exponential": gs.Exponential,
    "Matern": gs.Matern,
    "Stable": gs.Stable,
    "Rational": gs.Rational,
    "Spherical": gs.Spherical,
    "SuperSpherical": gs.SuperSpherical,
    "JBessel": gs.JBessel,
    "Cubic": gs.Cubic,
    "Integral": gs.Integral
}

# Parameter order for different models
PARAM_ORDER = {
    "Gaussian": ["sill", "range", "nugget"],
    "Exponential": ["sill", "range", "nugget"],
    "Matern": ["sill", "range", "nugget", "nu"],
    "Stable": ["sill", "range", "nugget", "alpha"],
    "Rational": ["sill", "range", "nugget", "alpha"],
    "Spherical": ["sill", "range", "nugget"],
    "SuperSpherical": ["sill", "range", "nugget", "alpha"],
    "JBessel": ["sill", "range", "nugget", "alpha"],
    "Cubic": ["sill", "range", "nugget"],
    "Integral": ["sill", "range", "nugget", "nu"]
}

# -----------------------------------------------------------------------------
# Core Helper Functions
# -----------------------------------------------------------------------------
def reproject_df(
    df: pd.DataFrame,
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    src_epsg: int = 4326,
    dst_epsg: int = 32633,
) -> pd.DataFrame:
    """Reproject DataFrame coordinates to a different CRS."""
    transformer = Transformer.from_crs(src_epsg, dst_epsg, always_xy=True)
    x, y = transformer.transform(df[lon_col].to_numpy(), df[lat_col].to_numpy())
    out = df.copy()
    out[lon_col] = x
    out[lat_col] = y
    return out

def create_grid_in_meters(
    lon_min_deg: float,
    lon_max_deg: float,
    lat_min_deg: float,
    lat_max_deg: float,
    resolution_deg: float = 0.02
) -> tuple:
    """Create a grid in meters based on geographic bounds and resolution."""
    geod = Geod(ellps="WGS84")
    center_lat = (lat_min_deg + lat_max_deg) / 2
    _, _, dx = geod.inv(lon_min_deg, center_lat, lon_min_deg + resolution_deg, center_lat)
    _, _, dy = geod.inv(lon_min_deg, center_lat, lon_min_deg, center_lat + resolution_deg)
    transformer = Transformer.from_crs(4326, 32633, always_xy=True)
    x_min_m, y_min_m = transformer.transform(lon_min_deg, lat_min_deg)
    x_max_m, y_max_m = transformer.transform(lon_max_deg, lat_max_deg)
    grid_x = np.arange(x_min_m, x_max_m + dx, dx)
    grid_y = np.arange(y_min_m, y_max_m + dy, dy)
    return grid_x, grid_y, transformer

def create_sicily_mask_rtree(
    shapefile_path: str, 
    grid_x: np.ndarray, 
    grid_y: np.ndarray, 
    buffer_meters: int = 2000
) -> np.ndarray:
    """Create a land mask for Sicily using a shapefile."""
    sicily = gpd.read_file(shapefile_path).to_crs(epsg=32633)
    buffered = unary_union(sicily.geometry).buffer(buffer_meters)
    mask = np.zeros((len(grid_y), len(grid_x)), dtype=bool)
    for j in range(len(grid_y)):
        for i in range(len(grid_x)):
            x, y = grid_x[i], grid_y[j]
            mask[j,i] = buffered.contains(Point(x,y))
    return ~mask

# -----------------------------------------------------------------------------
# Variogram Analysis Functions
# -----------------------------------------------------------------------------
def _aic_weighted(residuals: np.ndarray, weights: np.ndarray, k: int) -> float:
    """Calculate weighted Akaike Information Criterion (AIC)."""
    rss = np.sum(weights * residuals ** 2)
    n_eff = weights.sum()
    return n_eff * np.log(rss / n_eff) + 2 * k

def _fit_models(
    bins: np.ndarray,
    gamma: np.ndarray,
    counts: np.ndarray,
    models: Dict[str, type],
    max_eval: int = 30_000,
) -> Tuple[List[dict], dict]:
    """Fit variogram models and select the best one based on AIC."""
    all_fits = []
    best_fit = None
    best_aic = np.inf
    fit_opts = dict(weights=counts, loss="linear", max_eval=max_eval)

    for name, Model in models.items():
        mod = Model(dim=2)
        converged = False
        for guess in ("default", "current"):
            try:
                fit_para, _ = mod.fit_variogram(bins, gamma, init_guess=guess, **fit_opts)
                converged = True
                break
            except RuntimeError:
                continue
        if not converged:
            warnings.warn(f"{name} fit failed; skipping.")
            continue
        pred = mod.variogram(bins)
        aic = _aic_weighted(gamma - pred, counts, k=len(fit_para))
        fit_data = {"name": name, "model": mod, "params": fit_para, "aic": aic}
        all_fits.append(fit_data)
        if aic < best_aic:
            best_fit = fit_data
            best_aic = aic
    if not all_fits:
        raise RuntimeError("No variogram model converged.")
    return all_fits, best_fit

def create_gstools_variogram(model_name: str, params_dict: dict) -> Tuple[Callable, list]:
    """Create a GSTools variogram function from parameters."""
    if model_name not in PARAM_ORDER:
        raise ValueError(f"Unsupported model: {model_name}")
    
    nugget = params_dict["nugget"]
    var = params_dict["var"]
    len_scale = params_dict["len_scale"]
    full_sill = nugget + var
    opt_params = {}
    
    if model_name in ["Matern", "SuperSpherical","Integral"]: 
        opt_params["nu"] = params_dict["nu"]
    elif model_name in ["Stable", "Rational", "JBessel"]:
        opt_params["alpha"] = params_dict["alpha"]
    
    param_list = [full_sill, len_scale, nugget]
    if model_name in PARAM_ORDER and len(PARAM_ORDER[model_name]) > 3:
        param_list.append(opt_params[list(opt_params.keys())[0]])
    
    def variogram_function(params, dist):
        model_class = getattr(gs, model_name)
        model = model_class(
            dim=2,
            nugget=params[2],
            var=params[0] - params[2],
            len_scale=params[1],
            **opt_params
        )
        return model.variogram(dist)
    
    return variogram_function, param_list

# -----------------------------------------------------------------------------
# Transformation Functions
# -----------------------------------------------------------------------------
def apply_transformations(
    data: np.ndarray, 
    transformation_names: List[str], 
    plot: bool = True
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float]]:
    """Apply and evaluate different data transformations."""
    results = {}
    p_values = {}
    
    for name in transformation_names:
        try:
            if name == 'log':
                transformer = FunctionTransformer(
                    func=np.log, 
                    inverse_func=lambda x: np.exp(np.clip(x, -20, 20)),
                    check_inverse=False
                )
                transformed = transformer.transform(data.reshape(-1, 1))
            
            elif name == 'boxcox':
                data_boxcox = np.where(data <= 0, 1e-6, data)
                transformer = PowerTransformer(method='box-cox', standardize=False)
                transformer.fit(data_boxcox.reshape(-1, 1))
                transformed = transformer.transform(data_boxcox.reshape(-1, 1))
            
            elif name == 'yeojohnson':
                transformer = PowerTransformer(method='yeo-johnson', standardize=False)
                transformer.fit(data.reshape(-1, 1))
                transformed = transformer.transform(data.reshape(-1, 1))
            
            elif name == 'quantile':
                transformer = QuantileTransformer(
                    output_distribution='normal',
                    n_quantiles=min(1000, len(data))
                )
                transformer.fit(data.reshape(-1, 1))
                transformed = transformer.transform(data.reshape(-1, 1))
                
            else:
                continue
            
            transformed_flat = transformed.flatten()
            sample = transformed_flat if len(transformed_flat) <= 5000 else np.random.choice(transformed_flat, 5000)
            _, p_val = shapiro(sample)
            p_values[name] = p_val
            
            def transform_wrapper(x, t=transformer):
                return t.transform(x.reshape(-1, 1)).flatten()
            
            def inverse_wrapper(x, t=transformer):
                return t.inverse_transform(x.reshape(-1, 1)).flatten()
            
            results[name] = {
                'normalizer': transformer,
                'transform_func': transform_wrapper,
                'inverse_func': inverse_wrapper,
                'transformed': transformed_flat,
                'p_value': p_val
            }
            
        except Exception as e:
            warnings.warn(f"Transformation {name} failed: {str(e)}")
    
    return results, p_values

def select_best_transformation(trans_results: tuple) -> str:
    """Select the best transformation based on Shapiro-Wilk p-values."""
    _, p_values = trans_results 
    return max(p_values, key=p_values.get)

# -----------------------------------------------------------------------------
# Variogram Processing Functions
# -----------------------------------------------------------------------------
def variogram_magnitude_by_class(
    df: pd.DataFrame,
    class_dict: Dict[str, List[date]],
    transformer: Any,
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    rain_col: str = "Rain",
    maxlag_deg: float = 2.0,
    n_lags: int = 20,
    chunk_size: int = 25000,
    plot: bool = True,
) -> Dict[str, Dict[str, object]]:
    """Compute daily-pooled variograms for transformed rainfall magnitude per class."""
    DEG_TO_M = 111000.0
    bin_edges = np.linspace(0.0, maxlag_deg * DEG_TO_M, n_lags)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    n_bins = len(bin_centers)

    results = {}
    fig, axes = plt.subplots(3, 3, figsize=(15, 12)) if plot else (None, [None]*9)
    axes = axes.flatten() if plot else None
    max_gamma = 0

    for idx, (label, days) in enumerate(tqdm(class_dict.items(), desc="Magnitude Variogram")):
        sum_gamma = np.zeros(n_bins)
        sum_counts = np.zeros(n_bins)

        for day in days:
            sub = df[df["day"] == day]
            non_zero = sub[sub[rain_col] > 0]
            if len(non_zero) < 2:
                continue
            try:
                x = non_zero[lon_col].values
                y = non_zero[lat_col].values
                rain_vals = non_zero[rain_col].values.reshape(-1, 1)
                vals = transformer.transform(rain_vals).flatten()
                
                bins_d, gamma_d, counts_d = gs.vario_estimate(
                    (x, y), vals, bin_edges=bin_edges, sampling_size=chunk_size,
                    estimator="matheron", latlon=False, return_counts=True
                )
            except Exception as err:
                warnings.warn(f"{day}: {str(err)}; skipped.")
                continue
                
            gamma_d = np.nan_to_num(gamma_d)
            sum_gamma += gamma_d * counts_d
            sum_counts += counts_d

        valid = sum_counts > 0
        if not valid.any():
            warnings.warn(f"Class {label}: no variogram pairs; skipping.")
            continue
            
        gamma_mean = np.zeros(n_bins)
        gamma_mean[valid] = sum_gamma[valid] / sum_counts[valid]

        all_fits, best_fit = _fit_models(bin_centers, gamma_mean, sum_counts, GS_MODELS)
        results[label] = {
            "all_models": all_fits,
            "best_model": best_fit["name"],
            "params": best_fit["params"],
            "AIC": best_fit["aic"],
            "bin_centers": bin_centers,
            "gamma_mean": gamma_mean   
        }

    return results

# -----------------------------------------------------------------------------
# Kriging Functions
# -----------------------------------------------------------------------------
def conditional_kriging_magnitude(
    day: date,
    df: pd.DataFrame,
    normalizer_info: Dict[str, Any],
    variogram_func: callable,
    variogram_params: list,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    land_mask: np.ndarray,
    binary_map: np.ndarray,
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    rain_col: str = "Rain",
) -> np.ndarray:
    """Perform conditional Ordinary Kriging with proper mask handling."""
    result_grid = np.full((len(grid_y), len(grid_x)), np.nan)
    result_grid[~land_mask] = 0
    
    sub = df[df["day"] == day]
    if len(sub) == 0 or not np.any(binary_map == 1):
        return result_grid
    
    non_zero = sub[sub[rain_col] > 0]
    if len(non_zero) == 0:
        return result_grid
    
    wet_mask = np.logical_and(~land_mask, binary_map == 1)
    if len(non_zero[rain_col].values) == 1:
        try:    
            result_grid[wet_mask] = non_zero[rain_col].values[0]
        except Exception as e:
            warnings.warn(f"Back-transformation failed for {day}: {str(e)}")
        return result_grid
    
    transform_func = normalizer_info['transform_func']
    inverse_func = normalizer_info['inverse_func']
    
    try:
        z_vals = transform_func(non_zero[rain_col].values)
    except Exception as e:
        warnings.warn(f"Transformation failed for {day}: {str(e)}")
        return result_grid
    
    valid_mask = np.isfinite(z_vals)
    if not np.any(valid_mask):
        return result_grid
    
    x_vals = non_zero[lon_col].values[valid_mask]
    y_vals = non_zero[lat_col].values[valid_mask]
    z_vals = z_vals[valid_mask]    
    
    try:
        OK = OrdinaryKriging(
            x_vals, y_vals, z_vals,
            variogram_model='custom',
            variogram_function=variogram_func,
            variogram_parameters=variogram_params,
            enable_plotting=False,
            verbose=False,
            pseudo_inv=True,
            pseudo_inv_type='pinvh',
            weight=True,
            exact_values=False,  
            coordinates_type='euclidean'
        )
        
        z, _ = OK.execute(
            style='masked', 
            xpoints=grid_x, 
            ypoints=grid_y,
            mask=~wet_mask
        )
        
        valid_mask = np.isfinite(z.data)
        if np.any(valid_mask):
            try:
                z_valid = z.data[valid_mask]
                back_transformed = inverse_func(z_valid)
                wet_output = np.full(z.shape, np.nan)
                wet_output[valid_mask] = back_transformed
                result_grid[wet_mask] = wet_output[wet_mask]
            except Exception as e:
                warnings.warn(f"Back-transformation failed for {day}: {str(e)}")
        
        return result_grid
    
    except Exception as e:
        warnings.warn(f"Kriging failed for {day}: {str(e)}")
        return result_grid

# -----------------------------------------------------------------------------
# Data Management Functions
# -----------------------------------------------------------------------------
def create_magnitude_dataset(
    kriged_magnitude: Dict[date, np.ndarray],
    grid_x: np.ndarray,
    grid_y: np.ndarray
) -> xr.Dataset:
    """Create xarray Dataset for rainfall magnitude results."""
    sorted_dates = sorted(kriged_magnitude.keys())
    magnitude_stack = [kriged_magnitude[date] for date in sorted_dates]
    
    ds = xr.Dataset(
        {"rainfall_magnitude": (("time", "y", "x"), magnitude_stack)},
        coords={
            "time": pd.to_datetime(sorted_dates),
            "x": grid_x,
            "y": grid_y,
        }
    )
    ds.x.attrs = {"units": "meters", "crs": "EPSG:32633"}
    ds.y.attrs = {"units": "meters", "crs": "EPSG:32633"}
    ds.rainfall_magnitude.attrs = {"units": "mm", "long_name": "Rainfall magnitude"}
    return ds

def save_phase2_results(save_path: str, **kwargs) -> None:
    """Save Phase II results to disk."""
    os.makedirs(save_path, exist_ok=True)
    for name, obj in kwargs.items():
        with open(os.path.join(save_path, f"{name}.pkl"), "wb") as f:
            pickle.dump(obj, f)

# -----------------------------------------------------------------------------
# Visualization Functions
# -----------------------------------------------------------------------------
def plot_qq_transformations(trans_results_by_category: Dict[str, Dict[str, Any]]) -> None:
    """Plot Q-Q plots for all transformations in a 3x3 grid."""
    categories = list(trans_results_by_category.keys())
    transformation_names = ['log', 'boxcox', 'yeojohnson', 'quantile']
    colors = ['blue', 'green', 'red', 'purple']
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle("Q-Q Plots by Transformation Type", fontsize=20, y=0.95)
    
    for ax, category in zip(axes.flat, categories):
        if category not in trans_results_by_category:
            ax.axis('off')
            continue
            
        trans_results = trans_results_by_category[category]
        ax.set_title(category, fontsize=14)
        
        for i, name in enumerate(transformation_names):
            if name in trans_results:
                transformed = trans_results[name]['transformed']
                finite_vals = transformed[np.isfinite(transformed)]
                if len(finite_vals) < 2:
                    continue
                (osm, osr) = probplot(finite_vals, dist="norm", fit=False)
                ax.scatter(osm, osr, s=15, alpha=0.7, color=colors[i], label=name)
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                lims = [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]
                ax.plot(lims, lims, 'k-', alpha=0.3, lw=1)
        
        ax.set_xlabel("Theoretical Quantiles", fontsize=10)
        ax.set_ylabel("Ordered Values", fontsize=10)
        ax.grid(alpha=0.3)
    
    legend_elements = [Patch(color=colors[i], label=name) 
                       for i, name in enumerate(transformation_names)]
    
    fig.legend(handles=legend_elements, 
               loc='lower center', 
               ncol=4,
               fontsize=12,
               frameon=True,
               framealpha=0.9,
               bbox_to_anchor=(0.5, 0.01))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig("qq_plots.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_variograms_magnitude(variogram_results: Dict[str, Dict[str, Any]]) -> None:
    """Plot all variogram results in a single 3x3 grid with shared legend."""
    categories = list(variogram_results.keys())
    model_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    model_styles = ['-', '--', '-.', ':'] * 3
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle("Magnitude Variograms by Category", fontsize=20, y=0.95)
    
    legend_handles = []
    model_counter = {}
    
    for ax, category in zip(axes.flat, categories):
        if category not in variogram_results:
            ax.axis('off')
            continue
            
        data = variogram_results[category]
        bin_centers = data['bin_centers'] / 1000
        gamma_mean = data['gamma_mean']
        all_fits = data['all_models']
        best_fit = data['best_model']
        
        ax.scatter(bin_centers, gamma_mean, s=30, c='k', zorder=10, label='Empirical')
        
        dist_line = np.linspace(0, bin_centers[-1]*1.2, 200)
        for fit in all_fits:
            model_name = fit['name']
            if model_name not in model_counter:
                color_idx = len(model_counter) % 10
                style_idx = len(model_counter) // 10
                model_counter[model_name] = (model_colors[color_idx], model_styles[style_idx])
            
            color, style = model_counter[model_name]
            y_vals = fit['model'].variogram(dist_line * 1000)
            line = ax.plot(dist_line, y_vals, 
                           linestyle=style, color=color,
                           linewidth=1.5, 
                           alpha=0.8 if model_name == best_fit else 0.6)
            
            if model_name == best_fit:
                line[0].set_linewidth(2.5)
        
        ax.set_title(category, fontsize=14)
        ax.set_xlabel("Distance (km)", fontsize=12)
        ax.set_ylabel("γ(h)", fontsize=12)
        ax.grid(alpha=0.3)
    
    for model, (color, style) in model_counter.items():
        legend_handles.append(Line2D([0], [0], 
                                     color=color, 
                                     linestyle=style,
                                     lw=2, 
                                     label=model))
    
    fig.legend(handles=legend_handles, 
               loc='lower center', 
               ncol=5,
               fontsize=12,
               frameon=True,
               framealpha=0.9,
               bbox_to_anchor=(0.5, 0.01))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig("magnitude_variograms.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_four_panels(
    date_target: date,
    daily_df: pd.DataFrame,
    kriged_maps: Dict[date, Dict[str, np.ndarray]],
    kriged_magnitude: Dict[date, np.ndarray],
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    land_mask: np.ndarray,
    shapefile_path: str,
    save_path: str = None
) -> None:
    """Create a 2x2 panel plot for a specific date with consistent sizing."""
    fig = plt.figure(figsize=(25, 16))
    gs = GridSpec(3, 2, figure=fig, 
                 height_ratios=[1, 1, 0.08],
                 hspace=0.15, wspace=0.1)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    cax_prob = fig.add_subplot(gs[2, 0])
    cax_mag = fig.add_subplot(gs[2, 1])
    
    fig.suptitle(f"Rainfall Analysis - {date_target.strftime('%Y-%m-%d')}", fontsize=22, y=0.98)
    sicily = gpd.read_file(shapefile_path).to_crs(epsg=32633)
    transformer = Transformer.from_crs(32633, 4326, always_xy=True)
    
    def deg_formatter(x, pos):
        return f"{x:.2f}°"
    
    total_bounds = sicily.total_bounds
    minx, miny, maxx, maxy = total_bounds
    min_lon, min_lat = transformer.transform(minx, miny)
    max_lon, max_lat = transformer.transform(maxx, maxy)
    lon_buffer = 0.05 * (max_lon - min_lon)
    lat_buffer = 0.05 * (max_lat - min_lat)
    
    def create_sicily_patches():
        patches = []
        for geom in sicily.geometry:
            simplified = geom.simplify(100)
            if isinstance(simplified, shapely.geometry.Polygon):
                x, y = simplified.exterior.xy
                lon, lat = transformer.transform(x, y)
                polygon = Polygon(np.column_stack((lon, lat)), closed=True, fill=False)
                patches.append(polygon)
            elif isinstance(simplified, shapely.geometry.MultiPolygon):
                for poly in simplified.geoms:
                    x, y = poly.exterior.xy
                    lon, lat = transformer.transform(x, y)
                    polygon = Polygon(np.column_stack((lon, lat)), closed=True, fill=False)
                    patches.append(polygon)
        return patches
    
    sicily_patches = create_sicily_patches()
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlim(min_lon - lon_buffer, max_lon + lon_buffer)
        ax.set_ylim(min_lat - lat_buffer, max_lat + lat_buffer)
        ax.grid(False)
        border = PatchCollection(sicily_patches, edgecolor='black', 
                                 facecolor='none', linewidth=1.5, zorder=10)
        ax.add_collection(border)

    # Subplot 1: Occurrence Points
    day_data = daily_df[daily_df['day'] == date_target]
    dry_points = day_data[day_data['Rain'] == 0]
    wet_points = day_data[day_data['Rain'] > 0]
    
    if not dry_points.empty:
        dry_lon_deg, dry_lat_deg = transformer.transform(
            dry_points['Longitude'].values, dry_points['Latitude'].values
        )
        ax1.scatter(dry_lon_deg, dry_lat_deg, c='red', s=15, alpha=0.7, label='Dry')
    
    if not wet_points.empty:
        wet_lon_deg, wet_lat_deg = transformer.transform(
            wet_points['Longitude'].values, wet_points['Latitude'].values
        )
        ax1.scatter(wet_lon_deg, wet_lat_deg, c='blue', s=15, alpha=0.7, label='Wet')
    
    ax1.set_title('Occurrence at Gauges', fontsize=16)
    ax1.set_xlabel('Longitude', fontsize=12)
    ax1.set_ylabel('Latitude', fontsize=12)
    ax1.legend(loc='lower left', fontsize=12)
    ax1.xaxis.set_major_formatter(FuncFormatter(deg_formatter))
    ax1.yaxis.set_major_formatter(FuncFormatter(deg_formatter))

    # Subplot 2: Point Rainfall Magnitude
    all_lon_deg, all_lat_deg = transformer.transform(
        day_data['Longitude'].values, day_data['Latitude'].values
    )
    rain_max = day_data['Rain'].max()
    vmax = max(rain_max, 1)
    sc = ax2.scatter(all_lon_deg, all_lat_deg,
                     c=day_data['Rain'], cmap='RdYlBu', 
                     norm=mcolors.Normalize(vmin=0, vmax=vmax),
                     s=25)
    ax2.set_title('Gauge Rainfall Magnitude (All Stations)', fontsize=16)
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # Subplot 3: Probability Map
    prob_map = kriged_maps[date_target]['prob_map']
    prob_masked = np.ma.masked_where(land_mask, prob_map)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
    grid_lon, grid_lat = transformer.transform(grid_xx, grid_yy)
    im3 = ax3.pcolormesh(
        grid_lon, grid_lat, prob_masked,
        cmap='RdYlBu',
        vmin=0, vmax=1,
        shading='nearest'
    )
    ax3.set_aspect('equal')
    ax3.set_title('Probability of Rainfall Occurrence', fontsize=16)
    ax3.set_xticks([])
    ax3.set_yticks([])
    
    # Subplot 4: Kriged Rainfall Magnitude
    magnitude_grid = kriged_magnitude[date_target].copy()
    magnitude_plot = np.where(
        land_mask | np.isnan(magnitude_grid),
        np.nan,
        magnitude_grid
    )
    kriged_max = np.nanmax(magnitude_plot) if not np.all(np.isnan(magnitude_plot)) else 0
    vmax_kriged = max(kriged_max, vmax, 1)
    im4 = ax4.pcolormesh(
        grid_lon, grid_lat, magnitude_plot,
        cmap='RdYlBu',
        vmin=0,
        vmax=vmax_kriged,
        shading='nearest'
    )
    ax4.set_aspect('equal')
    ax4.set_title('Kriged Rainfall Magnitude', fontsize=16)
    ax4.set_xticks([])
    ax4.set_yticks([])
    
    # Color Bars
    cbar_prob = fig.colorbar(im3, cax=cax_prob, orientation='horizontal')
    cbar_prob.set_label('Probability', fontsize=12)
    cax_prob.xaxis.set_ticks_position('bottom')
    cbar_mag = fig.colorbar(im4, cax=cax_mag, orientation='horizontal')
    cbar_mag.set_label('Rainfall Magnitude (mm)', fontsize=12)
    cax_mag.xaxis.set_ticks_position('bottom')
    
    for ax in [ax1, ax2]:
        ax.set_aspect('equal')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()

# -----------------------------------------------------------------------------
# Main Workflow
# -----------------------------------------------------------------------------
def main():
    # Load Phase I results (uncomment and implement as needed)
    # daily_df, class_dict, kriged_maps, land_mask, final_group_days = load_phase1_results()
    
    # Reproject data and create grid
    daily_df = reproject_df(daily_df)
    grid_x, grid_y, _ = create_grid_in_meters(LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, RESOLUTION_DEG)
    land_mask = create_sicily_mask_rtree(SHAPEFILE_PATH, grid_x, grid_y)

    # Phase II Processing
    kriged_magnitudeII = {}
    trans_results_by_category = {}
    variogram_results_by_category = {}
    
    for category, days in final_group_days.items():
        print(f"\nProcessing category: {category}")
        cat_rain = []
        for day in days:
            day_data = daily_df[daily_df["day"] == day]
            non_zero = day_data[day_data["Rain"] > 0]["Rain"].values
            cat_rain.extend(non_zero)
        cat_rain = np.array(cat_rain)
        
        trans_results, p_values = apply_transformations(
            cat_rain, 
            ['log', 'boxcox', 'yeojohnson', 'quantile'],
            plot=True
        )
        best_trans_name = select_best_transformation((trans_results, p_values))
        best_transformer = trans_results[best_trans_name]['normalizer']  
        print(f"Selected transformation: {best_trans_name} (p-value: {p_values[best_trans_name]:.4f})")
        trans_results_by_category[category] = trans_results
        
        variogram_results = variogram_magnitude_by_class(
            daily_df, {category: days}, best_transformer, plot=False
        )
        variogram_results_by_category[category] = variogram_results[category]

        if category not in variogram_results:
            warnings.warn(f"Skipping category {category} - no valid variogram")
            continue
            
        best_model_info = variogram_results[category]
        variogram_func, variogram_params = create_gstools_variogram(
            best_model_info["best_model"], best_model_info["params"]
        )
        
        for day in tqdm(days, desc=f"Kriging {category} days"):
            if day not in kriged_maps:
                warnings.warn(f"Skipping {day} (missing in Phase I results)")
                continue
            
            binary_map = kriged_maps[day]['binary_map']
            rainfall_grid = conditional_kriging_magnitude(
                day=day,
                df=daily_df,
                normalizer_info=trans_results[best_trans_name],
                variogram_func=variogram_func,
                variogram_params=variogram_params,
                grid_x=grid_x,
                grid_y=grid_y,
                land_mask=land_mask,
                binary_map=binary_map
            )
            kriged_magnitudeII[day] = rainfall_grid

    # Visualization
    plot_variograms_magnitude(variogram_results_by_category)
    plot_qq_transformations(trans_results_by_category)

    # Output
    magnitude_dsII = create_magnitude_dataset(kriged_magnitudeII, grid_x, grid_y)
    magnitude_dsII.to_netcdf("rainfall_magnitudeII.nc")
    save_phase2_results(
        "phase2_results",
        kriged_magnitude=kriged_magnitudeII,
        magnitude_ds=magnitude_dsII
    )

    # Example visualization
    plot_four_panels(
        date_target=date(2022, 6, 7),
        daily_df=daily_df,
        kriged_maps=kriged_maps,
        kriged_magnitude=kriged_magnitudeII,
        grid_x=grid_x,
        grid_y=grid_y,
        land_mask=land_mask,
        shapefile_path=SHAPEFILE_PATH,
        save_path="four_panel_plot.png"
    )
    
    print("Phase II processing complete!")

if __name__ == "__main__":
    # Load required data before running main()
    # daily_df, class_dict, kriged_maps, final_group_days = load_data()
    main()
