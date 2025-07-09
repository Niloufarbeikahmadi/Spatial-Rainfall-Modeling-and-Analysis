# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 11:27:14 2025

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from tqdm import tqdm
import geopandas as gpd
import pandas as pd
from pyproj import Transformer
from typing import Dict, List, Tuple,Any
import gstools as gs
from datetime import datetime, date
import warnings
from pykrige.ok import OrdinaryKriging
from pyproj import Geod
from shapely.geometry import Point
from rtree import index
from shapely.ops import unary_union
from scipy.optimize import brentq
from scipy.stats import gaussian_kde
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Patch
import os

# -----------------------------------------------------------------------------
# Loadings
# -----------------------------------------------------------------------------
def load_multiple_pickles(file_list):
    for file_name in file_list:
        globals()[file_name.split('.')[0]] = load_pickle(file_name)

file_names = ["daily_df.pkl"]
data = load_multiple_pickles(file_names)
shapefile_path=r"C:/Users/..../sicily.shp"

# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

# Coordinate reprojection

def reproject_df(
    df: pd.DataFrame,
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    src_epsg: int = 4326,
    dst_epsg: int = 32633,
) -> pd.DataFrame:
    """Return *df* with metre coords (EPSG:3857)."""
    transformer = Transformer.from_crs(src_epsg, dst_epsg, always_xy=True)
    x, y = transformer.transform(df[lon_col].to_numpy(), df[lat_col].to_numpy())
    out = df.copy()
    out[lon_col] = x
    out[lat_col] = y
    return out

# Calculating grid resolutions in meter
def create_grid_in_meters(
    lon_min_deg: float,
    lon_max_deg: float,
    lat_min_deg: float,
    lat_max_deg: float,
    resolution_deg: float = 0.02
) -> tuple:
    """
    Creates a grid in meters that approximates the specified geographic resolution.
    
    Args:
        lon_min_deg: Minimum longitude (degrees)
        lon_max_deg: Maximum longitude (degrees)
        lat_min_deg: Minimum latitude (degrees)
        lat_max_deg: Maximum latitude (degrees)
        resolution_deg: Target geographic resolution in degrees
        
    Returns:
        grid_x_m: X-coordinates in meters (1D array)
        grid_y_m: Y-coordinates in meters (1D array)
        transformer: Pyproj transformer for coordinate conversion
    """
    # Initialize geodetic calculator (WGS84 ellipsoid)
    geod = Geod(ellps="WGS84")
    
    # Calculate center latitude for distance conversion
    center_lat = (lat_min_deg + lat_max_deg) / 2
    
    # Calculate meters per degree at center latitude
    _, _, dx = geod.inv(lon_min_deg, center_lat, 
                        lon_min_deg + resolution_deg, center_lat)
    
    # For latitude distance at center longitude
    _, _, dy = geod.inv(lon_min_deg, center_lat,
                        lon_min_deg, center_lat + resolution_deg)
    # Create transformer # UTM 33N
    transformer = Transformer.from_crs(4326, 32633, always_xy=True)  
    
    # Transform bounds to meters
    x_min_m, y_min_m = transformer.transform(lon_min_deg, lat_min_deg)
    x_max_m, y_max_m = transformer.transform(lon_max_deg, lat_max_deg)
    
    # Create grid with calculated resolution
    grid_x = np.arange(x_min_m, x_max_m + dx, dx)
    grid_y = np.arange(y_min_m, y_max_m + dy, dy)
    
   
    return grid_x, grid_y, transformer



# Rain‑presence binary indicator
def binary_indicator(df: pd.DataFrame, rain_col: str = "Rain") -> pd.DataFrame:
    out = df.copy()
    out["indicator"] = (out[rain_col] > 0).astype(int)
    return out

#Daily dry‑spell classification
def classify_days(
    df: pd.DataFrame,
    thresholds: List[Tuple[float, float]] = [(0, 25), (25, 75), (75, 100)],
    rain_col: str = "Rain",
    day_col: str = "day",
) -> Dict[str, List[datetime.date]]:
    zero_pct = df.groupby(day_col).apply(lambda g: (g[rain_col] == 0).sum()/len(g)*100)
    res: Dict[str, List[datetime.date]] = {}
    for lo, hi in thresholds:
        lbl = f"F{int(lo)}-{int(hi)}"
        res[lbl] = zero_pct[(zero_pct >= lo) & (zero_pct <= hi)].index.to_list()
    return res

# -----------------------------------------------------------------------------
# Variogram helpers
# -----------------------------------------------------------------------------

def _aic_weighted(residuals: np.ndarray, weights: np.ndarray, k: int) -> float:
    """Weighted AIC (pair‑count weights)."""
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
    """Fit each GSTools model with count weights; return all converged models and best."""
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
            warnings.warn(f"{name} fit failed to converge; skipping.")
            continue
            
        pred = mod.variogram(bins)
        aic = _aic_weighted(gamma - pred, counts, k=len(fit_para))
        fit_data = {
            "name": name,
            "model": mod,
            "params": fit_para,
            "aic": aic
        }
        all_fits.append(fit_data)
        
        if aic < best_aic:
            best_fit = fit_data
            best_aic = aic

    if not all_fits:
        raise RuntimeError("No variogram model converged for this class.")
        
    return all_fits, best_fit

# Make GSTool's parameters and variogram func prepared for Pykrige
def create_gstools_variogram(model_name: str, params_dict: dict):
    # Correct parameter order mapping for PyKrige
    PARAM_ORDER = {
        "Gaussian": ["sill", "range", "nugget"],
        "Exponential": ["sill", "range", "nugget"],
        "Matern": ["sill", "range", "nugget", "nu"],
        "Stable": ["sill", "range", "nugget", "alpha"],
        "Rational": ["sill", "range", "nugget", "alpha"],
        "Spherical": ["sill", "range", "nugget"],
        "SuperSpherical": ["sill", "range", "nugget", "alpha"],
        "JBessel": ["sill", "range", "nugget", "alpha"]
    }

    # Validate model support
    if model_name not in PARAM_ORDER:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Extract GSTools parameters
    nugget = params_dict["nugget"]
    var = params_dict["var"]  # Partial sill
    len_scale = params_dict["len_scale"]
    full_sill = nugget + var  # PyKrige requires full sill
    
    # Handle optional parameters
    opt_params = {}
    if model_name == "Matern":
        opt_params["nu"] = params_dict["nu"]
    elif model_name in ["Stable", "Rational", "SuperSpherical", "JBessel"]:
        opt_params["alpha"] = params_dict["alpha"]
    
    # Create parameter list in PyKrige order
    param_list = [full_sill, len_scale, nugget]
    if model_name in PARAM_ORDER and len(PARAM_ORDER[model_name]) > 3:
        param_list.append(opt_params[list(opt_params.keys())[0]])
    
    # Create variogram function
    def variogram_function(params, dist):
        """Adapts GSTools model to PyKrige interface"""
        # Create GSTools model instance
        model_class = getattr(gs, model_name)
        model = model_class(
            dim=2,
            nugget=params[2],
            var=params[0] - params[2],  # Partial sill = sill - nugget
            len_scale=params[1],
            **opt_params
        )
        return model.variogram(dist)
    
    return variogram_function, param_list

#  R-tree version mask of inetrpolation area

def create_sicily_mask_rtree(shapefile_path, grid_x, grid_y, buffer_meters=2000):
    sicily = gpd.read_file(shapefile_path).to_crs(epsg=32633)
    buffered = unary_union(sicily.geometry).buffer(buffer_meters)
    
    # Create spatial index
    idx = index.Index()
    for i, (x, y) in enumerate(np.nditer(np.meshgrid(grid_x, grid_y))):
        idx.insert(i, (x, y, x, y))
    
    # Generate mask
    mask = np.zeros((len(grid_y), len(grid_x)), dtype=bool)
    for j in range(len(grid_y)):
        for i in range(len(grid_x)):
            x, y = grid_x[i], grid_y[j]
            mask[j,i] = buffered.contains(Point(x,y))
    
    return ~mask

# -----------------------------------------------------------------------------
# Daily‑pooled variogram by class
# -----------------------------------------------------------------------------

def variogram_by_class(
    df: pd.DataFrame,
    class_dict: Dict[str, List[datetime.date]],
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    ind_col: str = "indicator",
    maxlag_deg: float = 2.0,
    n_lags: int = 20,
    chunk_size: int = 25000,
    plot: bool = True,
) -> Dict[str, Dict[str, object]]:
    """Compute **daily‑averaged** variograms and fit GSTools models per class."""

    # Fixed lag structure
    DEG_TO_M = 111000.0
    bin_edges = np.linspace(0.0, maxlag_deg * DEG_TO_M, n_lags)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    n_bins = len(bin_centers)

    # Model catalogue
    gs_models = {
        "Gaussian": gs.Gaussian,
        "Exponential": gs.Exponential,
        "Matern": gs.Matern,
        "Stable": gs.Stable,
        "Rational": gs.Rational,
        "Circular": gs.Circular,
        "Spherical": gs.Spherical,
        "SuperSpherical": gs.SuperSpherical,
        "JBessel": gs.JBessel,
    }

    results: Dict[str, Dict[str, object]] = {}
    fig, axes = (plt.subplots(1, 3, figsize=(15, 4)) if plot else (None, [None] * 3))
    max_gamma = 0 
    for ax, (label, days) in zip(axes, tqdm(class_dict.items(), desc="Variogram classes")):
        # Accumulate daily sums
        sum_gamma = np.zeros(n_bins)
        sum_counts = np.zeros(n_bins)

        for day in days:
            sub = df[df["day"] == day]
            if len(sub) < 2:
                continue
            try:
                x = sub[lon_col].to_numpy()
                y = sub[lat_col].to_numpy()
                vals = sub[ind_col].to_numpy()
    
                bins_d, gamma_d, counts_d = gs.vario_estimate(
                    (x,y),
                    vals,
                    bin_edges=bin_edges,
                    sampling_size=chunk_size,
                    estimator="matheron",
                    latlon=False,
                    return_counts=True,
                )
            except ValueError as err:
                warnings.warn(f"{day}: {err}; skipped.")
                continue
            # Ensure NaNs replaced with 0 before weighting
            gamma_d = np.nan_to_num(gamma_d)
            sum_gamma += gamma_d * counts_d
            sum_counts += counts_d

        valid = sum_counts > 0
        if not valid.any():
            warnings.warn(f"Class {label}: no variogram pairs found; skipping.")
            continue

        gamma_mean = np.zeros(n_bins)
        gamma_mean[valid] = sum_gamma[valid] / sum_counts[valid]

        # Fit models using pair‑count weights
        all_fits, best_fit  = _fit_models(bin_centers, gamma_mean, sum_counts, gs_models)
        results[label] = {
                    "all_models": all_fits,
                    "best_model": best_fit["name"],
                    "params": best_fit["params"],
                    "AIC": best_fit["aic"]
                }        
        # Plot
        if plot and ax is not None:
           # Empirical data
           ax.scatter(bin_centers[valid], gamma_mean[valid], s=20, 
                      c='k', label="Empirical", zorder=10)
           
           # Model curves
           dist_line = np.linspace(0, bin_centers[-1]*1.2, 200)
           for fit in all_fits:
               style = '-' if fit["name"] == best_fit["name"] else '--'
               y_vals = fit["model"].variogram(dist_line)
               ax.plot(dist_line, y_vals, label=fit["name"], linestyle=style)
               
               # Update global maximum gamma
               max_gamma = max(max_gamma, np.max(y_vals), np.max(gamma_mean[valid]))
           
           ax.set_title(f"{label} Variogram (Best: {best_fit['name']})")
           ax.set_xlabel("Distance [m]")
           ax.set_ylabel("γ(h)")
           ax.legend(frameon=False, fontsize=9)
           ax.grid(alpha=0.3)
   
   # Apply uniform y-axis limits to all subplots
    if plot:
       for ax in axes:
           if ax.has_data():  # Only adjust if subplot has data
               ax.set_ylim(0, max_gamma * 1.05)  # 5% padding above max value
               
       plt.tight_layout()
       plt.show()

    return results

# -----------------------------------------------------------------------------
# OK over 2km on frequency classes
# -----------------------------------------------------------------------------

def krige_with_best_variogram(
    df: pd.DataFrame,
    class_dict: Dict[str, List[date]],
    summary: Dict[str, Dict[str, object]],
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    ind_col: str = "indicator",
    resolution_deg: float = 0.02,
) -> Dict[date, Dict[str, object]]:
    
    # Create day-to-class mapping dictionary
    
    day_to_class = {d: lbl for lbl, days in class_dict.items() for d in days}
    
    # Define geographic bounds
    lon_min, lon_max = 12.0, 16.0
    lat_min, lat_max = 36.5, 38.5
    
    # Create geographic grid
    grid_x_m, grid_y_m, transformer = create_grid_in_meters(
        lon_min, lon_max, lat_min, lat_max, resolution_deg
    )
    
    # a tree-based BOOL MASK for inland interpolation: 
    path=r"C:/Users/Utente/.../sicily.shp"
    mask_array = create_sicily_mask_rtree(path, grid_x_m, grid_y_m, buffer_meters=2000)
     

    # Dictionary to store results for each day
    results: Dict[date, Dict[str, object]] = {}
    
    # Process each unique day in the dataframe
    for day in tqdm(sorted(df['day'].unique()), desc="Kriging days"):
        # Filter data for current day and remove duplicate locations
        sub = df[df['day'] == day].drop_duplicates(subset=[lon_col, lat_col])
        vals = sub[ind_col].values.astype(float)

        # Handle constant-value days
        if sub[ind_col].nunique() == 1:
            val = vals[0]
            results[day] = {
                'grid_x': grid_x_m,
                'grid_y': grid_y_m,
                'prob_map': np.full((len(grid_y_m), len(grid_x_m)), val)
            }
            continue

        # Skip days with insufficient data (less than 2 points)
        if len(sub) < 2:
            warnings.warn(f"{day}: <2 stations, skipping")
            continue
            
        # Get classification label for this day
        class_label = day_to_class.get(day)
        if class_label not in summary:
            warnings.warn(f"{day}: class not found in summary")
            continue
        
        # Extract model information from summary
        model_name = summary[class_label]['best_model']
        params_dict = summary[class_label]['params']
        
        # Create variogram components
        try:
            variogram_func, variogram_params = create_gstools_variogram(
                model_name, params_dict
            )
        except Exception as e:
            warnings.warn(f"Variogram setup failed for {day}: {str(e)}")
            continue
            
        # Create OrdinaryKriging with custom variogram
        
        try: 
            OK = OrdinaryKriging(
                sub[lon_col].values,
                sub[lat_col].values,
                vals,
                variogram_model='custom',
                variogram_function=variogram_func,  
                variogram_parameters=variogram_params, 
                enable_plotting=False,
                verbose=False,
                coordinates_type='euclidean',  # Input in x/y meter
                exact_values=True
            )
            
            # Execute kriging on grid

            z, ss = OK.execute(style='masked', xpoints=grid_x_m, 
                               ypoints=grid_y_m, mask=mask_array)
                                    
            # Ensure probabilities stay between 0-1
            z = np.clip(z, 0, 1)
            
            # Store results
            results[day] = {
                'grid_x': grid_x_m,
                'grid_y': grid_y_m,
                'prob_map': z
            }
            
        except Exception as e:
            warnings.warn(f"Kriging failed for {day}: {str(e)}")
            
    return results
  
# -----------------------------------------------------------------------------
# Calculating classified binary transformation cutoff threshold 
# -----------------------------------------------------------------------------

#1
def calculate_weighted_cutoffs(kriged_maps, class_dict, daily_df, grid_x, grid_y):
    """Calculate thresholds with spatial weighting based on gauge density"""
    thresholds = {}
    
    # Prepare gauge points for KDE
    gauge_points = daily_df[['Longitude', 'Latitude']].drop_duplicates().values
    
    # Handle case with too few gauges
    if len(gauge_points) < 2:
        for class_label in class_dict:
            thresholds[class_label] = 0.5
        return thresholds
    
    # Calculate KDE for gauge density
    try:
        kde = gaussian_kde(gauge_points.T, bw_method='scott')
    except:
        # Fallback to uniform weighting
        kde = None
    
    # Create grid for weights
    xx, yy = np.meshgrid(grid_x, grid_y)
    grid_points = np.vstack([xx.ravel(), yy.ravel()])
    
    # Compute weights (density estimates)
    if kde:
        weights = kde(grid_points).reshape(xx.shape)
    else:
        weights = np.ones(xx.shape)
    
    # Normalize weights
    weights /= np.nansum(weights)
    
    for class_label, class_dates in class_dict.items():
        valid_dates = [d for d in class_dates if d in kriged_maps]
        if not valid_dates:
            thresholds[class_label] = 0.5
            continue
            
        # Calculate target mean from observations
        obs_indicators = daily_df[daily_df['day'].isin(valid_dates)]['indicator']
        target_mean = np.nanmean(obs_indicators)
        
        # Calculate weighted mean of probabilities
        weighted_sum = 0
        valid_weight_sum = 0
        
        for date in valid_dates:
            prob_map = kriged_maps[date]['prob_map']
            # Create mask for valid (land) cells
            valid_mask = ~np.isnan(prob_map)
            valid_probs = prob_map[valid_mask]
            valid_weights = weights[valid_mask]
            
            # Normalize weights for this map
            map_weight_sum = np.nansum(valid_weights)
            if map_weight_sum > 0:
                valid_weights /= map_weight_sum
                weighted_sum += np.nansum(valid_probs * valid_weights)
                valid_weight_sum += 1
        
        if valid_weight_sum == 0:
            grid_mean = target_mean
        else:
            grid_mean = weighted_sum / valid_weight_sum
        
        # Set threshold (since grid_mean = target_mean by construction)
        thresholds[class_label] = grid_mean
    
    return thresholds
#2(choosen):
def calculate_cutoff_thresholds(kriged_maps, class_dict, daily_df):
    """
    Calculate optimal cutoff thresholds for each frequency class, handling NaN values
    
    Args:
        kriged_maps: Dict of kriging results (date: {grid_lons, grid_lats, prob_map})
        class_dict: Frequency class mapping (class: [dates])
        daily_df: DataFrame with observed data (columns: 'day', 'indicator')
    
    Returns:
        dict: Cutoff thresholds for each frequency class
    """
    thresholds = {}
    
    for class_label, class_dates in class_dict.items():
        # Filter dates that exist in both class_dict and kriged_maps
        valid_dates = [d for d in class_dates if d in kriged_maps]
        
        if not valid_dates:
            warnings.warn(f"No valid dates for class {class_label}")
            thresholds[class_label] = 0.5  # Default fallback
            continue
        
        # Calculate observed mean (target value) for the class
        obs_indicators = []
        for date in valid_dates:
            day_data = daily_df[daily_df['day'] == date]
            obs_indicators.extend(day_data['indicator'].values)
        target_mean = np.nanmean(obs_indicators)
        
        # Collect all kriged probabilities for this class, ignoring NaNs
        all_probs = []
        for date in valid_dates:
            prob_map = kriged_maps[date]['prob_map']
            # Flatten and remove NaNs
            flat_probs = prob_map[~np.isnan(prob_map)]
            all_probs.extend(flat_probs)
        
        # If no valid probabilities, use default
        if len(all_probs) == 0:
            warnings.warn(f"No valid probabilities for class {class_label}")
            thresholds[class_label] = 0.5
            continue
            
        all_probs = np.array(all_probs)
        
        # Handle edge cases
        if target_mean <= 0 or np.isnan(target_mean):
            thresholds[class_label] = 1.0  # All dry
            continue
        if target_mean >= 1:
            thresholds[class_label] = 0.0  # All wet
            continue
            
        # Optimization function to find threshold (ignores NaNs)
        def f(threshold):
            binary_map = (all_probs >= threshold).astype(float)
            return np.nanmean(binary_map) - target_mean
            
        try:
            # Find root where f(threshold) = 0
            threshold = brentq(f, 0, 1, xtol=1e-4)
            thresholds[class_label] = threshold
        except ValueError:
            # Fallback to quantile method if root finding fails
            try:
                quantile = 100 * (1 - target_mean)
                thresholds[class_label] = np.nanpercentile(all_probs, quantile)
            except:
                thresholds[class_label] = 0.5  # Final fallback
    
    return thresholds     
# -----------------------------------------------------------------------------
# Creating binary occurence maps 
# -----------------------------------------------------------------------------

def add_binary_maps(kriged_maps, class_dict, cutoffs):
    """
    Adds binary rain/no-rain maps to kriged results based on class-specific thresholds
    
    Args:
        kriged_maps: Dict of kriging results (date: {grid_lons, grid_lats, prob_map})
        class_dict: Frequency class mapping (class: [dates])
        cutoffs: Dictionary of thresholds for each class
        
    Returns:
        Updated kriged_maps with 'binary_map' added for each date
    """
    # Create date-to-class mapping
    date_to_class = {}
    for class_label, dates in class_dict.items():
        for date in dates:
            date_to_class[date] = class_label
    
    # Process each date with progress bar
    for date, data in tqdm(kriged_maps.items(), desc="Creating binary maps"):
        # Get class and corresponding threshold
        class_label = date_to_class.get(date)
        if class_label is None:
            warnings.warn(f"No class found for date {date}, using default threshold 0.5")
            threshold = 0.5
        else:
            threshold = cutoffs.get(class_label, 0.5)
        
        prob_map = data['prob_map']
        
        # Initialize binary map with NaNs (same shape as prob_map)
        binary_map = np.full_like(prob_map, np.nan)
        
        # Create mask for land cells (non-NaN)
        land_mask = ~np.isnan(prob_map)
        
        # Apply threshold only to land cells
        binary_map[land_mask] = np.where(
            prob_map[land_mask] >= threshold, 
            1,  # Rain
            0   # No rain
        )
        
        # Add binary map to results
        data['binary_map'] = binary_map
    
    return kriged_maps

# -----------------------------------------------------------------------------
# showcase random date
# -----------------------------------------------------------------------------
def plot_date_results(date_input, kriged_maps, daily_df, shapefile_path):
    """
    Create 4-panel plot for a specific date with corrected colors
    
    Args:
        date_input: Date input (string, datetime.date, or datetime.datetime)
        kriged_maps: Dictionary containing gridded results
        daily_df: DataFrame with point observations
        shapefile_path: Path to Sicily shapefile (.shp)
    """
    # Convert input to date object
    if isinstance(date_input, str):
        try:
            date_obj = datetime.strptime(date_input, '%Y-%m-%d').date()
        except ValueError:
            try:
                date_obj = datetime.strptime(date_input, '%Y/%m/%d').date()
            except:
                raise ValueError(f"Unsupported date format: {date_input}")
    elif isinstance(date_input, datetime):
        date_obj = date_input.date()
    elif isinstance(date_input, date):
        date_obj = date_input
    else:
        raise TypeError(f"Unsupported date type: {type(date_input)}")
    
    # Find date in kriged_maps
    date_found = False
    for key in kriged_maps.keys():
        if key == date_obj or key == date_obj.strftime('%Y-%m-%d') or key == str(date_obj):
            date_key = key
            date_found = True
            break
    
    if not date_found:
        # Try all keys to find matching date
        for key in kriged_maps.keys():
            if isinstance(key, date) and key == date_obj:
                date_key = key
                date_found = True
                break
            elif isinstance(key, str):
                try:
                    key_date = datetime.strptime(key, '%Y-%m-%d').date()
                    if key_date == date_obj:
                        date_key = key
                        date_found = True
                        break
                except:
                    continue
    
    if not date_found:
        available_dates = list(kriged_maps.keys())[:5]  # Show first 5
        raise ValueError(f"Date {date_obj} not found in kriged_maps. Available: {available_dates}")
    
    # Get data for date
    data = kriged_maps[date_key]
    
    # Get daily data
    date_str1 = date_obj.strftime('%Y-%m-%d')
    date_str2 = date_obj.strftime('%Y/%m/%d')
    date_str3 = str(date_obj)
    day_data = daily_df[
        (daily_df['day'] == date_obj) | 
        (daily_df['day'] == date_str1) | 
        (daily_df['day'] == date_str2) | 
        (daily_df['day'] == date_str3)
    ]
    
    if day_data.empty:
        warnings.warn(f"No point data found for date {date_obj}")
    
    # Setup coordinate transformers
    utm_to_deg = Transformer.from_crs(32633, 4326, always_xy=True)
    
    # Convert grid to geographic coordinates
    xx_m, yy_m = np.meshgrid(data['grid_x'], data['grid_y'])
    lons, lats = utm_to_deg.transform(xx_m, yy_m)
    
    # Convert point locations to geographic if available
    if not day_data.empty:
        point_lons, point_lats = utm_to_deg.transform(
            day_data['Longitude'].values,
            day_data['Latitude'].values
        )
    else:
        point_lons, point_lats = np.array([]), np.array([])
    
    # Load and prepare Sicily shapefile
    try:
        sicily = gpd.read_file(shapefile_path)
        sicily = sicily.to_crs(epsg=4326)  # Ensure geographic CRS
        minx, miny, maxx, maxy = sicily.total_bounds
    except Exception as e:
        warnings.warn(f"Error loading shapefile: {str(e)}")
        minx, miny = np.min(lons), np.min(lats)
        maxx, maxy = np.max(lons), np.max(lats)
        sicily = None
    
    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
    fig.suptitle(f"Rainfall Analysis - {date_obj}", fontsize=16)
    
    # 1. Upper Left: Rainfall Occurrence Points 
    ax = axs[0, 0]
    if not day_data.empty:
        # Create discrete colormap: red for dry (0), blue for wet (1)
        cmap_occurrence = mcolors.ListedColormap(['red', 'blue'])
        
        sc = ax.scatter(
            point_lons, point_lats, 
            c=day_data['indicator'], 
            cmap=cmap_occurrence,
            s=30, 
            edgecolor='k',
            vmin=0, vmax=1
        )
        
        # Create legend inside the plot
        legend_elements = [
            Patch(facecolor='red', edgecolor='k', label='Dry'),
            Patch(facecolor='blue', edgecolor='k', label='Wet')
        ]
        ax.legend(handles=legend_elements, loc='best')
    if sicily is not None:
        sicily.boundary.plot(ax=ax, color='k', linewidth=1)
    ax.set_title("Point Rainfall Occurrence")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    
    # 2. Upper Right: Rainfall Magnitude Points
    ax = axs[0, 1]
    if not day_data.empty:
        # Use RdYlBu colormap for rainfall magnitude
        cmap = plt.get_cmap('RdYlBu')  
        sc = ax.scatter(
            point_lons, point_lats, 
            c=day_data['Rain'], 
            cmap=cmap,
            s=30, 
            edgecolor='k'
        )
        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(sc, cax=cax, extend='max')
        cbar.set_label('Rainfall (mm)')
    if sicily is not None:
        sicily.boundary.plot(ax=ax, color='k', linewidth=1)
    ax.set_title("Rainfall Magnitude (mm)")
    
    # 3. Lower Left: Probability Map (CORRECTED to RdYlBu)
    ax = axs[1, 0]
    # Use RdYlBu colormap for probability
    pm = ax.pcolormesh(
        lons, lats, data['prob_map'],
        cmap='RdYlBu',
        shading='auto',
        vmin=0, vmax=1
    )
    if sicily is not None:
        sicily.boundary.plot(ax=ax, color='k', linewidth=1)
    ax.set_title("Probability of Rainfall Occurrence")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(pm, cax=cax, label='Probability')
    
    # 4. Lower Right: Binary Map (CORRECTED colors)
    ax = axs[1, 1]
    
    # Create discrete colormap: red for dry (0), blue for wet (1)
    cmap_binary = mcolors.ListedColormap(['red', 'blue'])
    bounds = [-0.5, 0.5, 1.5]
    norm_binary = mcolors.BoundaryNorm(bounds, cmap_binary.N)
    
    bm = ax.pcolormesh(
        lons, lats, data['binary_map'],
        cmap=cmap_binary,
        norm=norm_binary,
        shading='auto'
    )
    if sicily is not None:
        sicily.boundary.plot(ax=ax, color='k', linewidth=1)
    ax.set_title("Binary Rainfall Occurrence")
    
    # Create custom legend for binary map
    legend_elements = [
        Patch(facecolor='red', edgecolor='k', label='Dry'),
        Patch(facecolor='blue', edgecolor='k', label='Wet'),
        Patch(facecolor='white', edgecolor='k', label='Sea (NaN)')
    ]
    ax.legend(handles=legend_elements, loc='best')
    
    # Set consistent extent
    buffer = 0.2  # degrees
    for ax in axs.flat:
        ax.set_xlim(minx - buffer, maxx + buffer)
        ax.set_ylim(miny - buffer, maxy + buffer)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    return fig

# -----------------------------------------------------------------------------
# Creating XARRAY and savings
# -----------------------------------------------------------------------------

def create_xarray_dataset(kriged_maps):
    """
    Convert kriged_maps dictionary into an xarray Dataset handling masked arrays
    
    Args:
        kriged_maps: Dictionary of kriging results by date
        
    Returns:
        xarray.Dataset with dimensions (time, y, x)
    """
    # Sort dates chronologically
    sorted_dates = sorted(kriged_maps.keys())
    
    # Get grid dimensions from first date
    first_date = sorted_dates[0]
    grid_x = kriged_maps[first_date]['grid_x']
    grid_y = kriged_maps[first_date]['grid_y']
    
    # Initialize arrays
    time_coords = []
    prob_stack = []
    binary_stack = []
    
    # Process each date
    for date in tqdm(sorted_dates, desc="Creating xarray Dataset"):
        data = kriged_maps[date]
        time_coords.append(date)
        
        # Ensure grid consistency
        assert np.array_equal(data['grid_x'], grid_x), "X-grid mismatch"
        assert np.array_equal(data['grid_y'], grid_y), "Y-grid mismatch"
        
        # Convert masked arrays to regular arrays with NaN values
        if isinstance(data['prob_map'], np.ma.MaskedArray):
            prob_map = data['prob_map'].filled(np.nan)
        else:
            prob_map = data['prob_map']
            
        if isinstance(data['binary_map'], np.ma.MaskedArray):
            binary_map = data['binary_map'].filled(np.nan)
        else:
            binary_map = data['binary_map']
        
        # Store processed arrays
        prob_stack.append(prob_map)
        binary_stack.append(binary_map)
    
    # Convert to numpy arrays
    prob_stack = np.array(prob_stack)
    binary_stack = np.array(binary_stack)
    
    # Create land mask from NaN positions
    land_mask = np.where(np.isnan(prob_stack[0]), 0, 1).astype(np.int8)
    
    # Create xarray Dataset
    ds = xr.Dataset(
        {
            "prob_map": (("time", "y", "x"), prob_stack),
            "binary_map": (("time", "y", "x"), binary_stack),
            "land_mask": (("y", "x"), land_mask),
        },
        coords={
            "time": pd.to_datetime(time_coords),
            "x": grid_x,
            "y": grid_y,
        }
    )
    
    # Add attributes
    ds.x.attrs = {"units": "meters", "crs": "EPSG:32633"}
    ds.y.attrs = {"units": "meters", "crs": "EPSG:32633"}
    ds.time.attrs = {"long_name": "Date"}
    ds.prob_map.attrs = {"long_name": "Rainfall occurrence probability", "units": "0-1"}
    ds.binary_map.attrs = {"long_name": "Rainfall occurrence (0=dry, 1=wet)", "units": "binary"}
    ds.land_mask.attrs = {"long_name": "Land mask (1=land, 0=sea)"}
    
    return ds

def save_to_netcdf(ds, output_path):
    """
    Save xarray Dataset to NetCDF file with optimized encoding and
    integer fill values for NaNs.
    """
    encoding = {
        'prob_map': {
            'dtype': 'float32',
            '_FillValue': np.float32(np.nan)  # float can use NaN as fill
        },
        'binary_map': {
            'dtype': 'int8',
            '_FillValue': np.int8(-1)         # -1 will mark masked points
        },
        'land_mask': {
            'dtype': 'int8',
            '_FillValue': np.int8(0)          # sea already 0, so safe
        },
        'x': {'dtype': 'float32'},
        'y': {'dtype': 'float32'}
    }
    ds['binary_map'] = ds['binary_map'].fillna(encoding['binary_map']['_FillValue'])
    # Now call to_netcdf
    ds.to_netcdf(output_path, encoding=encoding)
    print(f"Dataset saved to {output_path!r}")
        
def save_analysis_data(path_save: str, **kwargs: Any) -> None:
    """
    Save multiple analysis objects to disk, preserving original names and types.

    Objects supported:
      - numpy arrays: saved as .npy
      - other objects: pickled as .pkl

    Args:
        path_save: Directory to save files. Will be created if it doesn't exist.
        **kwargs: Named objects to save, e.g. mask=mask, class_dict=class_dict
    """
    os.makedirs(path_save, exist_ok=True)
    for name, obj in kwargs.items():
        filepath = os.path.join(path_save, f"{name}")
        if isinstance(obj, np.ndarray):
            # Save numpy array
            np.save(filepath + ".npy", obj)
        else:
            # Save with pickle
            with open(filepath + ".pkl", "wb") as f:
                pickle.dump(obj, f)
# -----------------------------------------------------------------------------
# RUN Commanads
# -----------------------------------------------------------------------------
daily_df=reproject_df(daily_df)
daily_df = binary_indicator(daily_df)
class_dict = classify_days(daily_df)
summary = variogram_by_class(daily_df, class_dict)
print(summary)
kriged_maps = krige_with_best_variogram(daily_df, class_dict, summary, 
                                        resolution_deg=0.02)

cutoffs = calculate_cutoff_thresholds(
    kriged_maps=kriged_maps,
    class_dict=class_dict,
    daily_df=daily_df)
#cutoffs_weighted = calculate_weighted_cutoffs(kriged_maps, class_dict, daily_df,grid_x,grid_y)
kriged_maps = add_binary_maps(
    kriged_maps=kriged_maps,
    class_dict=class_dict,
    cutoffs=cutoffs
)

fig = plot_date_results(date(2022, 2, 5), kriged_maps, daily_df, shapefile_path)
occurrence =  create_xarray_dataset(kriged_maps)
#
save_to_netcdf(occurrence, "rainfall_occurrence.nc")

save_analysis_data(
    "C:/Users/Utente/.../saves",
    #mask=mask,
    class_dict=class_dict,
    cutoffs=cutoffs,
    kriged_maps=kriged_maps,
    summary=summary
)

