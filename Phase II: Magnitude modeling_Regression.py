# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 16:27:49 2025

@author: Utente
"""

# ----------------------------------------
# Imports
# Core libraries and geospatial/data-science modules
# ----------------------------------------
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from scipy.interpolate import griddata
from sklearn.preprocessing import MinMaxScaler
from pygam import GammaGAM, s, f, te
from gstools import vario_estimate, Gaussian, Exponential, Matern, Stable
from pykrige import OrdinaryKriging
from scipy.stats import probplot, shapiro
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from scipy.spatial import cKDTree
import gstools as gs
import geopandas as gpd
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import defaultdict
import joblib
import rioxarray
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from joblib import Parallel, delayed
from sklearn.base import clone
from IPython.display import display
from datetime import datetime

# ----------------------------------------
# Data Loading
# Load geospatial arrays and station data
# ----------------------------------------
# Geometry and input rasters
dem = xr.open_dataarray("dem.nc").rio.write_crs("EPSG:32633")
distance = xr.open_dataarray("distance.nc").rio.write_crs("EPSG:32633")
ds = xr.open_dataset("era5_downscaled.nc").rio.write_crs("EPSG:32633")
cape = ds['cape'].rio.write_crs("EPSG:32633")
vidmf = ds['vidmf'].rio.write_crs("EPSG:32633")
land_mask = ds['land_mask'].rio.write_crs("EPSG:32633")
occurrence = xr.open_dataset("rainfall_occurrence.nc").rio.write_crs("EPSG:32633")
mag = xr.open_dataarray("rainfall_magnitudeII.nc").rio.write_crs("EPSG:32633")

# Station and group data
file_names = ["daily_df.pkl", 'final_group_daysII.pkl']
data = load_multiple_pickles(file_names)
daily_df = reproject_df(daily_df)  # station data
final_group_days = final_group_daysII

# ----------------------------------------
# Helper Functions
# Core utilities for extraction and variogram
# ----------------------------------------
def extract_grid_value(grid_data, points_x, points_y, time=None, max_dist=7000):
    """Extract grid values at point locations by finding the closest NON-NaN cell"""
    if 'time' in grid_data.dims and time is not None:
        grid_data = grid_data.sel(time=time, method='nearest')
    grid_x = grid_data.x.values
    grid_y = grid_data.y.values
    X_grid, Y_grid = np.meshgrid(grid_x, grid_y)
    grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
    if grid_data.ndim == 2:
        values_flat = grid_data.values.ravel()
    else:
        values_flat = grid_data.values[0].ravel() if 'time' in grid_data.dims else grid_data.values.ravel()
    valid_mask = ~np.isnan(values_flat)
    valid_points = grid_points[valid_mask]
    valid_values = values_flat[valid_mask]
    if len(valid_points) == 0:
        return np.full(len(points_x), np.nan)
    tree = cKDTree(valid_points)
    distances, indices = tree.query(np.column_stack([points_x, points_y]), k=1, distance_upper_bound=max_dist)
    values = np.full(len(points_x), np.nan)
    valid_queries = distances < np.inf
    values[valid_queries] = valid_values[indices[valid_queries]]
    if np.any(~valid_queries):
        print(f"Warning: {np.sum(~valid_queries)} points exceed max_dist ({max_dist}m) from valid grid cells")
    return values


def create_variogram_from_model(vario_model):
    """Create GSTools variogram model from fitted parameters"""
    model_type = type(vario_model).__name__
    params = {"nugget": vario_model.nugget, "var": vario_model.var, "len_scale": vario_model.len_scale}
    if model_type in ["Matern"]:
        params["nu"] = vario_model.nu
    elif model_type in ["Stable"]:
        params["alpha"] = vario_model.alpha
    if model_type == "Gaussian":
        return Gaussian(dim=2, **params)
    elif model_type == "Exponential":
        return Exponential(dim=2, **params)
    elif model_type == "Matern":
        return Matern(dim=2, **params)
    elif model_type == "Stable":
        return Stable(dim=2, **params)


def extract_variogram_parameters(vario_model):
    """Extract variogram parameters in PyKrige-compatible order"""
    model_type = type(vario_model).__name__
    params = {"nugget": vario_model.nugget, "partial_sill": vario_model.var, "range": vario_model.len_scale}
    if model_type == "Matern":
        return [params["nugget"], params["partial_sill"], params["range"], vario_model.nu]
    elif model_type == "Stable":
        return [params["nugget"], params["partial_sill"], params["range"], vario_model.alpha]
    else:
        return [params["nugget"], params["partial_sill"], params["range"]]


def fit_scalers(df, features):
    """Fit MinMax scalers to specified features (0-1 scaling)"""
    scalers = {}
    for col in features:
        scaler = MinMaxScaler()
        scaler.fit(df[[col]])
        scalers[col] = scaler
    return scalers


def apply_scalers(df, scalers):
    """Apply fitted scalers to dataframe"""
    scaled_df = df.copy()
    for col, scaler in scalers.items():
        scaled_df[col] = scaler.transform(scaled_df[[col]]).flatten()
    return scaled_df

# ----------------------------------------
# Data Preparation
# Prepare dataset per rainfall category
# ----------------------------------------
def prepare_category_data(category_dates, daily_df, dem, distance, cape, vidmf, occu_prob):
    """Prepare dataset for a specific rainfall category with covariate extraction and scaling"""
    category_data = []
    features = ['dem', 'distance', 'prob', 'cape', 'vidmf', 'doy_sin', 'doy_cos']
    for date in tqdm(category_dates, desc=f"Processing {len(category_dates)} days"):
        day_df = daily_df[daily_df['day'] == date].copy()
        if day_df.empty:
            continue
        points_x = day_df['Longitude'].values
        points_y = day_df['Latitude'].values
        day_df['dem'] = extract_grid_value(dem, points_x, points_y)
        day_df['distance'] = extract_grid_value(distance, points_x, points_y)
        day_df['prob'] = extract_grid_value(occu_prob, points_x, points_y, time=date)
        day_df['cape'] = extract_grid_value(cape, points_x, points_y, time=date)
        day_df['vidmf'] = extract_grid_value(vidmf, points_x, points_y, time=date)
        doy = date.timetuple().tm_yday
        day_df['doy_sin'] = np.sin(2 * np.pi * doy / 365)
        day_df['doy_cos'] = np.cos(2 * np.pi * doy / 365)
        category_data.append(day_df)
    full_df = pd.concat(category_data, ignore_index=True)
    full_df.iloc[:,0], full_df.iloc[:,1:] = pd.to_datetime(full_df.iloc[:,0]), full_df.iloc[:,1:].astype(float)
    full_df.dropna(subset=features[:-2], inplace=True)
    full_df = full_df[full_df['Rain'] > 0]
    scalers = fit_scalers(full_df, features[:-2])
    scaled_df = apply_scalers(full_df, scalers)
    return scaled_df, full_df, scalers

# ----------------------------------------
# Modeling Functions
# Fit GAM, transform residuals, variogram
# ----------------------------------------
def fit_gam(X, y, n_splines=20, lam=0.1, n_jobs=-1):
    """Optimized GAM fitting with parallel grid search"""
    n_splines_candidates = [max(5, n_splines-5), n_splines, min(40, n_splines+5)]
    lam_candidates = np.logspace(np.log10(lam*0.01), np.log10(lam*100), 100)
    base_gam = GammaGAM(
        s(0) + s(1) + s(2) + te(3, 4, n_splines=[8,5], constraints=['monotonic_inc', 'monotonic_dec']) + f(5) + f(6),
        fit_intercept=True
    )
    def evaluate_params(n_splines_val, lam_val):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                gam = clone(base_gam)
                gam.terms = (
                    s(0, n_splines=n_splines_val) + s(1, n_splines=n_splines_val) + s(2, n_splines=n_splines_val) + s(3, n_splines=n_splines_val) + s(4, n_splines=n_splines_val, constraints='monotonic_dec') + f(5) + f(6)
                )
                gam.lam = lam_val
                gam.fit(X, y)
                score = (gam.statistics_.get('gcv', float('inf')) or gam.statistics_.get('AIC', float('inf')))
                return (gam, score, n_splines_val, lam_val)
            except:
                return (None, float('inf'), n_splines_val, lam_val)
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_params)(n, l) for n in n_splines_candidates for l in lam_candidates
    )
    valid_results = [r for r in results if r[0] is not None]
    if not valid_results:
        print("All grid search attempts failed. Trying basic model...")
        best_gam = clone(base_gam)
        best_gam.fit(X, y)
        print(f"Basic model fitted with score: {(best_gam.statistics_.get('gcv', float('inf')) or best_gam.statistics_.get('AIC', float('inf'))):.2f}")
        return best_gam
    best_gam, best_score, best_n, best_lam = min(valid_results, key=lambda x: x[1])
    print(f"Best model: n_splines={best_n}, lam={best_lam:.4f}, score={best_score:.2f}")
    return best_gam


def calculate_residuals(gam, X, y_true):
    """Calculate residuals from GAM prediction"""
    y_pred = gam.predict(X)
    return y_true - y_pred


def transform_residuals(residuals):
    """Apply and evaluate different residual transformations"""
    transformations = {
        'yeojohnson': PowerTransformer(method='yeo-johnson', standardize=False),
        'quantile': QuantileTransformer(output_distribution='normal', n_quantiles=min(1000, len(residuals)))
    }
    results = {}
    for name, transformer in transformations.items():
        try:
            transformed = transformer.fit_transform(residuals.reshape(-1, 1)).flatten()
            sample = transformed[~np.isnan(transformed)]
            if len(sample) > 5000:
                sample = np.random.choice(sample, 5000, replace=False)
            _, p_val = shapiro(sample)
            results[name] = {'transformer': transformer, 'transformed': transformed, 'p_value': p_val}
        except Exception as e:
            print(f"Transformation {name} failed: {str(e)}")
            results[name] = None
    valid_results = {k: v for k, v in results.items() if v is not None}
    best_trans = max(valid_results, key=lambda k: valid_results[k]['p_value'])
    return results, best_trans


def plot_qq(residuals, transformations, best_trans, category):
    """Generate Q-Q plots for residual transformations"""
    fig, axs = plt.subplots(1, len(transformations), figsize=(15, 5))
    fig.suptitle(f'Q-Q Plots: {category}', fontsize=16)
    for ax, (name, result) in zip(axs, transformations.items()):
        if result is None:
            continue
        transformed = result['transformed']
        finite_vals = transformed[np.isfinite(transformed)]
        if len(finite_vals) < 2:
            continue
        osm, osr = probplot(finite_vals, dist="norm", fit=False)
        ax.scatter(osm, osr, alpha=0.5)
        ax.plot([osm.min(), osm.max()], [osm.min(), osm.max()], 'r--')
        title = f"{name} (p={result['p_value']:.4f})"
        if name == best_trans:
            title += " [BEST]"
        ax.set_title(title)
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Ordered Values")
        ax.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"qq_plot_{category}.png", dpi=300)
    plt.close()


def fit_variogram(coords, residuals, transformer):
    """Fit variogram model to transformed residuals with automatic model selection"""
    transformed_res = transformer.transform(residuals.reshape(-1, 1)).flatten()
    valid_mask = np.isfinite(transformed_res)
    transformed_res = transformed_res[valid_mask]
    valid_coords = coords[valid_mask]
    pos_tuple = (valid_coords[:, 0], valid_coords[:, 1])
    bins = np.linspace(0, 200000, 20)
    bin_center, gamma, counts = vario_estimate(
        pos=pos_tuple,
        field=transformed_res,
        bin_edges=bins,
        sampling_size=5000,
        latlon=False,
        estimator="matheron",
        return_counts=True
    )
    models = {'Gaussian': Gaussian, 'Exponential': Exponential, 'Matern': Matern, 'Stable': Stable}
    best_model = None
    best_rmse = np.inf
    for name, model_class in models.items():
        try:
            model = model_class(dim=2)
            model.fit_variogram(bin_center, gamma, nugget=True)
            pred = model.variogram(bin_center)
            rmse = np.sqrt(np.mean((gamma - pred)**2))
            if rmse < best_rmse:
                best_model = model
                best_rmse = rmse
        except:
            continue
    return best_model, (bin_center, gamma)


def get_variogram_function_and_params(fitted_model):
    """Convert fitted GSTools model to PyKrige-compatible custom variogram"""
    model_class = type(fitted_model).__name__
    nugget = fitted_model.nugget
    var = fitted_model.var
    len_scale = fitted_model.len_scale
    full_sill = nugget + var
    params = [full_sill, len_scale, nugget]
    if model_class == "Matern":
        params.append(fitted_model.nu)
    elif model_class == "Stable":
        params.append(fitted_model.alpha)
    def variogram_func(params, dists):
        if model_class == "Matern":
            model = gs.Matern(dim=2, nugget=params[2], var=params[0] - params[2], len_scale=params[1], nu=params[3])
        elif model_class == "Stable":
            model = gs.Stable(dim=2, nugget=params[2], var=params[0] - params[2], len_scale=params[1], alpha=params[3])
        else:
            model = getattr(gs, model_class)(dim=2, nugget=params[2], var=params[0] - params[2], len_scale=params[1])
        return model.variogram(dists)
    return variogram_func, params

# ----------------------------------------
# Prediction Functions
# Predict rainfall for single dates
# ----------------------------------------
def predict_single_date(date, category, daily_df, dem, distance, cape, vidmf, occu_prob, occu_binary,
                        land_mask, gam, scalers, transformer, vario_model):
    """Predict rainfall using two-stage approach for a specific date"""
    grid_x = dem.x.values
    grid_y = dem.y.values
    X_full, Y_full = np.meshgrid(grid_x, grid_y)
    full_points = np.column_stack([X_full.ravel(), Y_full.ravel()])
    land_mask_vals = extract_grid_value(land_mask, full_points[:, 0], full_points[:, 1]).astype(bool)
    land_points = full_points[land_mask_vals]
    grid_df = pd.DataFrame({'x': land_points[:, 0], 'y': land_points[:, 1]})
    grid_df['dem'] = extract_grid_value(dem, grid_df['x'], grid_df['y'])
    grid_df['distance'] = extract_grid_value(distance, grid_df
