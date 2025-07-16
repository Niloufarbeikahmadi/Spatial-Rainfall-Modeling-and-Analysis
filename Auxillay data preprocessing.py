#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Downscale ERA5 CAPE and VIDMF data to 2km grid in EPSG:32633 projection

Author: Niloufar Beikahmadi
Created: Wed Jul 16 21:57:51 2025
"""

import numpy as np
import xarray as xr
import pygrib
import pandas as pd
import geopandas as gpd
from scipy.interpolate import LinearNDInterpolator
from pyproj import Transformer, Geod
from tqdm import tqdm
from shapely.geometry import Point
from shapely.ops import unary_union


def create_sicily_mask_rtree(
    shapefile_path: str, 
    grid_x: np.ndarray, 
    grid_y: np.ndarray, 
    buffer_meters: int = 2000
) -> np.ndarray:
    """
    Create a sea mask for Sicily region
    
    Parameters:
    shapefile_path (str): Path to Sicily shapefile
    grid_x (np.ndarray): X coordinates of target grid
    grid_y (np.ndarray): Y coordinates of target grid
    buffer_meters (int): Buffer distance around Sicily
    
    Returns:
    np.ndarray: Boolean mask where True = sea, False = land
    """
    sicily = gpd.read_file(shapefile_path).to_crs(epsg=32633)
    buffered = unary_union(sicily.geometry).buffer(buffer_meters)
    mask = np.zeros((len(grid_y), len(grid_x)), dtype=bool)
    
    for j in range(len(grid_y)):
        for i in range(len(grid_x)):
            x, y = grid_x[i], grid_y[j]
            mask[j, i] = buffered.contains(Point(x, y))
            
    return ~mask


def create_grid_in_meters(
    lon_min_deg: float,
    lon_max_deg: float,
    lat_min_deg: float,
    lat_max_deg: float,
    resolution_deg: float = 0.02
) -> tuple:
    """
    Create target grid in EPSG:32633 projection
    
    Parameters:
    lon_min_deg (float): Minimum longitude (degrees)
    lon_max_deg (float): Maximum longitude (degrees)
    lat_min_deg (float): Minimum latitude (degrees)
    lat_max_deg (float): Maximum latitude (degrees)
    resolution_deg (float): Grid resolution in degrees
    
    Returns:
    tuple: (grid_x, grid_y, transformer)
    """
    geod = Geod(ellps="WGS84")
    center_lat = (lat_min_deg + lat_max_deg) / 2
    
    # Calculate grid spacing in meters
    _, _, dx = geod.inv(lon_min_deg, center_lat, 
                       lon_min_deg + resolution_deg, center_lat)
    _, _, dy = geod.inv(lon_min_deg, center_lat, 
                       lon_min_deg, center_lat + resolution_deg)
    
    # Create coordinate transformer
    transformer = Transformer.from_crs(4326, 32633, always_xy=True)
    
    # Transform boundaries to meters
    x_min_m, y_min_m = transformer.transform(lon_min_deg, lat_min_deg)
    x_max_m, y_max_m = transformer.transform(lon_max_deg, lat_max_deg)
    
    # Create grid arrays
    grid_x = np.arange(x_min_m, x_max_m + dx, dx)
    grid_y = np.arange(y_min_m, y_max_m + dy, dy)
    
    return grid_x, grid_y, transformer


def process_era5_variable(
    grib_path: str,
    parameter_name: str,
    time_index: pd.DatetimeIndex,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    sea_mask: np.ndarray
) -> xr.DataArray:
    """
    Process and downscale ERA5 variable to target grid
    
    Parameters:
    grib_path (str): Path to GRIB file
    parameter_name (str): Name of parameter to extract
    time_index (pd.DatetimeIndex): Target time index
    grid_x (np.ndarray): Target x-coordinates
    grid_y (np.ndarray): Target y-coordinates
    sea_mask (np.ndarray): Sea mask for filtering
    
    Returns:
    xr.DataArray: Downscaled data array
    """
    # Open GRIB file and extract messages
    grbs = pygrib.open(grib_path)
    messages = [msg for msg in grbs if msg.parameterName == parameter_name]
    
    if not messages:
        raise ValueError(f"Parameter {parameter_name} not found in GRIB file")
    
    # Extract data and coordinates
    data = np.array([msg.values for msg in messages])
    times = [msg.validDate for msg in messages]
    lat_grid, lon_grid = messages[0].latlons()
    
    # Create DataArray
    da = xr.DataArray(
        data,
        dims=('time', 'lat', 'lon'),
        coords={
            'time': times,
            'lat': lat_grid[:, 0],
            'lon': lon_grid[0, :]
        },
        name=parameter_name
    )
    
    # Daily aggregation
    da_daily = da.resample(time='1D').sum()
    da_daily = da_daily.reindex(time=time_index)
    
    # Project coarse grid to EPSG:32633
    transformer = Transformer.from_crs(4326, 32633, always_xy=True)
    lon_grid, lat_grid = np.meshgrid(da_daily.lon.values, da_daily.lat.values)
    x_coarse, y_coarse = transformer.transform(lon_grid, lat_grid)
    points = np.column_stack((x_coarse.ravel(), y_coarse.ravel()))
    
    # Prepare fine grid
    X_fine, Y_fine = np.meshgrid(grid_x, grid_y)
    downscaled_data = np.full((len(time_index), len(grid_y), len(grid_x)), np.nan)
    
    # Process each timestep
    for i, t in enumerate(tqdm(time_index, desc=f'Processing {parameter_name}')):
        if t not in da_daily.time:
            continue
            
        values = da_daily.sel(time=t).values.ravel()
        
        if np.all(np.isnan(values)):
            continue
            
        # Create and apply interpolator
        interp = LinearNDInterpolator(points, values, fill_value=np.nan)
        downscaled = interp(X_fine, Y_fine)
        
        # Apply sea mask
        downscaled[sea_mask] = np.nan
        downscaled_data[i] = downscaled
    
    grbs.close()
    
    return xr.DataArray(
        downscaled_data,
        dims=('time', 'y', 'x'),
        coords={'time': time_index, 'x': grid_x, 'y': grid_y},
        name=parameter_name
    )


def main():
    """Main processing workflow"""
    # Configuration parameters
    SHAPEFILE_PATH = "C:/Users/.../sicily.shp"
    GRIB_PATH = 'C:/Users/.../ERA5_reanalysis.grib'
    OUTPUT_PATH = 'era5_downscaled.nc'
    RESOLUTION_DEG = 0.02  # ~2km resolution
    LON_MIN, LON_MAX = 12.0, 16.0
    LAT_MIN, LAT_MAX = 36.5, 38.5
    
    # Create target grid and sea mask
    print("Creating target grid...")
    grid_x, grid_y, _ = create_grid_in_meters(
        LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, RESOLUTION_DEG
    )
    
    print("Creating sea mask...")
    sea_mask = create_sicily_mask_rtree(SHAPEFILE_PATH, grid_x, grid_y)
    
    # Define time range (1067 days)
    print("Creating time index...")
    time_index = pd.date_range('2022-01-01', '2024-12-02', freq='D')
    
    # Process ERA5 variables
    print("\nProcessing CAPE...")
    da_cape = process_era5_variable(
        GRIB_PATH, 
        'Convective available potential energy', 
        time_index, 
        grid_x, 
        grid_y, 
        sea_mask
    )
    
    print("\nProcessing VIDMF...")
    da_vidmf = process_era5_variable(
        GRIB_PATH, 
        'Vertical integral of divergence of moisture flux', 
        time_index, 
        grid_x, 
        grid_y, 
        sea_mask
    )
    
    # Create final dataset
    print("\nCreating final dataset...")
    ds_final = xr.Dataset({
        'cape': da_cape,
        'vidmf': da_vidmf,
        'land_mask': (('y', 'x'), (~sea_mask).astype(np.float32))
    })
    
    # Set data types
    ds_final['x'] = ds_final.x.astype(np.float32)
    ds_final['y'] = ds_final.y.astype(np.float32)
    ds_final['time'] = ds_final.time.astype('datetime64[ns]')
    
    # Save to NetCDF
    print(f"Saving results to {OUTPUT_PATH}")
    ds_final.to_netcdf(OUTPUT_PATH)
    print("Processing complete!")


if __name__ == "__main__":
    main()
