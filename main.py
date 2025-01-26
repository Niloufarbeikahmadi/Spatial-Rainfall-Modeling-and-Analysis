# -*- coding: utf-8 -*-
"""
Created on Sun March 26 15:14:34 2024

@author: niloufar
"""
# main.py

from data_processing import load_data, aggregate_daily, binary_indicator
from variogram_models import fit_spherical_model, spherical_model
from kriging_methods import krige_with_best_variogram
from visualizations import plot_variogram

# Load and preprocess data
file_path = "your_excel_file.xlsx"
df = load_data(file_path)
daily_df = aggregate_daily(df)
daily_df = binary_indicator(daily_df)

# Fit variogram model
lags = np.array([0.1, 0.2, 0.3])  # Example lags
gamma = np.array([0.01, 0.02, 0.03])  # Example semivariance
params = fit_spherical_model(lags, gamma)
model_gamma = spherical_model(lags, *params)

# Plot variogram
plot_variogram(lags, gamma, model_gamma)

# Perform kriging
class_dict = {}  # Example classification
grid_results = krige_with_best_variogram(daily_df, class_dict, {})
