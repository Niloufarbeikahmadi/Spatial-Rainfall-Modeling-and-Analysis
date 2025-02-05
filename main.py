#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main.py
-------

Main script that ties together data loading, variogram modeling, kriging, and visualization.
"""

import datetime
import pickle
import pandas as pd

# Import our modules.
import data
import variogram
import kriging
import visualization

# ====== Data Loading & Preprocessing ======
data_path = "C:/Users/nilou/OneDrive/Desktop/UNIPA PHD/obs reconstrution/data/data di protezione civile/filtered/daily/merged.xlsx"
df = data.load_data(data_path)
daily_df = data.aggregate_daily(df)
thresholds = [(0, 25), (25, 75), (75, 100)]
class_dict = data.classify_days(daily_df, thresholds)
daily_df = data.binary_indicator(daily_df)

# ====== Phase 1: Occurrence Variogram Modeling and Kriging ======
# (Plot variograms and select best models based on AIC.)
comparison_results = visualization.plot_variograms_with_models(
    daily_df, class_dict, maxlag=2, n_lags=20, chunk_size=5000, variogram_module=variogram
)
variogram_params = variogram.extract_best_variogram_params(comparison_results)
results_2km = kriging.krige_with_best_variogram(daily_df, class_dict, variogram_params, grid_res=0.02)
class_thresholds = kriging.calculate_class_cutoffs(results_2km, class_dict)
percentages_df = data.compute_class_percentages_by_month(class_dict, daily_df)
percentages_df.to_csv("class_percentages_by_month.csv", index_label="Month")
results_2km_with_binary = kriging.create_binary_maps(results_2km, class_thresholds, class_dict)

# Save Phase 1 results.
occ_save_path = r"C:/Users/nilou/OneDrive/Desktop/UNIPA PHD/obs reconstrution/python_codes/my outputs/results_2km.pkl"
with open(occ_save_path, 'wb') as file:
    pickle.dump(results_2km, file)

# Optionally, plot a daily probability map.
example_day = datetime.date(2024, 11, 10)
visualization.plot_daily_probability_map(
    prob_map=results_2km[example_day]['prob_map'],
    grid_lons=results_2km[example_day]['grid_lons'],
    grid_lats=results_2km[example_day]['grid_lats'],
    day=example_day,
    daily_df=daily_df
)

# ====== Phase 2: Rainfall Magnitude Analysis ======
rainfall_stats = data.compute_rainfall_stats(daily_df)

# Define groups based on μ_category and h_max_category (from your extracted table).
group_defs = {
    "Group1": {"μ_category": "Light (A)",         "h_max_category": "Light (A)"},
    "Group2": {"μ_category": "Light (A)",         "h_max_category": "Light-Moderate (B)"},
    "Group3": {"μ_category": "Light (A)",         "h_max_category": "Moderate-Heavy (C1)"},
    "Group4": {"μ_category": "Light (A)",         "h_max_category": "Heavy (C2)"},
    "Group5": {"μ_category": "Light-Moderate (B)",  "h_max_category": "Heavy (C2)"},
    "Group6": {"μ_category": "Light-Moderate (B)",  "h_max_category": "Heavy-Torrential (D1)"},
    "Group7": {"μ_category": "Light-Moderate (B)",  "h_max_category": "Torrential (D2)"},
    "Group8": {"μ_category": "Light-Moderate (B)",  "h_max_category": "Moderate-Heavy (C1)"}
}

def assign_group_days(rainfall_stats: pd.DataFrame, group_defs: dict) -> dict:
    group_days = {grp: [] for grp in group_defs.keys()}
    for _, row in rainfall_stats.iterrows():
        for grp, crit in group_defs.items():
            if (row['μ_category'] == crit["μ_category"]) and (row['h_max_category'] == crit["h_max_category"]):
                group_days[grp].append(row['day'])
                break
    return group_days

group_days = assign_group_days(rainfall_stats, group_defs)

# Compute magnitude variograms for each group.
magnitude_variogram_results = {}
maxlag_magnitude = 2
n_lags_magnitude = 20
chunk_size_magnitude = 5000

for grp, days in group_days.items():
    if len(days) < 5:
        print(f"Skipping {grp} because of too few days ({len(days)})")
        continue
    print(f"\nProcessing {grp} with {len(days)} days ...")
    lags, exp_gamma = variogram.compute_magnitude_variogram(daily_df, days,
                                                            maxlag=maxlag_magnitude,
                                                            n_lags=n_lags_magnitude,
                                                            chunk_size=chunk_size_magnitude)
    if lags is None or exp_gamma is None:
        print(f"No valid experimental variogram for {grp}")
        continue
    best_model = variogram.select_best_variogram_model(lags, exp_gamma, maxlag_magnitude)
    if best_model is None:
        print(f"Could not fit any model for {grp}")
        continue
    model_name, popt, aic, bic = best_model
    magnitude_variogram_results[grp] = {
        'Best Model': model_name,
        'Parameters': popt,
        'AIC': aic,
        'BIC': bic,
        'Lags': lags,
        'Experimental Variogram': exp_gamma
    }
print("\nBest fitted theoretical variogram models for each group:")
for grp, res in magnitude_variogram_results.items():
    print(f"\n{grp}:")
    print(f"  Best Model: {res['Best Model']}")
    print(f"  Parameters: {res['Parameters']}")
    print(f"  AIC: {res['AIC']:.2f}")

# Optionally, plot experimental variograms and the fitted models for each group.
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()
for idx, (grp, res) in enumerate(magnitude_variogram_results.items()):
    lags = res['Lags']
    exp_gamma = res['Experimental Variogram']
    model_name = res['Best Model']
    popt = res['Parameters']
    h_fit = np.linspace(0, maxlag_magnitude, 100)
    if model_name == 'Matérn':
        fitted_curve = variogram.matern_model(h_fit, *popt)
    elif model_name == 'Spherical':
        fitted_curve = variogram.spherical_model(h_fit, *popt)
    elif model_name == 'Exponential':
        fitted_curve = variogram.exponential_model(h_fit, *popt)
    elif model_name == 'Gaussian':
        fitted_curve = variogram.gaussian_model(h_fit, *popt)
    elif model_name == 'Wave Effect':
        fitted_curve = variogram.wave_effect_model(h_fit, *popt)
    else:
        fitted_curve = np.zeros_like(h_fit)
    ax = axes[idx]
    ax.plot(lags, exp_gamma, 'o', label='Experimental')
    ax.plot(h_fit, fitted_curve, '-', label=f'Fitted {model_name}')
    ax.set_xlabel("Lag distance (degrees)")
    ax.set_ylabel("Semivariance")
    ax.set_title(f"{grp}")
    ax.legend()
    ax.grid(True)
plt.tight_layout()
plt.show()

# Krige rainfall magnitude using binary occurrence maps.
final_rain_maps = kriging.krige_final_rainfall_maps(daily_df, results_2km_with_binary,
                                                    magnitude_variogram_results, group_days, grid_res=0.02)
final_rain_maps = kriging.replace_negative_with_zero(final_rain_maps)
final_save_path = r"C:/Users/nilou/OneDrive/Desktop/UNIPA PHD/obs reconstrution/python_codes/my outputs/final_rain_maps.pkl"
with open(final_save_path, 'wb') as file:
    pickle.dump(final_rain_maps, file)

# Overview visualization for a sample day.
day_to_plot = datetime.date(2022, 12, 13)
fig, axes = visualization.plot_day_overview(day=day_to_plot,
                                            daily_df=daily_df,
                                            occurrence_results=results_2km_with_binary,
                                            magnitude_results=final_rain_maps,
                                            title_prefix="Rainfall Overview")

# Optionally, extract and display the occurrence and magnitude classes for the sample day.
occurrence_class = None
for class_label, dates in class_dict.items():
    if pd.Timestamp(day_to_plot) in pd.to_datetime(dates):
        occurrence_class = class_label
        break
magnitude_group = None
for group_label, dates in group_days.items():
    if pd.Timestamp(day_to_plot) in pd.to_datetime(dates):
        magnitude_group = group_label
        break
print(f"Occurrence Frequency Class: {occurrence_class if occurrence_class else 'Not Found'}")
print(f"Magnitude Group: {magnitude_group if magnitude_group else 'Not Found'}")
