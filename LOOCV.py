
# =============================================================================
# Validation Section: LOOCV for Rainfall Interpolation
# =============================================================================


# -------------------------------
# LOOCV for Occurrence Interpolation
# -------------------------------
def loocv_occurrence(daily_df, class_dict, variogram_params, grid_res=0.02):
    """
    Perform Leave-One-Out Cross Validation (LOOCV) for the rainfall occurrence interpolation.
    For each day in the dataset, remove one gauge at a time, krige the remaining points,
    and compare the predicted indicator (0 or 1) with the observed indicator.

    Parameters:
        daily_df (pd.DataFrame): Daily data with columns including 'day', 'Longitude',
                                 'Latitude', and 'indicator'.
        class_dict (dict): Dictionary mapping class labels (e.g., "F0-25", "F25-75", "F75-100")
                           to lists of days.
        variogram_params (dict): Dictionary mapping class labels to variogram parameters (nugget, sill, range).
        grid_res (float): (Not used in LOOCV but provided for consistency with your workflow.)

    Returns:
        rmse (float): Root Mean Square Error over all left‐out points.
        mae (float): Mean Absolute Error over all left‐out points.
        predicted_values (list): List of predicted indicator values.
        observed_values (list): List of observed indicator values.
    """
    errors = []
    predicted_values = []
    observed_values = []
    
    unique_days = daily_df['day'].unique()
    for day in unique_days:
        day_data = daily_df[daily_df['day'] == day]
        # Skip days with too few points
        if len(day_data) < 3:
            continue

        # Determine the occurrence class for the day
        class_label = None
        for label, days in class_dict.items():
            # Convert days from class_dict to Timestamps for robust comparison
            days_ts = [pd.Timestamp(d) for d in days]
            if pd.Timestamp(day) in days_ts:
                class_label = label
                break
        if class_label is None:
            continue

        # Get the variogram parameters for this class
        if class_label not in variogram_params:
            continue
        params = variogram_params[class_label]
        # Here we use the exponential model as in your kriging function
        variogram_model = 'exponential'
        variogram_parameters = {'sill': params[1], 'range': params[2], 'nugget': params[0]}
        
        # For each gauge point (LOOCV: leave this point out and predict it)
        for idx, test_row in day_data.iterrows():
            train_data = day_data.drop(idx)
            # Ensure that there are enough remaining points
            if len(train_data) < 3:
                continue
            try:
                OK = OrdinaryKriging(
                    x=train_data['Longitude'].values,
                    y=train_data['Latitude'].values,
                    z=train_data['indicator'].values.astype(float),
                    variogram_model=variogram_model,
                    variogram_parameters=variogram_parameters,
                    verbose=False,
                    enable_plotting=False
                )
                # Predict the indicator at the test point location
                pred, ss = OK.execute('points',
                                      np.array([test_row['Longitude']]),
                                      np.array([test_row['Latitude']]))
                predicted = pred[0]
                observed = test_row['indicator']
                error = observed - predicted
                errors.append(error)
                predicted_values.append(predicted)
                observed_values.append(observed)
            except Exception as e:
                print(f"LOOCV occurrence error for day {day} at index {idx}: {e}")
                continue

    errors = np.array(errors)
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(np.abs(errors))
    return rmse, mae, predicted_values, observed_values

# -------------------------------
# LOOCV for Rainfall Magnitude Interpolation
# -------------------------------
def loocv_magnitude(daily_df, group_days, magnitude_variogram_results, grid_res=0.02):
    """
    Perform LOOCV for the rainfall magnitude interpolation.
    For each day, remove one gauge point at a time, perform kriging with the remaining points
    using the variogram parameters corresponding to the day’s magnitude group, and compare the
    predicted rainfall amount to the observed gauge record.

    Parameters:
        daily_df (pd.DataFrame): Daily data with columns including 'day', 'Longitude',
                                 'Latitude', and 'Rain'.
        group_days (dict): Dictionary mapping magnitude group names (e.g., "Group1", "Group2", ...)
                           to lists of days.
        magnitude_variogram_results (dict): Dictionary mapping magnitude group names to dictionaries
                                            containing the best-fit model parameters.
        grid_res (float): (Not used in LOOCV but provided for consistency.)

    Returns:
        rmse (float): Root Mean Square Error over all left‐out points.
        mae (float): Mean Absolute Error over all left‐out points.
        predicted_values (list): List of predicted rainfall amounts.
        observed_values (list): List of observed rainfall amounts.
    """
    errors = []
    predicted_values = []
    observed_values = []
    
    unique_days = daily_df['day'].unique()
    for day in unique_days:
        day_data = daily_df[daily_df['day'] == day]
        if len(day_data) < 3:
            continue
        
        # Determine the magnitude group for the day
        group_label = None
        # Convert group_days values to normalized Timestamps for comparison
        for grp, days in group_days.items():
            days_norm = pd.to_datetime(days).normalize()
            if pd.to_datetime(day).normalize() in days_norm.values:
                group_label = grp
                break
        if group_label is None:
            continue
        
        # Retrieve the magnitude variogram result for this group
        mag_var_result = magnitude_variogram_results.get(group_label, None)
        if mag_var_result is None:
            continue
        best_model = mag_var_result['Best Model']
        popt = mag_var_result['Parameters']  # Expected format: [nugget, sill, range, ...]
        
        # Map best model name to a PyKrige acceptable variogram model string.
        # (Here we use 'spherical' as a fallback for custom models.)
        model_map = {
            'Spherical': 'spherical',
            'Exponential': 'exponential',
            'Gaussian': 'gaussian',
            'Matérn': 'spherical',
            'Wave Effect': 'spherical'
        }
        variogram_model = model_map.get(best_model, 'spherical')
        variogram_parameters = {'nugget': popt[0], 'sill': popt[1], 'range': popt[2]}
        
        # For each gauge point (LOOCV)
        for idx, test_row in day_data.iterrows():
            train_data = day_data.drop(idx)
            if len(train_data) < 3:
                continue
            try:
                OK = OrdinaryKriging(
                    x=train_data['Longitude'].values,
                    y=train_data['Latitude'].values,
                    z=train_data['Rain'].values.astype(float),
                    variogram_model=variogram_model,
                    variogram_parameters=variogram_parameters,
                    verbose=False,
                    enable_plotting=False
                )
                # Predict at the location of the left-out gauge
                pred, ss = OK.execute('points',
                                      np.array([test_row['Longitude']]),
                                      np.array([test_row['Latitude']]))
                predicted = pred[0]
                observed = test_row['Rain']
                error = observed - predicted
                errors.append(error)
                predicted_values.append(predicted)
                observed_values.append(observed)
            except Exception as e:
                print(f"LOOCV magnitude error for day {day} at index {idx}: {e}")
                continue
                
    errors = np.array(errors)
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(np.abs(errors))
    return rmse, mae, predicted_values, observed_values

# -------------------------------
# Run the LOOCV Validation
# -------------------------------

# LOOCV for Occurrence (Rain/No Rain) Interpolation
rmse_occ, mae_occ, pred_occ, obs_occ = loocv_occurrence(daily_df, class_dict, variogram_params, grid_res=0.02)
print("LOOCV Occurrence Interpolation Validation:")
print(f"  RMSE: {rmse_occ:.4f}")
print(f"  MAE: {mae_occ:.4f}")

# LOOCV for Magnitude (Rainfall Amount) Interpolation
rmse_mag, mae_mag, pred_mag, obs_mag = loocv_magnitude(daily_df, group_days, magnitude_variogram_results, grid_res=0.02)
print("\nLOOCV Magnitude Interpolation Validation:")
print(f"  RMSE: {rmse_mag:.4f}")
print(f"  MAE: {mae_mag:.4f}")

# plot predicted vs. observed values to visually inspect the performance:


plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(obs_occ, pred_occ, alpha=0.6, edgecolor='k')
plt.plot([min(obs_occ), max(obs_occ)], [min(obs_occ), max(obs_occ)], 'r--')
plt.xlabel("Observed Occurrence (0/1)")
plt.ylabel("Predicted Occurrence")
plt.title("LOOCV: Occurrence Field")

plt.subplot(1, 2, 2)
plt.scatter(obs_mag, pred_mag, alpha=0.6, edgecolor='k')
plt.plot([min(obs_mag), max(obs_mag)], [min(obs_mag), max(obs_mag)], 'r--')
plt.xlabel("Observed Rainfall (mm)")
plt.ylabel("Predicted Rainfall (mm)")
plt.title("LOOCV: Rainfall Magnitude")
plt.tight_layout()
plt.show()
