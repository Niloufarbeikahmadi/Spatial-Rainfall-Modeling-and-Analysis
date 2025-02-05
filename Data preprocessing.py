#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
data preprocessing.py
-------

Handles data input, preprocessing, and rainfall statistics.
"""

import pandas as pd

def load_data(excel_file: str) -> pd.DataFrame:
    """
    Load the rainfall data from an Excel file.
    
    Expected columns: ID, Latitude, Longitude, Date, Rain.
    """
    df = pd.read_excel(excel_file, parse_dates=['Date'])
    return df

def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the rainfall data to daily totals per station.
    """
    df['day'] = df['Date'].dt.date  # Extract date without time
    daily_df = df.groupby(['day', 'ID', 'Latitude', 'Longitude'], as_index=False).agg({'Rain': 'sum'})
    return daily_df

def classify_days(daily_df: pd.DataFrame, thresholds: list) -> dict:
    """
    Classify days based on the percentage of stations with zero rainfall.
    
    thresholds: List of tuples (min_percentage, max_percentage) for each class.
    Returns a dict mapping class labels (e.g., "F0-25") to lists of day identifiers.
    """
    day_groups = daily_df.groupby('day')
    # Compute the frequency of zero-rainfall stations per day.
    zero_freq = day_groups.apply(lambda g: (g['Rain'] == 0).sum() / len(g) * 100, include_groups=False)
    
    class_dict = {}
    for th in thresholds:
        label = f"F{int(th[0])}-{int(th[1])}"
        class_days = zero_freq[(zero_freq >= th[0]) & (zero_freq <= th[1])].index.tolist()
        class_dict[label] = class_days

    all_classified_days = set(sum(class_dict.values(), []))
    unclassified_days = set(zero_freq.index) - all_classified_days
    if unclassified_days:
        print(f"Unclassified days: {sorted(list(unclassified_days))}")
    return class_dict

def binary_indicator(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a binary indicator column for rainfall occurrence (1 if Rain > 0, else 0).
    """
    daily_df['indicator'] = (daily_df['Rain'] > 0).astype(int)
    return daily_df

def compute_rainfall_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily rainfall statistics: mean (μ), max (h_max), std (σ), coefficient of variation (CV),
    and assign rainfall categories.
    """
    grouped = df.groupby('day')['Rain'].agg(['mean', 'max', 'std']).reset_index()
    grouped.rename(columns={'mean': 'μ', 'max': 'h_max', 'std': 'σ'}, inplace=True)
    grouped['CV'] = grouped['σ'] / grouped['μ']
    grouped['μ_category'] = grouped['μ'].apply(classify_rainfall)
    grouped['h_max_category'] = grouped['h_max'].apply(classify_rainfall)
    return grouped

def classify_rainfall(value: float) -> str:
    """
    Classify rainfall amount into standard Mediterranean categories.
    """
    if value <= 4:
        return "Light (A)"
    elif value <= 16:
        return "Light-Moderate (B)"
    elif value <= 32:
        return "Moderate-Heavy (C1)"
    elif value <= 64:
        return "Heavy (C2)"
    elif value <= 128:
        return "Heavy-Torrential (D1)"
    else:
        return "Torrential (D2)"
