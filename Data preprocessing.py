# -*- coding: utf-8 -*-
"""
Created on Sun March 26 10:09:30 2024

@author: Niloufar
"""

# data_processing.py

import pandas as pd





def load_data(excel_file: str) -> pd.DataFrame:
    """
    Load the rainfall data from an Excel file.

    Parameters:
        excel_file (str): Path to the Excel file containing rainfall data.

    Returns:
        pd.DataFrame: DataFrame with columns ['ID', 'Latitude', 'Longitude', 'Date', 'Rain'].
    """
    df = pd.read_excel(excel_file, parse_dates=['Date'])
    return df

def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate rainfall data to daily totals per station.

    Parameters:
        df (pd.DataFrame): Input data with 'Date' and 'Rain' columns.

    Returns:
        pd.DataFrame: Aggregated daily rainfall data.
    """
    df['day'] = df['Date'].dt.date
    daily_df = (df.groupby(['day', 'ID', 'Latitude', 'Longitude'], as_index=False)
                  .agg({'Rain': 'sum'}))
    return daily_df

def binary_indicator(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a binary indicator column where 1 indicates rainfall and 0 indicates no rainfall.

    Parameters:
        daily_df (pd.DataFrame): Daily rainfall data.

    Returns:
        pd.DataFrame: Modified DataFrame with a binary 'indicator' column.
    """
    daily_df['indicator'] = (daily_df['Rain'] > 0).astype(int)
    return daily_df
