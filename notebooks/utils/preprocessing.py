import pandas as pd
import numpy as np
import scipy 
from scipy import stats

def rolled(data, window_size):
    """
    Generator to yield batches of rows from a data frame of specified window size.
    
    Parameters:
    data (array-like): Input data from which windows are generated.
    window_size (int): The size of each window.

    Yields:
    array-like: Subsequent windows of the input data.
    """
    count = 0
    while count <= len(data) - window_size:
        yield data[count: count + window_size]
        count += 1
        
def max_rolled(data, window_size):
    """
    Returns the maximum value for each rolling sliding window.
    
    Parameters:
    data (array-like): List of values from which rolling windows are generated.
    window_size (int): The size of each window.

    Returns:
    np.array: Array of maximum values for each rolling window.
    """
    max_values = []
    for window in rolled(data, window_size):
        max_values.append(max(window))
    
    return np.array(max_values)

def get_windows_data(data_frame, labels, window_size, tsfresh=True):
    """
    Prepare data for autoencoder and tsfresh processing.
    
    Parameters:
    data_frame (pd.DataFrame): Input data frame containing features.
    labels (array-like): Corresponding labels for the data.
    window_size (int): The size of each window.
    tsfresh (bool): Indicator whether to prepare dataframe for tsfresh (add 'id' and 'time' columns).

    Returns:
    tuple: A tuple (X, y) where X is the processed data and y are the corresponding labels.
    """
    all_windows = []

    # Iterate over windows generated from the input data frame
    for index, window in enumerate(rolled(data_frame, window_size)):
        window = window.copy()
        if tsfresh:
            window['id'] = [index] * window.shape[0]
            window['time'] = list(range(window.shape[0]))
        all_windows.append(window)

    # Combine all windows into a single data frame or array
    if all_windows:
        X = pd.concat(all_windows, ignore_index=True) if tsfresh else np.array([w.values for w in all_windows])
    else:
        X = pd.DataFrame() if tsfresh else np.array([])

    # Compute the max rolled values for the labels
    y = max_rolled(labels, window_size)

    return X, y

def diff_data(df, cols, lag, order):
    """
    Apply time series differencing to the data.
    
    Parameters:
    df (pd.DataFrame): Input data frame.
    cols (list): Columns to apply differencing.
    lag (int): Number of periods to lag for differencing.
    order (int): Number of differencing steps to apply.
    
    Returns:
    pd.DataFrame: Data frame with differenced data.
    """
    # Ensure lag and order are positive integers
    assert lag > 0, "Lag must be greater than 0"
    assert order > 0, "Order must be greater than 0"

    # Apply differencing to specified columns
    df_diff = df[cols].copy()
    
    for _ in range(order):
        df_diff = df_diff.diff(periods=lag)
        # Remove NaN value rows resulting from differencing
        df_diff = df_diff[lag:]

    # Include columns that were not differenced
    excluded_cols = [col for col in df.columns if col not in cols]
    
    for col in excluded_cols:
        df_diff[col] = df[col][lag * order:].values

    return df_diff

def filter_outliers(df, std=5, cols=None):
    """
    Remove extreme outliers in the data based on z-score.
    
    Parameters:
    df (pd.DataFrame): Input data frame.
    std (int): Number of standard deviations to use as the threshold. Values beyond this threshold are considered outliers.
    cols (list): List of columns to apply outlier filtering. If None, all columns are used.
    
    Returns:
    pd.DataFrame: Filtered data frame with outliers removed.
    """
    # Select columns to apply outlier filtering
    selected_cols = df.columns if cols is None else cols

    # Calculate the z-score for the selected columns
    z_scores = np.abs(stats.zscore(df[selected_cols]))

    # Filter out rows with z-scores beyond the specified number of standard deviations
    filtered_df = df[(z_scores < std).all(axis=1)]

    return filtered_df