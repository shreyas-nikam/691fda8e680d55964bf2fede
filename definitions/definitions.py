import pandas as pd
import numpy as np

def stress_scenario_volatility(X_data, factor, vol_cols):
    """
    Applies a volatility shock to specified columns of a Pandas DataFrame.

    Arguments:
    X_data: Pandas DataFrame, the input features to be stressed.
    factor: float, the multiplicative factor.
    vol_cols: list of strings, names of columns to stress. If None, all numerical columns.

    Output:
    Pandas DataFrame, with specified columns modified by the factor.
    """
    # Create a deep copy of the input DataFrame to avoid modifying the original.
    stressed_df = X_data.copy(deep=True)

    # Determine which columns to apply the stress to
    if vol_cols is None:
        # If None, apply stress to all numerical columns
        target_cols = stressed_df.select_dtypes(include=np.number).columns
    else:
        # If a list is provided, use those columns.
        # Pandas will handle TypeErrors if non-numerical columns are included during multiplication
        # and KeyErrors if columns do not exist.
        target_cols = vol_cols
    
    # Apply the volatility factor to the target columns.
    # This operation is robust for empty DataFrames and will propagate errors
    # for invalid column types or non-existent columns as expected by test cases.
    if len(target_cols) > 0:
        stressed_df[target_cols] = stressed_df[target_cols] * factor

    return stressed_df

import pandas as pd
import numpy as np

def stress_scenario_add_noise(X_data, noise_std_dev, vol_cols):
    """    This function introduces random Gaussian noise to specified columns of a Pandas DataFrame, simulating unpredictable measurement errors or market fluctuations. It creates a deep copy of the input DataFrame to avoid modifying the original data, preserving the original test set for baseline comparisons.
Arguments:
X_data: Pandas DataFrame, the input features to be stressed.
noise_std_dev: float, the standard deviation of the Gaussian noise to add (mean is 0).
vol_cols: list of strings, the names of the columns to add noise to. If None, noise is added to all numerical columns in the DataFrame.
Output:
Pandas DataFrame, the DataFrame with specified columns having added random Gaussian noise.
    """

    # Create a deep copy of the input DataFrame to avoid modifying the original data
    X_stressed = X_data.copy(deep=True)

    # Determine the columns to which noise will be added
    if vol_cols is None:
        # If vol_cols is None, add noise to all numerical columns
        target_cols = X_stressed.select_dtypes(include=np.number).columns
    else:
        # If vol_cols is specified, use these columns directly.
        # This will naturally raise a KeyError if a column in vol_cols does not exist
        # in the DataFrame, as per the test case requirements.
        target_cols = vol_cols

    # Add Gaussian noise to each of the target columns
    for col in target_cols:
        # Generate Gaussian noise with mean 0 and specified standard deviation.
        # The size of the noise array matches the number of rows in the DataFrame.
        noise = np.random.normal(loc=0, scale=noise_std_dev, size=len(X_stressed))
        
        # Add the generated noise to the corresponding column in the stressed DataFrame.
        # If 'col' does not exist in X_stressed (and was passed via vol_cols),
        # this line will raise a KeyError, which is expected by the tests.
        X_stressed[col] = X_stressed[col] + noise

    return X_stressed

import numpy as np
import pandas as pd

def calculate_mean_shift(baseline_predictions, stressed_predictions):
    """
    Calculates the quantitative difference between the mean of stressed predictions
    and the mean of baseline predictions.

    Arguments:
    baseline_predictions: NumPy array or Pandas Series, the model's predictions on
                          the original, unstressed test dataset.
    stressed_predictions: NumPy array or Pandas Series, the model's predictions on
                          the test dataset after applying a specific stress transformation.

    Output:
    float, a scalar value representing the difference between the mean of stressed
    predictions and baseline predictions.
    """
    mean_baseline = baseline_predictions.mean()
    mean_stressed = stressed_predictions.mean()
    return float(mean_stressed - mean_baseline)