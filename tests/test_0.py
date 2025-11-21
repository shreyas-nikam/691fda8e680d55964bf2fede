import pytest
import pandas as pd
import numpy as np # Required for functions like select_dtypes(include=np.number) if implemented

from definition_2041379442914893ba3f2719f5972927 import stress_scenario_volatility

@pytest.mark.parametrize("input_args, expected", [
    # Test Case 1: Basic functionality - Stress a specific numerical column with a positive factor
    (
        (pd.DataFrame({'A': [1, 2], 'B': [3, 4]}), 2.0, ['A']),
        pd.DataFrame({'A': [2.0, 4.0], 'B': [3, 4]})
    ),

    # Test Case 2: `vol_cols` is None - Stress all numerical columns in the DataFrame
    # Non-numerical 'C' should remain unaffected.
    (
        (pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': ['x', 'y']}), 0.5, None),
        pd.DataFrame({'A': [0.5, 1.0], 'B': [1.5, 2.0], 'C': ['x', 'y']})
    ),

    # Test Case 3: Empty DataFrame - Should return an empty DataFrame with the same structure
    (
        (pd.DataFrame(columns=['A', 'B'], dtype=float), 3.0, ['A']),
        pd.DataFrame(columns=['A', 'B'], dtype=float)
    ),

    # Test Case 4: Explicitly specified `vol_cols` includes a non-numerical column.
    # Attempting to multiply a non-numerical (e.g., string) column by a float should raise a TypeError.
    (
        (pd.DataFrame({'A': [1, 2], 'B': ['x', 'y']}), 2.0, ['A', 'B']),
        TypeError
    ),
    
    # Test Case 5: Error handling - `X_data` is not a Pandas DataFrame.
    # Attempting to call `.copy()` on a non-DataFrame object (like a list) should raise an AttributeError.
    (
        ([1, 2, 3], 2.0, ['A']), # X_data as a list
        AttributeError
    ),
])
def test_stress_scenario_volatility(input_args, expected):
    X_data_input, factor, vol_cols = input_args
    try:
        result_df = stress_scenario_volatility(X_data_input, factor, vol_cols)
        # If no exception, it should be a DataFrame, so assert equality using pandas testing utilities
        pd.testing.assert_frame_equal(result_df, expected)
    except Exception as e:
        # If an exception is expected, verify its type
        assert isinstance(e, expected)
