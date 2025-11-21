import pytest
import pandas as pd
import numpy as np

# definition_d947714c63414674ae828a7031040655 block - DO NOT REPLACE or REMOVE
from definition_d947714c63414674ae828a7031040655 import stress_scenario_add_noise

def test_add_noise_to_specified_columns(mocker):
    """
    Test that noise is correctly added to specified numerical columns
    and other columns remain unchanged.
    """
    # Mock np.random.normal to return deterministic noise.
    # The function calls np.random.normal for each column with size=len(X_data).
    # X_data has 3 rows, so size will be 3.
    mocker.patch('numpy.random.normal', side_effect=[
        np.array([0.1, 0.2, 0.3]),  # Noise for 'col1'
        np.array([0.05, 0.1, 0.15]) # Noise for 'col3'
    ])

    X_data = pd.DataFrame({
        'col1': [1.0, 2.0, 3.0],
        'col2': [10.0, 20.0, 30.0],
        'col3': [100.0, 200.0, 300.0]
    })
    noise_std_dev = 0.1
    vol_cols = ['col1', 'col3']

    X_stressed = stress_scenario_add_noise(X_data, noise_std_dev, vol_cols)

    # 1. Assert that a deep copy is returned (object identity)
    assert X_stressed is not X_data
    assert X_stressed.equals(X_data) is False # Values should be different due to noise

    # 2. Assert that values in specified columns have changed as expected
    expected_col1 = pd.Series([1.0 + 0.1, 2.0 + 0.2, 3.0 + 0.3], name='col1')
    expected_col3 = pd.Series([100.0 + 0.05, 200.0 + 0.1, 300.0 + 0.15], name='col3')
    pd.testing.assert_series_equal(X_stressed['col1'], expected_col1)
    pd.testing.assert_series_equal(X_stressed['col3'], expected_col3)

    # 3. Assert that non-stressed columns remain unchanged
    pd.testing.assert_series_equal(X_stressed['col2'], X_data['col2'])

def test_add_noise_to_all_numerical_columns_if_vol_cols_is_none(mocker):
    """
    Test that if vol_cols is None, noise is added to all numerical columns
    and non-numerical columns remain unchanged.
    """
    # Mock np.random.normal to return deterministic noise.
    # X_data has 3 rows, so size will be 3.
    mocker.patch('numpy.random.normal', side_effect=[
        np.array([0.1, 0.2, 0.3]),   # Noise for 'num_col1'
        np.array([0.05, 0.1, 0.15])  # Noise for 'num_col2'
    ])

    X_data = pd.DataFrame({
        'num_col1': [1.0, 2.0, 3.0],
        'str_col': ['a', 'b', 'c'],
        'num_col2': [10.0, 20.0, 30.0],
        'bool_col': [True, False, True] # Another non-numerical type
    })
    noise_std_dev = 0.5
    vol_cols = None # Should apply to all numerical columns

    X_stressed = stress_scenario_add_noise(X_data, noise_std_dev, vol_cols)

    # 1. Assert deep copy
    assert X_stressed is not X_data
    assert X_stressed.equals(X_data) is False

    # 2. Assert that numerical columns have noise as expected
    expected_num_col1 = pd.Series([1.0 + 0.1, 2.0 + 0.2, 3.0 + 0.3], name='num_col1')
    expected_num_col2 = pd.Series([10.0 + 0.05, 20.0 + 0.1, 30.0 + 0.15], name='num_col2')
    pd.testing.assert_series_equal(X_stressed['num_col1'], expected_num_col1)
    pd.testing.assert_series_equal(X_stressed['num_col2'], expected_num_col2)

    # 3. Assert that non-numerical columns are unchanged
    pd.testing.assert_series_equal(X_stressed['str_col'], X_data['str_col'])
    pd.testing.assert_series_equal(X_stressed['bool_col'], X_data['bool_col'])


def test_empty_dataframe(mocker):
    """
    Test handling of an empty input DataFrame.
    """
    # For an empty DataFrame, len(X_data) is 0, so size=0 for np.random.normal
    mocker.patch('numpy.random.normal', return_value=np.array([]))

    X_data = pd.DataFrame(columns=['col1', 'col2'])
    noise_std_dev = 0.1
    vol_cols = ['col1'] # Even if specified, on empty DF it won't be applied

    X_stressed = stress_scenario_add_noise(X_data, noise_std_dev, vol_cols)

    # 1. Assert deep copy (even for empty DataFrame, it should be a new object)
    assert X_stressed is not X_data
    # 2. Assert that it's still an empty DataFrame with the same columns
    pd.testing.assert_frame_equal(X_stressed, X_data)

def test_non_existent_column_in_vol_cols_raises_key_error():
    """
    Test that a KeyError is raised if vol_cols contains a non-existent column.
    """
    X_data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    noise_std_dev = 0.1
    vol_cols = ['col1', 'non_existent_col'] # 'non_existent_col' does not exist

    # Expecting a KeyError when trying to access X_stressed['non_existent_col']
    with pytest.raises(KeyError, match="non_existent_col"):
        stress_scenario_add_noise(X_data, noise_std_dev, vol_cols)

def test_zero_noise_std_dev_results_in_no_change_to_values(mocker):
    """
    Test the edge case where noise_std_dev is 0, implying no actual noise
    should be added to the values.
    """
    # When std_dev is 0, np.random.normal should produce an array of zeros.
    # Mocking it explicitly confirms this behavior for the test.
    mocker.patch('numpy.random.normal', return_value=np.array([0.0, 0.0, 0.0]))

    X_data = pd.DataFrame({
        'colA': [1.0, 2.0, 3.0],
        'colB': [10.0, 20.0, 30.0]
    })
    noise_std_dev = 0.0 # Zero standard deviation
    vol_cols = ['colA']

    X_stressed = stress_scenario_add_noise(X_data, noise_std_dev, vol_cols)

    # 1. Assert deep copy
    assert X_stressed is not X_data
    # 2. Assert values in specified column are unchanged (since noise is zero)
    pd.testing.assert_series_equal(X_stressed['colA'], X_data['colA'])
    # 3. Assert values in other columns are unchanged
    pd.testing.assert_series_equal(X_stressed['colB'], X_data['colB'])
