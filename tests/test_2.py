import pytest
import numpy as np
import pandas as pd
from definition_bfc1ac19469147e0a3cc7bbb10a0bbd9 import calculate_mean_shift

@pytest.mark.parametrize("baseline, stressed, expected", [
    # Test Case 1: Standard scenario with a positive mean shift (NumPy arrays)
    (np.array([1, 2, 3]), np.array([4, 5, 6]), 3.0),
    # Test Case 2: Standard scenario with a negative mean shift (Pandas Series)
    (pd.Series([4, 5, 6]), pd.Series([1, 2, 3]), -3.0),
    # Test Case 3: Scenario with no mean shift (equal means, including negative values)
    (np.array([-1, 0, 1]), np.array([-1, 0, 1]), 0.0),
    # Test Case 4: Edge case - Empty input arrays/series (mean of empty sequence is NaN)
    (np.array([]), np.array([]), np.nan),
    # Test Case 5: Edge case - Non-numeric data in inputs (should raise TypeError when calculating mean)
    (np.array(['a', 'b', 'c'], dtype=object), np.array([1, 2, 3]), TypeError),
])
def test_calculate_mean_shift(baseline, stressed, expected):
    if expected is np.nan:
        # Special handling for NaN comparison
        result = calculate_mean_shift(baseline, stressed)
        assert np.isnan(result)
    elif isinstance(expected, type) and issubclass(expected, Exception):
        # Special handling for expected exceptions
        with pytest.raises(expected):
            calculate_mean_shift(baseline, stressed)
    else:
        # Standard comparison for float results using numpy.testing.assert_allclose
        # to account for potential floating point inaccuracies.
        result = calculate_mean_shift(baseline, stressed)
        np.testing.assert_allclose(result, expected, rtol=1e-6)