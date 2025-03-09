import numpy as np
import pytest

from pysgn.utils import _create_k_col


def test_create_k_col_even_integer():
    """Test _create_k_col with an even integer k."""
    k = 4
    n = 10
    result = _create_k_col(k, n)
    expected = np.full(n, k // 2, dtype=int)
    assert np.array_equal(result, expected), "Failed on even integer k"


def test_create_k_col_odd_integer():
    """Test _create_k_col with an odd integer k."""
    k = 5
    n = 10
    result = _create_k_col(k, n, random_state=42)
    assert np.all(np.isin(result, [k // 2, k // 2 + 1])), "Failed on odd integer k"
    assert np.isclose(result.mean(), k / 2, atol=0.1), (
        "Mean not as expected for odd integer k"
    )


def test_create_k_col_float():
    """Test _create_k_col with a float k."""
    k = 6.4
    n = 10
    result = _create_k_col(k, n, random_state=42)
    lower = int(np.floor(k / 2))
    upper = int(np.ceil(k / 2))
    assert np.all(np.isin(result, [lower, upper])), "Failed on float k"
    assert np.isclose(result.mean(), k / 2, atol=0.1), (
        "Mean not as expected for float k"
    )


def test_create_k_col_invalid_k():
    """Test _create_k_col with an invalid k."""
    k = "invalid"
    n = 10
    with pytest.raises(ValueError, match="k must be an integer or a float."):
        _create_k_col(k, n)


def test_create_k_col_reproducibility():
    """Test _create_k_col reproducibility with a random state."""
    k = 5
    n = 10
    random_state = 42
    result1 = _create_k_col(k, n, random_state=random_state)
    result2 = _create_k_col(k, n, random_state=random_state)
    assert np.array_equal(result1, result2), (
        "Results are not reproducible with the same random state"
    )
