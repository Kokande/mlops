"""Numeric check with an expected answer computed independently of NumPy arithmetic."""
import numpy as np
import pytest


@pytest.fixture
def mac():
    from tensor_ops import mac

    return mac


def test_non_cubic_shape_matches_pure_python_loop(mac):
    shape = (2, 3, 5)
    a = [[[0.25 * (i - j) + k for k in range(shape[2])] for j in range(shape[1])] for i in range(shape[0])]
    b = [[[-1.5 + i + 0.5 * j - 0.125 * k for k in range(shape[2])] for j in range(shape[1])] for i in range(shape[0])]
    c = [[[10.0 * i - j + 0.75 * k for k in range(shape[2])] for j in range(shape[1])] for i in range(shape[0])]
    expected = [
        [[a[i][j][k] * b[i][j][k] + c[i][j][k] for k in range(shape[2])] for j in range(shape[1])]
        for i in range(shape[0])
    ]

    out = mac(np.array(a), np.array(b), np.array(c))

    assert out.shape == shape
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)
