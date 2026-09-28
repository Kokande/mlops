"""Passing the same array object as every argument is valid and must not alias the output."""
import numpy as np
import pytest


@pytest.fixture
def mac():
    from tensor_ops import mac

    return mac


def test_same_object_for_all_inputs(mac):
    x = np.array([[[1.0, -2.0], [0.5, 3.0]]])
    snapshot = x.copy()

    out = mac(x, x, x)

    np.testing.assert_allclose(out, [[[2.0, 2.0], [0.75, 12.0]]], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(x, snapshot)
    assert not np.shares_memory(out, x)
    out.fill(0.0)
    np.testing.assert_array_equal(x, snapshot)
