"""Layout violations must raise instead of being silently copied."""
import numpy as np
import pytest


@pytest.fixture
def mac():
    from tensor_ops import mac

    return mac


@pytest.mark.parametrize("position", [0, 1, 2])
def test_fortran_layout_is_rejected(mac, position):
    shape = (2, 3, 4)
    args = [np.ones(shape, dtype=np.float64) for _ in range(3)]
    args[position] = np.asfortranarray(np.arange(24.0).reshape(shape))
    assert args[position].shape == shape
    assert args[position].flags.f_contiguous and not args[position].flags.c_contiguous
    with pytest.raises(ValueError):
        mac(*args)
