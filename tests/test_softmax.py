import ctypes
import numpy as np
import numpy.testing as npt
import pytest

from python.mlp import Softmax


@pytest.mark.parametrize(
    "batch_sz,n_feat",
    [(1, 8), (4, 64), (8, 128), (1000, 1000)],
)
def test_softmax_forward(lib: ctypes.CDLL, batch_sz: int, n_feat: int):
    rng = np.random.default_rng(42)
    softmax = Softmax()

    inp = rng.normal(size=(batch_sz, n_feat)).astype(np.float32)

    ref = softmax.forward(inp)
    lib.softmax(
        inp.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(batch_sz),
        ctypes.c_size_t(n_feat),
    )

    npt.assert_allclose(inp, ref, rtol=1e-4, atol=1e-6)
