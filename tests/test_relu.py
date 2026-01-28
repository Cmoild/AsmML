import ctypes
import numpy as np
import numpy.testing as npt
import pytest

from python.mlp import ReLU
from .structures import ReLUASM


@pytest.mark.parametrize(
    "batch_sz,n_feat",
    [
        (1, 8),
        (4, 64),
        (8, 128),
    ],
)
def test_relu_forward(lib: ctypes.CDLL, batch_sz: int, n_feat: int):
    rng = np.random.default_rng(42)
    relu = ReLU()

    inp = rng.normal(size=(batch_sz, n_feat)).astype(np.float32)
    out = np.empty((batch_sz, n_feat), dtype=np.float32)

    module = ReLUASM(
        inp.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        inp.shape[0],
        inp.shape[1],
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        None,
        None,
    )

    lib.relu_forward(ctypes.byref(module))
    ref = relu.forward(inp)

    npt.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "batch_sz,n_feat",
    [
        (1, 8),
        (4, 64),
        (8, 128),
    ],
)
def test_relu_backward(lib: ctypes.CDLL, batch_sz: int, n_feat: int):
    rng = np.random.default_rng(42)

    relu = ReLU()
    inp = rng.normal(size=(batch_sz, n_feat)).astype(np.float32)
    grad_out = rng.normal(size=(batch_sz, n_feat)).astype(np.float32)

    _ = relu.forward(inp)
    grad_in_ref = relu.update_grad(grad_out)

    grad_in = np.zeros_like(inp)

    module = ReLUASM(
        inp.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        inp.shape[0],
        inp.shape[1],
        None,
        grad_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        grad_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )

    lib.relu_update_grad(ctypes.byref(module))

    npt.assert_allclose(grad_in, grad_in_ref, rtol=1e-5)
