import ctypes
import numpy as np
import numpy.testing as npt
import pytest

from python.mlp import Linear
from .structures import LinearASM


@pytest.mark.parametrize(
    "batch_sz,in_feat,out_feat",
    [(1, 8, 4), (4, 64, 16), (8, 128, 32), (13, 345, 23)],
)
def test_linear_forward(lib: ctypes.CDLL, batch_sz: int, in_feat: int, out_feat: int):
    rng = np.random.default_rng(42)
    fc = Linear(in_feat, out_feat)

    inp = rng.normal(size=(batch_sz, in_feat)).astype(np.float32)
    out = np.empty((batch_sz, out_feat), dtype=np.float32)

    module = LinearASM(
        fc.weight.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        fc.weight.shape[0],
        fc.weight.shape[1],
        fc.bias.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        fc.bias.shape[0],
        inp.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        inp.shape[0],
        inp.shape[1],
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.shape[0],
        out.shape[1],
        None,
        None,
        None,
        None,
    )

    lib.linear_forward(ctypes.byref(module))
    ref = fc.forward(inp)

    npt.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "batch_sz,in_feat,out_feat",
    [
        (1, 8, 4),
        (4, 64, 16),
        (25, 231, 78),
    ],
)
def test_linear_backward(lib: ctypes.CDLL, batch_sz: int, in_feat: int, out_feat: int):
    rng = np.random.default_rng(42)

    fc = Linear(in_feat, out_feat)
    x = rng.normal(size=(batch_sz, in_feat)).astype(np.float32)
    grad_out = rng.normal(size=(batch_sz, out_feat)).astype(np.float32)

    _ = fc.forward(x)
    grad_in_ref = fc.update_grad(grad_out)

    grad_in = np.zeros_like(x)
    grad_w = np.zeros_like(fc.weight)
    grad_b = np.zeros_like(fc.bias)

    module = LinearASM(
        fc.weight.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out_feat,
        in_feat,
        fc.bias.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out_feat,
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        batch_sz,
        in_feat,
        None,
        batch_sz,
        out_feat,
        grad_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        grad_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        grad_w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        grad_b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )

    lib.linear_update_grad(ctypes.byref(module))
    npt.assert_allclose(grad_in, grad_in_ref, rtol=1e-3)
    npt.assert_allclose(grad_w, fc.grad_weight, rtol=1e-3)
    npt.assert_allclose(grad_b, fc.grad_bias, rtol=1e-3)
