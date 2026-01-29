import ctypes
import numpy as np
import numpy.testing as npt
import pytest

from python.mlp import Softmax, compute_loss


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


@pytest.mark.parametrize(
    "batch_sz,n_feat",
    [(1, 10), (2, 8), (8, 128), (1000, 1000)],
)
def test_ce_loss_function(lib: ctypes.CDLL, batch_sz: int, n_feat: int):
    rng = np.random.default_rng(42)
    softmax = Softmax()

    x = rng.normal(size=(batch_sz, n_feat)).astype(np.float32)
    pred = softmax.forward(x)
    true = np.eye(batch_sz, n_feat, dtype=np.float32)

    loss_ref, grad_loss_ref = compute_loss(true, pred)
    grad_loss = np.empty_like(grad_loss_ref)
    lib.ce_loss_grad(
        true.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        pred.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(batch_sz),
        ctypes.c_size_t(n_feat),
        grad_loss.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    print(true)
    print(pred)
    print(batch_sz)
    print(n_feat)
    loss = lib.ce_loss_val(
        true.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        pred.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(batch_sz),
        ctypes.c_size_t(n_feat),
    )
    loss = np.float32(loss)

    npt.assert_allclose(grad_loss, grad_loss_ref, rtol=1e-5, atol=1e-6)
    npt.assert_allclose(loss, loss_ref, rtol=1e-5, atol=1e-6)
