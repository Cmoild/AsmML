import ctypes
import pathlib
import pytest
from .structures import LinearASM, ReLUASM

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


def find_shared_object() -> pathlib.Path:
    candidates = [
        PROJECT_ROOT / "build/lib.so",
        PROJECT_ROOT / "tests/lib.so",
    ]
    for c in candidates:
        if not c:
            continue
        p = pathlib.Path(c)
        if p.exists():
            return p
    raise RuntimeError("Shared library not found (set LIBNN_PATH)")


@pytest.fixture(scope="session")
def lib():
    so_path = find_shared_object()
    lib = ctypes.CDLL(str(so_path))

    # Linear
    lib.linear_forward.argtypes = [ctypes.POINTER(LinearASM)]
    lib.linear_forward.restype = None

    lib.linear_update_grad.argtypes = [ctypes.POINTER(LinearASM)]
    lib.linear_update_grad.restype = None

    # ReLU
    lib.relu_forward.argtypes = [ctypes.POINTER(ReLUASM)]
    lib.relu_forward.restype = None

    lib.relu_update_grad.argtypes = [ctypes.POINTER(ReLUASM)]
    lib.relu_update_grad.restype = None

    # Softmax
    lib.softmax.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    lib.softmax.restype = None

    lib.ce_loss_grad.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.ce_loss_grad.restype = None

    lib.ce_loss_val.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    lib.ce_loss_val.restype = ctypes.c_float

    return lib
