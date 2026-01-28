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

    return lib
