import ctypes


class LinearASM(ctypes.Structure):
    _fields_ = [
        ("weight", ctypes.POINTER(ctypes.c_float)),
        ("weight_m", ctypes.c_size_t),
        ("weight_n", ctypes.c_size_t),
        ("bias", ctypes.POINTER(ctypes.c_float)),
        ("bias_n", ctypes.c_size_t),
        ("input", ctypes.POINTER(ctypes.c_float)),
        ("input_m", ctypes.c_size_t),
        ("input_n", ctypes.c_size_t),
        ("output", ctypes.POINTER(ctypes.c_float)),
        ("output_m", ctypes.c_size_t),
        ("output_n", ctypes.c_size_t),
        ("grad_input", ctypes.POINTER(ctypes.c_float)),
        ("grad_output", ctypes.POINTER(ctypes.c_float)),
        ("grad_weight", ctypes.POINTER(ctypes.c_float)),
        ("grad_bias", ctypes.POINTER(ctypes.c_float)),
    ]


class ReLUASM(ctypes.Structure):
    _fields_ = [
        ("input", ctypes.POINTER(ctypes.c_float)),
        ("input_m", ctypes.c_size_t),
        ("input_n", ctypes.c_size_t),
        ("output", ctypes.POINTER(ctypes.c_float)),
        ("grad_input", ctypes.POINTER(ctypes.c_float)),
        ("grad_output", ctypes.POINTER(ctypes.c_float)),
    ]
