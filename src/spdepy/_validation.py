import math
from numbers import Integral, Real

import numpy as np


def valid_not_bool(name, value):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"Invalid type: {name} ({type(value)}) must be a real number")


def valid_number(name, value, num_type):
    if not isinstance(value, num_type):
        raise TypeError(
            f"Invalid type: {name} ({type(value)}) must be a {num_type.__name__} number"
        )


def valid_real(name, value) -> float:
    valid_not_bool(name, value)
    valid_number(name, value, Real)
    value = float(value)

    if not math.isfinite(value):
        raise ValueError(f"Invalid value: {name} ({value}) must be finite")

    return value


def valid_pos_int(name, value) -> int:
    valid_not_bool(name, value)
    valid_number(name, value, Integral)
    if value <= 0:
        raise ValueError(f"Invalid value: {name} ({value}) must be a positive integer")

    return int(value)


def valid_real_array(name, value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in ("i", "u", "f"):
            raise TypeError(
                f"Invalid dtype: {name} ({value.dtype}) must contain real numbers, excluding booleans"
            )
    else:
        value = np.asarray(value, dtype=object)

        for v in value.flat:
            valid_not_bool(name, v)
            valid_number(name, v, Real)

    value = value.astype(np.float64, copy=False)

    if not np.all(np.isfinite(value)):
        raise ValueError(f"Invalid value: {name} must contain only finite numbers")

    return value
