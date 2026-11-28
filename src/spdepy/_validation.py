import math
from numbers import Integral, Real
from typing import TypeVar

T = TypeVar("T")

import numpy as np
from numpy.typing import ArrayLike


def valid_not_bool(name: str, value: object) -> None:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"Invalid type: {name} ({type(value)}) must be a real number")


def valid_number(name: str, value: object, num_type: type) -> None:
    if not isinstance(value, num_type):
        raise TypeError(
            f"Invalid type: {name} ({type(value)}) must be a {num_type.__name__} number"
        )


def valid_pos_int(name: str, value: int) -> int:
    valid_not_bool(name, value)
    valid_number(name, value, Integral)
    if value <= 0:
        raise ValueError(f"Invalid value: {name} ({value}) must be a positive integer")

    return int(value)


def valid_real(name: str, value: float) -> float:
    valid_not_bool(name, value)
    valid_number(name, value, Real)
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Invalid value: {name} ({value}) must be finite")

    return value


def valid_nonneg_real(name: str, value: float) -> float:
    value = valid_real(name, value)
    if value < 0:
        raise ValueError(f"Invalid value: {name} ({value}) must be non-negative")

    return value


def valid_pos_real(name: str, value: float) -> float:
    value = valid_real(name, value)
    if value <= 0:
        raise ValueError(f"Invalid value: {name} ({value}) must be positive")

    return value


def valid_real_array(
    name: str,
    value: ArrayLike,
    *,
    ndim: int | None = None,
    shape: tuple[int, ...] | None = None,
    allow_empty: bool = True,
) -> np.ndarray:
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

    if ndim is not None:
        value = valid_dims(name, value, ndim)

    if shape is not None:
        value = valid_shape(name, value, shape)

    if not allow_empty:
        value = valid_not_empty(name, value)

    return value


def valid_type(name: str, value: object, expected_type: type[T]) -> T:
    if not isinstance(value, expected_type):
        raise TypeError(
            f"Invalid type: {name} ({type(value)}) must be an instance of {expected_type.__name__}"
        )

    return value


def valid_rng(rng: object) -> np.random.Generator:
    """Validate that rng is an instance of np.random.Generator."""
    return valid_type("rng", rng, np.random.Generator)


def valid_dims(name: str, value: np.ndarray, expected_dims: int) -> np.ndarray:
    value = valid_type(name, value, np.ndarray)
    if value.ndim != expected_dims:
        raise ValueError(
            f"Invalid value: {name} ({value.ndim}) must have {expected_dims} dimensions"
        )

    return value


def valid_shape(
    name: str, value: np.ndarray, expected_shape: tuple[int, ...]
) -> np.ndarray:
    value = valid_type(name, value, np.ndarray)
    if value.shape != expected_shape:
        raise ValueError(
            f"Invalid value: {name} ({value.shape}) must have shape {expected_shape}"
        )

    return value


def valid_not_empty(name: str, value: np.ndarray) -> np.ndarray:
    value = valid_type(name, value, np.ndarray)
    if value.size == 0:
        raise ValueError(f"Invalid value: {name} must not be empty")

    return value
