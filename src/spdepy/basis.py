from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import quad

from spdepy._validation import valid_pos_int, valid_real_array
from spdepy.domains import Interval


class SineBasis:
    """Represents sine basis functions on a given interval."""

    def __init__(self, domain: Interval):
        if not isinstance(domain, Interval):
            raise TypeError(
                f"Invalid type: domain ({type(domain)}) must be an instance of Interval"
            )

        self._domain = domain

    @property
    def domain(self) -> Interval:
        """Return the domain of the sine basis functions."""
        return self._domain

    @property
    def is_orthonormal(self) -> bool:
        """Check if the sine basis functions are orthonormal on the given domain."""
        return True  # Sine basis functions are orthonormal on the interval [a, b]

    def _valid_coords(self, xs: ArrayLike) -> np.ndarray:
        """Validate that xs are within the domain and return them as a numpy array."""
        xs = valid_real_array("xs", xs)

        if not np.all(self._domain.contains(xs)):
            raise ValueError(f"Invalid value: xs ({xs}) is out of the domain")

        return xs

    def evaluate(self, k: int, xs: ArrayLike) -> float | np.ndarray:
        """Evaluates the k-th sine basis function at x. Mode indexing starts from 1. Input shape is preserved."""
        k = valid_pos_int("k", k)
        xs = self._valid_coords(xs)
        T = self._domain.to_reference(xs)

        result = np.sin(k * T) * np.sqrt(2) / np.sqrt(self._domain.length)

        if result.ndim == 0:
            return result.item()
        else:
            return result

    def matrix(self, num_modes: int, xs: ArrayLike) -> np.ndarray:
        """Returns a matrix where each row corresponds to the evaluation of the sine basis functions at a point in xs. Output shape is `(len(xs), num_modes)`."""
        num_modes = valid_pos_int("num_modes", num_modes)
        xs = self._valid_coords(xs)
        if xs.ndim != 1:
            raise ValueError(
                f"Invalid value: xs ({xs}) must be a one-dimensional array"
            )

        n = np.arange(1, num_modes + 1)
        T = self._domain.to_reference(xs)

        matrix = (
            np.sin(n[None, :] * T[:, None]) * np.sqrt(2) / np.sqrt(self._domain.length) # type: ignore
        )

        return matrix

    def project(self, f: Callable[[float], float], num_modes: int) -> np.ndarray:
        """Projects a function onto the sine basis functions. `f` is a real-valued scalar function. Output contains `num_modes` coefficients."""
        if not callable(f):
            raise TypeError("Invalid type: f must be callable")

        num_modes = valid_pos_int("num_modes", num_modes)

        coeffs = np.zeros(num_modes)

        def integrand(x, k):
            return f(x) * self.evaluate(k, x)

        for k in range(1, num_modes + 1):
            integral, _error = quad(
                integrand, self._domain.a, self._domain.b, args=(k,)
            )
            coeffs[k - 1] = integral

        return valid_real_array("coeffs", coeffs)

    def reconstruct(self, coeffs: ArrayLike, xs: ArrayLike) -> np.ndarray:
        """Returns the reconstruction of a function from its sine basis coefficients at points xs. Output shape is `(len(xs),)`."""
        coeffs = valid_real_array("coeffs", coeffs)
        if coeffs.ndim != 1:
            raise ValueError(
                f"Invalid value: coeffs ({coeffs}) must be a one-dimensional array"
            )
        if coeffs.size == 0:
            raise ValueError(f"Invalid value: coeffs ({coeffs}) must not be empty")

        num_modes = coeffs.size

        return self.matrix(num_modes, xs) @ coeffs
