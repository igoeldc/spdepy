from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

from spdepy._validation import (
    valid_nonneg_real,
    valid_pos_int,
    valid_real_array,
)
from spdepy.basis import SineBasis


class SpectralCovariance:
    """Represents a covariance operator that is diagonal in a given sine basis."""

    def __init__(self, basis: SineBasis, eigenvalues: Callable[[int], float]):
        """
        Initialize the spectral covariance operator with a sine basis and an eigenvalue function.

        The eigenvalue function must be deterministic, accept a positive integer mode index, and return a finite, non-negative real number. Eigenvalues are validated upon evaluation rather than during initialization.
        """
        if not isinstance(basis, SineBasis):
            raise TypeError(
                f"Invalid type: basis ({type(basis)}) must be an instance of SineBasis"
            )
        if not callable(eigenvalues):
            raise TypeError(
                f"Invalid type: eigenvalues ({type(eigenvalues)}) must be a callable function"
            )
        self._basis = basis
        self._eigenvalues = eigenvalues

    @property
    def basis(self) -> SineBasis:
        """Return the sine basis associated with the spectral covariance operator."""
        return self._basis

    def eigenvalue(self, k: int) -> float:
        """Return the eigenvalue of the spectral covariance operator for the k-th mode. Mode indexing starts from 1."""
        k = valid_pos_int("k", k)
        return valid_nonneg_real(f"q_{k}", self._eigenvalues(k))

    def eigenvalues(self, num_modes: int) -> np.ndarray:
        """Return the eigenvalues of the spectral covariance operator for the first `num_modes` modes. Mode indexing starts from 1."""
        num_modes = valid_pos_int("num_modes", num_modes)
        eigenvalues = np.empty(num_modes, dtype=np.float64)
        for k in range(1, num_modes + 1):
            eigenvalues[k - 1] = self.eigenvalue(k)
        return eigenvalues

    def matrix(self, num_modes: int) -> np.ndarray:
        """Return the matrix representation of the spectral covariance operator in the sine basis for the first `num_modes` modes. Mode indexing starts from 1."""
        return np.diag(self.eigenvalues(num_modes))

    def sqrt_matrix(self, num_modes: int) -> np.ndarray:
        """Return the square root of the matrix representation of the spectral covariance operator in the sine basis for the first `num_modes` modes. Mode indexing starts from 1."""
        return np.diag(np.sqrt(self.eigenvalues(num_modes)))

    def apply(self, coeffs: ArrayLike) -> np.ndarray:
        """Applies the spectral covariance operator to a vector of coefficients."""
        coeffs = valid_real_array("coeffs", coeffs)
        if coeffs.ndim != 1:
            raise ValueError(
                f"Invalid value: coeffs ({coeffs}) must be a one-dimensional array"
            )
        if coeffs.size == 0:
            raise ValueError(f"Invalid value: coeffs ({coeffs}) must not be empty")
        num_modes = coeffs.size
        eigenvalues = self.eigenvalues(num_modes)
        return eigenvalues * coeffs

    def trace(self, num_modes: int) -> float:
        """Return the trace of the truncated spectral covariance operator for the first `num_modes` modes. Mode indexing starts from 1."""
        return float(np.sum(self.eigenvalues(num_modes)))
