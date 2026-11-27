import numpy as np
from numpy.typing import ArrayLike

from spdepy._validation import valid_pos_int, valid_real_array
from spdepy.basis import SineBasis
from spdepy.boundaries import Dirichlet
from spdepy.domains import Interval


class Laplacian:
    """Represents the 1D Laplacian operator with homogeneous Dirichlet boundary conditions."""

    def __init__(self, domain: Interval, boundary_conditions: Dirichlet):
        if not isinstance(domain, Interval):
            raise TypeError(
                f"Invalid type: domain ({type(domain)}) must be an instance of Interval"
            )

        if not isinstance(boundary_conditions, Dirichlet):
            raise TypeError(
                f"Invalid type: boundary_conditions ({type(boundary_conditions)}) must be an instance of Dirichlet"
            )

        if not boundary_conditions.is_homogeneous:
            raise ValueError(
                "Invalid value: boundary_conditions must be homogeneous (both left and right values must be zero)"
            )

        self._domain = domain
        self._bc = boundary_conditions
        self._basis = SineBasis(domain)

    @property
    def domain(self) -> Interval:
        """Return the domain of the Laplacian operator."""
        return self._domain

    @property
    def boundary_conditions(self) -> Dirichlet:
        """Return the boundary conditions of the Laplacian operator."""
        return self._bc

    @property
    def eigenbasis(self) -> SineBasis:
        """Return the sine basis associated with the Laplacian operator."""
        return self._basis

    def eigenvalue(self, k: int) -> float:
        """Return the eigenvalue of the Laplacian operator for the k-th mode. Mode indexing starts from 1."""
        k = valid_pos_int("k", k)
        return -((k * np.pi / self._domain.length) ** 2)

    def eigenvalues(self, num_modes: int) -> np.ndarray:
        """Return the eigenvalues of the Laplacian operator for the first `num_modes` modes. Mode indexing starts from 1."""
        num_modes = valid_pos_int("num_modes", num_modes)
        ks = np.arange(1, num_modes + 1, dtype=np.float64)
        length = self._domain.length
        return -((ks * np.pi / length) ** 2)

    def matrix(self, num_modes: int) -> np.ndarray:
        """Return the matrix representation of the Laplacian operator in the sine basis for the first `num_modes` modes. Mode indexing starts from 1."""
        return np.diag(self.eigenvalues(num_modes))

    def apply(self, coeffs: ArrayLike) -> np.ndarray:
        """Apply the Laplacian to sine basis coefficients and return the resulting spectral coefficients."""
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
