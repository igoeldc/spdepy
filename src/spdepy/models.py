import numpy as np
from numpy.typing import ArrayLike

from spdepy._validation import (
    valid_nonneg_real,
    valid_pos_int,
    valid_pos_real,
    valid_real_array,
    valid_type,
)
from spdepy.basis import SineBasis
from spdepy.domains import Interval
from spdepy.noise import SpectralWienerNoise
from spdepy.operators import Laplacian


class LinearSpectralModel:
    """Represents a linear SPDE model in the spectral domain."""

    def __init__(
        self,
        operator: Laplacian,
        noise: SpectralWienerNoise,
        num_modes: int,
        theta: float,
        sigma: float,
    ):
        self._operator = valid_type("operator", operator, Laplacian)
        self._noise = valid_type("noise", noise, SpectralWienerNoise)
        if self._operator.domain.bounds != self._noise.basis.domain.bounds:
            raise ValueError("Operator and noise must be defined on the same domain")
        self._num_modes = valid_pos_int("num_modes", num_modes)
        self._theta = valid_pos_real("theta", theta)
        self._sigma = valid_nonneg_real("sigma", sigma)

    @property
    def operator(self) -> Laplacian:
        """Return the linear operator associated with the model."""
        return self._operator

    @property
    def noise(self) -> SpectralWienerNoise:
        """Return the noise process associated with the model."""
        return self._noise

    @property
    def eigenbasis(self) -> SineBasis:
        """Return the basis functions associated with the model."""
        return self._operator.eigenbasis

    @property
    def domain(self) -> Interval:
        """Return the domain of the model."""
        return self._operator.domain

    @property
    def num_modes(self) -> int:
        """Return the number of modes used in the model."""
        return self._num_modes

    @property
    def theta(self) -> float:
        """Return the drift coefficient of the model."""
        return self._theta

    @property
    def sigma(self) -> float:
        """Return the diffusion coefficient of the model."""
        return self._sigma

    @property
    def drift_eigenvalues(self) -> np.ndarray:
        """Return the eigenvalues of the drift operator in the sine basis."""
        return self._theta * self._operator.eigenvalues(self._num_modes)

    @property
    def drift_matrix(self) -> np.ndarray:
        """Return the matrix representation of the drift operator in the sine basis."""
        return np.diag(self.drift_eigenvalues)

    @property
    def diffusion_eigenvalues(self) -> np.ndarray:
        """Return the eigenvalues of the diffusion operator in the sine basis."""
        return self._sigma * np.sqrt(
            self._noise.covariance.eigenvalues(self._num_modes)
        )

    @property
    def diffusion_matrix(self) -> np.ndarray:
        """Return the matrix representation of the diffusion operator in the sine basis."""
        return np.diag(self.diffusion_eigenvalues)

    @property
    def diffusion_covariance(self) -> np.ndarray:
        """Return the instantaneous diffusion covariance matrix of the diffusion process."""
        return self._sigma**2 * self._noise.covariance.matrix(self._num_modes)

    def drift(self, coeffs: ArrayLike) -> np.ndarray:
        coeffs = valid_real_array("coeffs", coeffs, ndim=1, shape=(self._num_modes,))
        return self.drift_eigenvalues * coeffs
