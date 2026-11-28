import numpy as np

from spdepy._validation import valid_pos_int, valid_pos_real, valid_rng, valid_type
from spdepy.basis import SineBasis
from spdepy.covariance import SpectralCovariance


class SpectralWienerNoise:
    """Represents a spectral Wiener noise process. Generates finite-dimensional increments of Wiener noise."""

    def __init__(self, covariance: SpectralCovariance):
        self._covariance = valid_type("covariance", covariance, SpectralCovariance)

    @property
    def covariance(self) -> SpectralCovariance:
        """Return the covariance operator associated with the spectral Wiener noise."""
        return self._covariance

    @property
    def basis(self) -> SineBasis:
        """Return the basis functions for the spectral Wiener noise."""
        return self._covariance.basis

    def increment(
        self, dt: float, num_modes: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Return a single increment of the spectral Wiener noise process over a time step dt for the first num_modes modes. Mode indexing starts from 1."""
        dt = valid_pos_real("dt", dt)
        num_modes = valid_pos_int("num_modes", num_modes)
        rng = valid_rng(rng)

        eigenvalues = self._covariance.eigenvalues(num_modes)
        gaussian_samples = rng.standard_normal(size=num_modes)
        increment = np.sqrt(dt) * np.sqrt(eigenvalues) * gaussian_samples

        return increment

    def increments(
        self, dt: float, num_modes: int, num_steps: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Return multiple increments of the spectral Wiener noise process over a time step dt for the first num_modes modes. Mode indexing starts from 1. The output shape is (num_steps, num_modes). Increments are independent and correspond to the same time step dt."""
        dt = valid_pos_real("dt", dt)
        num_modes = valid_pos_int("num_modes", num_modes)
        num_steps = valid_pos_int("num_steps", num_steps)
        rng = valid_rng(rng)

        eigenvalues = self._covariance.eigenvalues(num_modes)
        gaussian_samples = rng.standard_normal(size=(num_steps, num_modes))
        increments = np.sqrt(dt) * np.sqrt(eigenvalues) * gaussian_samples

        return increments
