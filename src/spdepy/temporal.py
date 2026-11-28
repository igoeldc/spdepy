import numpy as np
from numpy.typing import ArrayLike

from spdepy._validation import (
    valid_pos_real,
    valid_real_array,
    valid_rng,
    valid_type,
)
from spdepy.models import LinearSpectralModel


class ExactLinear:
    """Exact time integrator for finite-dimensional linear spectral models."""

    def __init__(self, model: LinearSpectralModel):
        self._model = valid_type("model", model, LinearSpectralModel)

    @property
    def model(self) -> LinearSpectralModel:
        """Return the underlying linear spectral model."""
        return self._model

    def transition_eigenvalues(self, dt: float) -> np.ndarray:
        """Return the transition eigenvalues of the linear spectral model over a time step dt."""
        dt = valid_pos_real("dt", dt)
        drift_eigenvalues = self._model.drift_eigenvalues
        return np.exp(drift_eigenvalues * dt)

    def transition_variances(self, dt: float) -> np.ndarray:
        """Return the transition variances of the linear spectral model over a time step dt. Handles both zero and non-zero drift eigenvalues to avoid division by zero."""
        dt = valid_pos_real("dt", dt)
        drift_eigenvalues = self._model.drift_eigenvalues
        diffusion_eigenvalues = self._model.diffusion_eigenvalues

        variances = np.empty_like(drift_eigenvalues)

        nonzero = drift_eigenvalues != 0
        zero = ~nonzero

        a = drift_eigenvalues[nonzero]
        g = diffusion_eigenvalues[nonzero]

        variances[nonzero] = g**2 * np.expm1(2 * a * dt) / (2 * a)
        variances[zero] = diffusion_eigenvalues[zero] ** 2 * dt

        return variances

    def transition_matrix(self, dt: float) -> np.ndarray:
        """Return the transition matrix of the linear spectral model over a time step dt."""
        return np.diag(self.transition_eigenvalues(dt))

    def transition_covariance_matrix(self, dt: float) -> np.ndarray:
        """Return the transition covariance matrix of the linear spectral model over a time step dt."""
        return np.diag(self.transition_variances(dt))

    def step(
        self, coeffs: ArrayLike, dt: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Return the next step of the linear spectral model given the current coefficients, time step dt, and random number generator rng."""
        coeffs = valid_real_array("coeffs", coeffs, shape=(self._model.num_modes,))
        rng = valid_rng(rng)

        transition_eigenvalues = self.transition_eigenvalues(dt)
        transition_variances = self.transition_variances(dt)
        gaussian_samples = rng.standard_normal(size=coeffs.shape)

        return (
            transition_eigenvalues * coeffs
            + np.sqrt(transition_variances) * gaussian_samples
        )
