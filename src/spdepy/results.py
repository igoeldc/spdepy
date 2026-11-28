from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike

from spdepy._validation import (
    valid_not_bool,
    valid_number,
    valid_real,
    valid_real_array,
    valid_times,
    valid_type,
)
from spdepy.basis import SineBasis
from spdepy.domains import Interval
from spdepy.models import LinearSpectralModel


class SimulationResult:
    """Class to store the results of a simulation."""

    def __init__(self, model: LinearSpectralModel, times: ArrayLike, coeffs: ArrayLike):
        self._model = valid_type("model", model, LinearSpectralModel)
        self._times = valid_times(times)[0].copy()
        self._coeffs = valid_real_array(
            "coeffs", coeffs, shape=(len(self._times), self._model.num_modes)
        ).copy()

        self._times.setflags(write=False)
        self._coeffs.setflags(write=False)

    @property
    def model(self) -> LinearSpectralModel:
        """The model used for the simulation."""
        return self._model

    @property
    def times(self) -> np.ndarray:
        """The times at which the simulation was evaluated."""
        return self._times.view()

    @property
    def coeffs(self) -> np.ndarray:
        """The coefficients of the simulation."""
        return self._coeffs.view()

    @property
    def eigenbasis(self) -> SineBasis:
        """The eigenbasis of the model."""
        return self._model.eigenbasis

    @property
    def domain(self) -> Interval:
        """The domain of the model."""
        return self._model.domain

    @property
    def num_modes(self) -> int:
        """The number of modes in the model."""
        return self._model.num_modes

    @property
    def num_times(self) -> int:
        """The number of time points in the simulation."""
        return len(self._times)

    @property
    def num_steps(self) -> int:
        """The number of time steps in the simulation."""
        return len(self._times) - 1

    @property
    def t_span(self) -> tuple[float, float]:
        """The time span of the simulation."""
        return (float(self._times[0]), float(self._times[-1]))

    def reconstruct(self, xs: ArrayLike) -> np.ndarray:
        """Reconstruct the solution from the coefficients."""
        basis_matrix = self.eigenbasis.matrix(self.num_modes, xs)
        return self._coeffs @ basis_matrix.T

    def snapshot(self, time_index: int, xs: ArrayLike) -> np.ndarray:
        """Return the solution at a specific time index."""
        valid_not_bool("time_index", time_index)
        valid_number("time_index", time_index, Integral)
        time_index = int(time_index)
        if not (-self.num_times <= time_index < self.num_times):
            raise IndexError(
                f"Invalid time index: {time_index} must be in the range [-{self.num_times}, {self.num_times})"
            )
        return self.eigenbasis.reconstruct(self._coeffs[time_index, :], xs)

    def trajectory(self, x: float) -> np.ndarray:
        """Return the trajectory at a specific spatial point."""
        x = valid_real("x", x)
        return self._coeffs @ self.eigenbasis.matrix(self.num_modes, [x])[0]
