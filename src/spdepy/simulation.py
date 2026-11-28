from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

from spdepy._validation import (
    valid_pos_int,
    valid_pos_real,
    valid_real_array,
    valid_rng,
    valid_type,
)
from spdepy.domains import Interval
from spdepy.models import LinearSpectralModel
from spdepy.temporal import ExactLinear


class Simulator:
    """A class for simulating SPDEs using the spectral Galerkin method."""

    def __init__(self, integrator: ExactLinear):
        self._integrator = valid_type("integrator", integrator, ExactLinear)

    @property
    def integrator(self) -> ExactLinear:
        """Return the integrator used for simulation."""
        return self._integrator

    @property
    def model(self) -> LinearSpectralModel:
        """Return the SPDE model associated with this simulator."""
        return self.integrator.model

    @property
    def num_modes(self) -> int:
        """Return the number of modes in the SPDE model."""
        return self.model.num_modes

    def _valid_times(self, times: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Validate the times array."""
        times = valid_real_array("times", times, ndim=1, allow_empty=False)
        dts = np.diff(times)
        if not np.all(dts > 0):
            raise ValueError("Invalid value: times must be strictly increasing")
        if not np.all(np.isfinite(dts)):
            raise ValueError(
                "Invalid value: times must have finite consecutive differences"
            )
        return times, dts

    def _valid_t_span(self, t_span: Interval | tuple[float, float]) -> Interval:
        """Validate the time span."""
        if isinstance(t_span, tuple):
            if len(t_span) != 2:
                raise ValueError("Invalid value: t_span must be a tuple of length 2")
            t_span = Interval(*t_span)
        t_span = valid_type("t_span", t_span, Interval)
        return t_span

    def simulate(
        self,
        initial_condition: ArrayLike | Callable[[float], float],
        rng: np.random.Generator,
        *,
        times: ArrayLike | None = None,
        t_span: Interval | tuple[float, float] | None = None,
        dt: float | None = None,
        num_steps: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate the SPDE given an initial condition and times."""
        rng = valid_rng(rng)
        times, dts = self._resolve_times(times, t_span, dt, num_steps)
        initial_condition = self._resolve_initial_condition(initial_condition)

        trajectory = np.empty((len(times), self.num_modes), dtype=np.float64)
        trajectory[0] = initial_condition

        for t in range(1, len(times)):
            trajectory[t] = self._integrator.step(trajectory[t - 1], dts[t - 1], rng)

        return times, trajectory

    def _times_from_dt(
        self, t_span: Interval | tuple[float, float], dt: float
    ) -> np.ndarray:
        """Generate an array of times from a time span and time step."""
        t_span = self._valid_t_span(t_span)
        dt = valid_pos_real("dt", dt)

        num_steps = int(np.floor((t_span.b - t_span.a) / dt))
        last_full_step = t_span.a + num_steps * dt

        if last_full_step >= t_span.b:
            return np.linspace(t_span.a, t_span.b, num_steps + 1, dtype=np.float64)
        else:
            times = np.linspace(
                t_span.a, last_full_step, num_steps + 1, dtype=np.float64
            )
            return np.append(times, t_span.b)

    def _times_from_num_steps(
        self, t_span: Interval | tuple[float, float], num_steps: int
    ) -> np.ndarray:
        """Generate an array of times from a time span and number of steps."""
        t_span = self._valid_t_span(t_span)
        num_steps = valid_pos_int("num_steps", num_steps)

        return np.linspace(t_span.a, t_span.b, num_steps + 1, dtype=np.float64)

    def _resolve_times(
        self,
        times: ArrayLike | None,
        t_span: Interval | tuple[float, float] | None,
        dt: float | None,
        num_steps: int | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resolve the times array based on the provided parameters."""
        if times is not None:
            if t_span is not None or dt is not None or num_steps is not None:
                raise ValueError(
                    "Invalid combination: Provide either times or t_span/dt/num_steps, not both."
                )

            return self._valid_times(times)

        if dt is not None and num_steps is not None:
            raise ValueError(
                "Invalid combination: Provide either dt or num_steps, not both."
            )

        if t_span is None:
            raise ValueError("Invalid combination: Provide either times or t_span.")

        if dt is not None:
            times = self._times_from_dt(t_span, dt)
        elif num_steps is not None:
            times = self._times_from_num_steps(t_span, num_steps)
        else:
            raise ValueError(
                "Invalid combination: t_span requires either dt or num_steps."
            )

        return self._valid_times(times)

    def _resolve_initial_condition(
        self, initial_condition: ArrayLike | Callable[[float], float]
    ) -> np.ndarray:
        """Resolve the initial condition to a numpy array."""
        if callable(initial_condition):
            initial_condition = self.model.eigenbasis.project(
                initial_condition, self.num_modes
            )

        return valid_real_array(
            "initial_condition", initial_condition, shape=(self.num_modes,)
        )
