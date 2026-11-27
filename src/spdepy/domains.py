import math

import numpy as np

from spdepy._validation import valid_real


class Interval:
    """Represents a 1D interval [a, b] in the real line."""

    def __init__(self, a: float, b: float):
        self._a = valid_real("a", a)
        self._b = valid_real("b", b)

        if (length := self.length) <= 0 or not math.isfinite(
            length
        ):  # check if length is positive and finite
            raise ValueError(
                f"Invalid interval: length ({length}) must be positive and finite"
            )

    @property
    def a(self) -> float:
        """Return the left endpoint of the interval."""
        return self._a

    @property
    def b(self) -> float:
        """Return the right endpoint of the interval."""
        return self._b

    @property
    def length(self) -> float:
        """Return the length of the interval."""
        return self.b - self.a

    @property
    def midpoint(self) -> float:
        """Return the midpoint of the interval."""
        return self.a + self.length / 2

    @property
    def dimension(self) -> int:
        """Return the dimension of the interval (always 1)."""
        return 1

    @property
    def bounds(self) -> tuple[float, float]:
        """Return the bounds of the interval as a tuple (a, b)."""
        return (self.a, self.b)

    def contains(self, x: float | np.ndarray) -> bool | np.bool_ | np.ndarray:
        """Check if x is within the interval."""
        return (self.a <= x) & (x <= self.b)

    def to_reference(self, x: float | np.ndarray) -> float | np.ndarray:
        """Map x from the interval [a, b] to the reference interval [0, pi]."""
        return (x - self.a) / self.length * np.pi

    def from_reference(self, x: float | np.ndarray) -> float | np.ndarray:
        """Map x from the reference interval [0, pi] to the interval [a, b]."""
        return self.a + (x / np.pi) * self.length
