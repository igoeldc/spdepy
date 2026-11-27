from spdepy._validation import valid_real


class Dirichlet:
    """Class representing Dirichlet boundary conditions."""

    def __init__(self, left: float = 0.0, right: float = 0.0):
        self._left = valid_real("left", left)
        self._right = valid_real("right", right)

    @property
    def left(self) -> float:
        return self._left

    @property
    def right(self) -> float:
        return self._right

    @property
    def is_homogeneous(self) -> bool:
        """Check if the Dirichlet boundary conditions are homogeneous (both left and right are zero)."""
        return self._left == 0.0 and self._right == 0.0

    @property
    def values(self) -> tuple[float, float]:
        """Return the Dirichlet boundary values as a tuple (left, right)."""
        return (self._left, self._right)
