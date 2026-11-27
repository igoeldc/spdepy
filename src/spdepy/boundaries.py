import math
import numbers


class Dirichlet:
    """Class representing Dirichlet boundary conditions."""
    
    def __init__(self, left: float = 0.0, right: float = 0.0):
        if isinstance(left, bool) or isinstance(right, bool): # check if left or right are bool
            raise TypeError(f"Invalid types: left ({type(left)}) and right ({type(right)}) must be real numerical scalars, not booleans")
        
        if not isinstance(left, numbers.Real) or not isinstance(right, numbers.Real): # check if left and right are real numerical scalars
            raise TypeError(f"Invalid types: left ({type(left)}) and right ({type(right)}) must be real numerical scalars, excluding booleans")
        
        self._left = float(left)
        self._right = float(right)
        
        if not math.isfinite(self._left) or not math.isfinite(self._right): # check if left and right are finite
            raise ValueError(f"Invalid boundary conditions: left ({self._left}) and right ({self._right}) must be finite")
    
    @property
    def left(self) -> float:
        return self._left

    @property
    def right(self) -> float:
        return self._right
    
    @property
    def is_homogeneous(self) -> bool:
        """Check if the Dirichlet boundary conditions are homogeneous (both left and right are zero)."""
        return (self._left == 0.0 and self._right == 0.0)

    @property
    def values(self) -> tuple[float, float]:
        """Return the Dirichlet boundary values as a tuple (left, right)."""
        return (self._left, self._right)
    