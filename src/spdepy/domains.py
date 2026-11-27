import math
import numbers

import numpy as np

# TODO: vectorized membership
# TODO: type annotations don't reflect array support

class Interval:
    """Represents a 1D interval [a, b] in the real line."""
    
    def __init__(self, a: float, b: float):
        if isinstance(a, bool) or isinstance(b, bool): # check if a or b are bool
            raise TypeError(f"Invalid types: a ({type(a)}) and b ({type(b)}) must be real numerical scalars, not booleans")
        
        if not isinstance(a, numbers.Real) or not isinstance(b, numbers.Real): # check if a and b are real numerical scalars
            raise TypeError(f"Invalid types: a ({type(a)}) and b ({type(b)}) must be real numerical scalars, excluding booleans")
        
        self._a = float(a)
        self._b = float(b)
        
        if not math.isfinite(self._a) or not math.isfinite(self._b): # check if a and b are finite
            raise ValueError(f"Invalid interval: a ({self._a}) and b ({self._b}) must be finite")
        
        if (len := self.length) <= 0 or not math.isfinite(len): # check if length is positive and finite
            raise ValueError(f"Invalid interval: length ({len}) must be positive and finite")
    
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
        return self.a + self.length/2
    
    @property
    def dimension(self) -> int:
        """Return the dimension of the interval (always 1)."""
        return 1
    
    @property
    def bounds(self) -> tuple[float, float]:
        """Return the bounds of the interval as a tuple (a, b)."""
        return (self.a, self.b)
    
    def contains(self, x: float) -> bool:
        """Check if x is within the interval."""
        return self.a <= x <= self.b
    
    def to_reference(self, x: float) -> float:
        """Map x from the interval [a, b] to the reference interval [0, pi]."""
        return (x - self.a) / self.length * np.pi
    
    def from_reference(self, x: float) -> float:
        """Map x from the reference interval [0, pi] to the interval [a, b]."""
        return self.a + (x / np.pi) * self.length