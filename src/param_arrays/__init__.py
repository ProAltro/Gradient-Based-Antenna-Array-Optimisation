"""
Parameterized antenna array geometries for optimization.

Each ParamArray subclass defines:
- Learnable parameters and their bounds
- Encoding/decoding between normalized [0,1] and physical values
- Geometry construction from parameters
- Constraint checking (spacing, bounds)
"""

from .base import ParamArray, ParamArrayConfig
from .grid import GridParamArray, GridParamArrayConfig
from .spiral import SpiralParamArray

# from .broadband_spiral import BroadbandSpiralParamArray, BroadbandCircularParamArray  # Not implemented
from .circular import CircularParamArray, CircularParamArrayConfig
from .hexagonal import HexagonalParamArray, HexagonalParamArrayConfig
from .random_array import RandomParamArray, RandomParamArrayConfig
from .linear import (
    LinearParamArray,
    LinearParamArrayConfig,
    TaperedLinearParamArray,
    TaperedLinearParamArrayConfig,
)
from .cross import CrossParamArray, CrossParamArrayConfig

__all__ = [
    # Base
    "ParamArray",
    "ParamArrayConfig",
    # Grid
    "GridParamArray",
    "GridParamArrayConfig",
    # Spiral
    "SpiralParamArray",
    # "BroadbandSpiralParamArray",  # Not implemented
    # "BroadbandCircularParamArray",  # Not implemented
    # Circular/Ring
    "CircularParamArray",
    "CircularParamArrayConfig",
    # Hexagonal
    "HexagonalParamArray",
    "HexagonalParamArrayConfig",
    # Random/Sparse
    "RandomParamArray",
    "RandomParamArrayConfig",
    # Linear
    "LinearParamArray",
    "LinearParamArrayConfig",
    "TaperedLinearParamArray",
    "TaperedLinearParamArrayConfig",
    # Cross
    "CrossParamArray",
    "CrossParamArrayConfig",
]
