"""
CST Constructors package for building antenna structures in CST Microwave Studio.
"""

from .patches import (
    C0,
    get_feed_position_circular,
    get_feed_position_rectangular,
)

from .cst_patch_constructor import (
    DEFAULTS,
    create_circular_array,
    create_rectangular_array,
    grid_positions,
    spiral_positions,
)

__all__ = [
    "C0",
    "DEFAULTS",
    "get_feed_position_circular",
    "get_feed_position_rectangular",
    "create_circular_array",
    "create_rectangular_array",
    "grid_positions",
    "spiral_positions",
]
