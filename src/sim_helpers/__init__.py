"""
Simulation helpers for antenna array analysis.

Modules:
- arrays: Far-field evaluation for antenna arrays (NumPy/CuPy/PyTorch)
- patches: Patch antenna calculations (dimensions, feed position)
- results: Pattern analysis utilities (directivity, beamwidth, SLL)
- storage: Save antenna configurations and results
"""

from .arrays import (
    C0,
    HAS_TORCH,
    HAS_CUDA,
    evaluate_array,
    evaluate_array_torch,
    evaluate_array_xp,
    rectangular_patch_pattern_torch,
    rectangular_patch_pattern_xp,
    circular_patch_pattern_torch,
    circular_patch_pattern_xp,
)

from .patches import (
    calculate_circular_patch_resistance,
    calculate_rectangular_patch_resistance,
    calculate_circular_feed_position,
    calculate_rectangular_feed_position,
    calculate_rectangular_patch_dimensions,
    calculate_circular_patch_radius,
    get_feed_position_circular,
    get_feed_position_rectangular,
)

from .results import (
    get_theta_phi_grids,
    compute_directivity,
    compute_gain_pattern,
    find_peak_direction,
    get_phi_cut,
    get_theta_cut,
    get_principal_plane_cuts,
    compute_hpbw,
    compute_beamwidths,
    compute_sidelobe_level,
    compute_sidelobes,
    compute_power_in_cone,
    compute_power_in_solid_angle,
    compute_front_to_back_ratio,
    results,
    directivity_torch,
    cone_power_torch,
    sll_torch,
    combined_objective_torch,
)

from .storage import (
    get_output_dir,
    create_antenna_folder,
    save_element_positions,
    save_element_dimensions,
    save_element_weights,
    save_design_parameters,
    save_optimization_history,
    save_broadband_results,
    AntennaConfig,
)

__all__ = [
    # Constants
    "C0",
    "HAS_TORCH",
    "HAS_CUDA",
    # Arrays
    "evaluate_array",
    "evaluate_array_torch",
    "evaluate_array_xp",
    "rectangular_patch_pattern_torch",
    "rectangular_patch_pattern_xp",
    "circular_patch_pattern_torch",
    "circular_patch_pattern_xp",
    # Patches
    "calculate_circular_patch_resistance",
    "calculate_rectangular_patch_resistance",
    "calculate_circular_feed_position",
    "calculate_rectangular_feed_position",
    "calculate_rectangular_patch_dimensions",
    "calculate_circular_patch_radius",
    "get_feed_position_circular",
    "get_feed_position_rectangular",
    # Results
    "get_theta_phi_grids",
    "compute_directivity",
    "compute_gain_pattern",
    "find_peak_direction",
    "get_phi_cut",
    "get_theta_cut",
    "get_principal_plane_cuts",
    "compute_hpbw",
    "compute_beamwidths",
    "compute_sidelobe_level",
    "compute_sidelobes",
    "compute_power_in_cone",
    "compute_power_in_solid_angle",
    "compute_front_to_back_ratio",
    "results",
    "directivity_torch",
    "cone_power_torch",
    "sll_torch",
    "combined_objective_torch",
    # Storage
    "get_output_dir",
    "create_antenna_folder",
    "save_element_positions",
    "save_element_dimensions",
    "save_element_weights",
    "save_design_parameters",
    "save_optimization_history",
    "save_broadband_results",
    "AntennaConfig",
]
