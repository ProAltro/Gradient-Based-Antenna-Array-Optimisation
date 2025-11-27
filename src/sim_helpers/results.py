from __future__ import annotations

import math
import numpy as np
from typing import Tuple, Dict, Any, Optional

# Lazy torch import
_torch = None


def _get_torch():
    global _torch
    if _torch is None:
        import torch

        _torch = torch
    return _torch


def get_theta_phi_grids(
    theta_samples: int = 181,
    phi_step_deg: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    theta_deg = np.linspace(0, 90, theta_samples)
    phi_deg = np.arange(0, 360, phi_step_deg)
    theta_rad = np.deg2rad(theta_deg)
    phi_rad = np.deg2rad(phi_deg)
    return theta_deg, phi_deg, theta_rad, phi_rad


def compute_directivity(
    power: np.ndarray,
    theta_samples: int = 181,
    phi_step_deg: float = 1.0,
) -> Tuple[float, float]:

    theta_deg, phi_deg, theta_rad, phi_rad = get_theta_phi_grids(
        theta_samples, phi_step_deg
    )

    dtheta = theta_rad[1] - theta_rad[0]
    dphi = phi_rad[1] - phi_rad[0]

    # Integrate power over upper hemisphere: ∫∫ U(θ,φ) sin(θ) dθ dφ
    sin_theta = np.sin(theta_rad)[:, np.newaxis]
    radiated_power = np.sum(power * sin_theta) * dtheta * dphi

    peak_power = power.max()
    directivity = 4 * np.pi * peak_power / radiated_power
    directivity_dbi = 10 * np.log10(directivity)

    return float(directivity), float(directivity_dbi)


def compute_gain_pattern(
    power: np.ndarray,
    theta_samples: int = 181,
    phi_step_deg: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:

    theta_deg, phi_deg, theta_rad, phi_rad = get_theta_phi_grids(
        theta_samples, phi_step_deg
    )

    dtheta = theta_rad[1] - theta_rad[0]
    dphi = phi_rad[1] - phi_rad[0]

    sin_theta = np.sin(theta_rad)[:, np.newaxis]
    radiated_power = np.sum(power * sin_theta) * dtheta * dphi

    gain_linear = 4 * np.pi * power / radiated_power
    gain_dbi = 10 * np.log10(gain_linear + 1e-12)

    return gain_linear, gain_dbi


def find_peak_direction(
    power: np.ndarray,
    theta_samples: int = 181,
    phi_step_deg: float = 1.0,
) -> Dict[str, float]:

    theta_deg, phi_deg, _, _ = get_theta_phi_grids(theta_samples, phi_step_deg)

    peak_idx = np.unravel_index(np.argmax(power), power.shape)
    return {
        "theta_deg": float(theta_deg[peak_idx[0]]),
        "phi_deg": float(phi_deg[peak_idx[1]]),
        "peak_power": float(power[peak_idx]),
    }


# =============================================================================
# Pattern Cuts
# =============================================================================


def get_phi_cut(
    power: np.ndarray,
    phi_deg_target: float,
    theta_samples: int = 181,
    phi_step_deg: float = 1.0,
    return_db: bool = True,
    normalized: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:

    theta_deg, phi_deg, _, _ = get_theta_phi_grids(theta_samples, phi_step_deg)

    phi_idx = int(round(phi_deg_target / phi_step_deg)) % len(phi_deg)
    cut = power[:, phi_idx].copy()

    if normalized:
        cut = cut / power.max()

    if return_db:
        cut = 10 * np.log10(cut + 1e-12)

    return theta_deg, cut


def get_theta_cut(
    power: np.ndarray,
    theta_deg_target: float,
    theta_samples: int = 181,
    phi_step_deg: float = 1.0,
    return_db: bool = True,
    normalized: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:

    theta_deg, phi_deg, _, _ = get_theta_phi_grids(theta_samples, phi_step_deg)

    theta_idx = int(round(theta_deg_target / (90 / (theta_samples - 1))))
    theta_idx = min(theta_idx, theta_samples - 1)
    cut = power[theta_idx, :].copy()

    if normalized:
        cut = cut / power.max()

    if return_db:
        cut = 10 * np.log10(cut + 1e-12)

    return phi_deg, cut


def get_principal_plane_cuts(
    power: np.ndarray,
    theta_samples: int = 181,
    phi_step_deg: float = 1.0,
    return_db: bool = True,
    normalized: bool = True,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:

    e_plane = get_phi_cut(
        power, 0.0, theta_samples, phi_step_deg, return_db, normalized
    )
    h_plane = get_phi_cut(
        power, 90.0, theta_samples, phi_step_deg, return_db, normalized
    )

    return {
        "e_plane": e_plane,
        "h_plane": h_plane,
    }


# =============================================================================
# Beamwidth Analysis
# =============================================================================


def compute_hpbw(
    power: np.ndarray,
    phi_deg_target: float = 0.0,
    theta_samples: int = 181,
    phi_step_deg: float = 1.0,
) -> float:

    theta_deg, cut = get_phi_cut(
        power,
        phi_deg_target,
        theta_samples,
        phi_step_deg,
        return_db=False,
        normalized=True,
    )

    half_power = 0.5
    peak_idx = np.argmax(cut)

    # Find -3dB points on both sides of peak
    left_idx = None
    right_idx = None

    # Search left of peak
    for i in range(peak_idx, -1, -1):
        if cut[i] < half_power:
            left_idx = i
            break

    # Search right of peak
    for i in range(peak_idx, len(cut)):
        if cut[i] < half_power:
            right_idx = i
            break

    if left_idx is None or right_idx is None:
        return float("nan")

    # Linear interpolation for more accurate values
    if left_idx > 0:
        frac = (half_power - cut[left_idx]) / (cut[left_idx + 1] - cut[left_idx])
        left_angle = theta_deg[left_idx] + frac * (
            theta_deg[left_idx + 1] - theta_deg[left_idx]
        )
    else:
        left_angle = theta_deg[left_idx]

    if right_idx < len(cut) - 1:
        frac = (half_power - cut[right_idx - 1]) / (cut[right_idx] - cut[right_idx - 1])
        right_angle = theta_deg[right_idx - 1] + frac * (
            theta_deg[right_idx] - theta_deg[right_idx - 1]
        )
    else:
        right_angle = theta_deg[right_idx]

    return float(right_angle - left_angle)


def compute_beamwidths(
    power: np.ndarray,
    theta_samples: int = 181,
    phi_step_deg: float = 1.0,
) -> Dict[str, float]:

    return {
        "hpbw_e_plane": compute_hpbw(power, 0.0, theta_samples, phi_step_deg),
        "hpbw_h_plane": compute_hpbw(power, 90.0, theta_samples, phi_step_deg),
    }


# =============================================================================
# Sidelobe Analysis
# =============================================================================


def compute_sidelobe_level(
    power: np.ndarray,
    phi_deg_target: float = 0.0,
    theta_samples: int = 181,
    phi_step_deg: float = 1.0,
) -> Dict[str, float]:

    theta_deg, cut_db = get_phi_cut(
        power,
        phi_deg_target,
        theta_samples,
        phi_step_deg,
        return_db=True,
        normalized=True,
    )

    # Find main lobe peak
    peak_idx = np.argmax(cut_db)

    # Find first null after main lobe (where pattern goes below -20 dB or starts rising)
    null_idx = peak_idx
    for i in range(peak_idx + 1, len(cut_db) - 1):
        if cut_db[i] < -20 or (cut_db[i + 1] > cut_db[i] and cut_db[i] < -3):
            null_idx = i
            break

    # Find highest sidelobe after the null
    if null_idx < len(cut_db) - 1:
        sidelobe_region = cut_db[null_idx:]
        sidelobe_peak_idx = np.argmax(sidelobe_region) + null_idx
        sll_db = cut_db[sidelobe_peak_idx]
        sll_theta = theta_deg[sidelobe_peak_idx]
    else:
        sll_db = float("nan")
        sll_theta = float("nan")

    return {
        "sll_db": float(sll_db),
        "sll_theta_deg": float(sll_theta),
    }


def compute_sidelobes(
    power: np.ndarray,
    theta_samples: int = 181,
    phi_step_deg: float = 1.0,
) -> Dict[str, Dict[str, float]]:

    return {
        "e_plane": compute_sidelobe_level(power, 0.0, theta_samples, phi_step_deg),
        "h_plane": compute_sidelobe_level(power, 90.0, theta_samples, phi_step_deg),
    }


# =============================================================================
# Power Integration
# =============================================================================


def compute_power_in_cone(
    power: np.ndarray,
    half_angle_deg: float,
    theta_samples: int = 181,
    phi_step_deg: float = 1.0,
) -> Tuple[float, float]:

    theta_deg, phi_deg, theta_rad, phi_rad = get_theta_phi_grids(
        theta_samples, phi_step_deg
    )

    dtheta = theta_rad[1] - theta_rad[0]
    dphi = phi_rad[1] - phi_rad[0]
    sin_theta = np.sin(theta_rad)[:, np.newaxis]

    # Total radiated power
    total_power = np.sum(power * sin_theta) * dtheta * dphi

    # Power within cone
    cone_mask = theta_deg <= half_angle_deg
    cone_power = np.sum(power[cone_mask, :] * sin_theta[cone_mask]) * dtheta * dphi

    fraction = cone_power / total_power
    return float(fraction), float(fraction * 100)


def compute_power_in_solid_angle(
    power: np.ndarray,
    theta_min_deg: float,
    theta_max_deg: float,
    phi_min_deg: float,
    phi_max_deg: float,
    theta_samples: int = 181,
    phi_step_deg: float = 1.0,
) -> Tuple[float, float]:

    theta_deg, phi_deg, theta_rad, phi_rad = get_theta_phi_grids(
        theta_samples, phi_step_deg
    )

    dtheta = theta_rad[1] - theta_rad[0]
    dphi = phi_rad[1] - phi_rad[0]
    sin_theta = np.sin(theta_rad)[:, np.newaxis]

    # Total radiated power
    total_power = np.sum(power * sin_theta) * dtheta * dphi

    # Create mask for the solid angle region
    theta_mask = (theta_deg >= theta_min_deg) & (theta_deg <= theta_max_deg)
    phi_mask = (phi_deg >= phi_min_deg) & (phi_deg <= phi_max_deg)

    # Handle phi wrap-around
    if phi_min_deg > phi_max_deg:
        phi_mask = (phi_deg >= phi_min_deg) | (phi_deg <= phi_max_deg)

    # Apply masks
    region_power = 0.0
    for i, t_in in enumerate(theta_mask):
        if t_in:
            for j, p_in in enumerate(phi_mask):
                if p_in:
                    region_power += power[i, j] * np.sin(theta_rad[i])

    region_power *= dtheta * dphi
    fraction = region_power / total_power

    return float(fraction), float(fraction * 100)


def compute_front_to_back_ratio(
    power: np.ndarray,
    theta_samples: int = 181,
    phi_step_deg: float = 1.0,
) -> float:

    front_power = power[0, :].mean()

    back_power = power[-1, :].mean()

    if back_power < 1e-12:
        return float("inf")

    fb_ratio = 10 * np.log10(front_power / back_power)
    return float(fb_ratio)


# =============================================================================
# Top-Level Results Function
# =============================================================================


def results(
    power: np.ndarray,
    theta_samples: int = 181,
    phi_step_deg: float = 1.0,
    verbose: bool = True,
) -> Dict[str, Any]:

    directivity, directivity_dbi = compute_directivity(
        power, theta_samples, phi_step_deg
    )
    peak = find_peak_direction(power, theta_samples, phi_step_deg)
    beamwidths = compute_beamwidths(power, theta_samples, phi_step_deg)
    sidelobes = compute_sidelobes(power, theta_samples, phi_step_deg)
    fb_ratio = compute_front_to_back_ratio(power, theta_samples, phi_step_deg)

    _, power_10deg = compute_power_in_cone(power, 10.0, theta_samples, phi_step_deg)
    _, power_30deg = compute_power_in_cone(power, 30.0, theta_samples, phi_step_deg)

    result = {
        "directivity_linear": directivity,
        "directivity_dbi": directivity_dbi,
        "peak_direction": peak,
        "beamwidths": beamwidths,
        "sidelobes": sidelobes,
        "front_to_back_db": fb_ratio,
        "power_in_10deg_cone_percent": power_10deg,
        "power_in_30deg_cone_percent": power_30deg,
    }

    if verbose:
        print("\n" + "=" * 50)
        print("ANTENNA PATTERN RESULTS")
        print("=" * 50)
        print(f"  Directivity:        {directivity_dbi:.2f} dBi")
        print(
            f"  Peak direction:     θ={peak['theta_deg']:.1f}°, φ={peak['phi_deg']:.1f}°"
        )
        print(f"  HPBW (E-plane):     {beamwidths['hpbw_e_plane']:.1f}°")
        print(f"  HPBW (H-plane):     {beamwidths['hpbw_h_plane']:.1f}°")
        print(f"  SLL (E-plane):      {sidelobes['e_plane']['sll_db']:.1f} dB")
        print(f"  SLL (H-plane):      {sidelobes['h_plane']['sll_db']:.1f} dB")
        print(f"  Front/Back ratio:   {fb_ratio:.1f} dB")
        print(f"  Power in ±10° cone: {power_10deg:.1f}%")
        print(f"  Power in ±30° cone: {power_30deg:.1f}%")
        print("=" * 50)

    return result


__all__ = [
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
    # Torch objectives
    "directivity_torch",
    "cone_power_torch",
    "sll_torch",
    "combined_objective_torch",
    "beam_steering_objective_torch",
    "find_peak_direction_torch",
]


# =============================================================================
# PyTorch Differentiable Objectives (for optimization)
# =============================================================================


def directivity_torch(
    power,
    theta_samples: int = 181,
    phi_step_deg: float = 1.0,
):
    """
    Compute directivity (differentiable) from power pattern.

    Args:
        power: Shape (batch, theta_samples, num_phi) or (theta_samples, num_phi)

    Returns:
        directivity_dbi: Shape (batch,) or scalar
    """
    torch = _get_torch()

    is_batched = power.dim() == 3
    if not is_batched:
        power = power.unsqueeze(0)

    batch = power.shape[0]
    device = power.device

    theta_rad = torch.linspace(
        0, math.pi / 2, theta_samples, device=device, dtype=torch.float64
    )
    dtheta = theta_rad[1] - theta_rad[0]
    dphi = math.radians(phi_step_deg)

    sin_theta = torch.sin(theta_rad).unsqueeze(0).unsqueeze(-1)  # (1, theta, 1)

    # Integrate: ∫∫ U(θ,φ) sin(θ) dθ dφ
    radiated_power = (power * sin_theta).sum(dim=(1, 2)) * dtheta * dphi  # (batch,)
    peak_power = power.reshape(batch, -1).max(dim=1).values  # (batch,)

    directivity = 4 * math.pi * peak_power / (radiated_power + 1e-12)
    directivity_dbi = 10 * torch.log10(directivity + 1e-12)

    return directivity_dbi.squeeze(0) if not is_batched else directivity_dbi


def cone_power_torch(
    power,
    half_angle_deg: float = 15.0,
    theta_samples: int = 181,
    phi_step_deg: float = 1.0,
    soft_edge: bool = True,
    edge_sharpness: float = 5.0,
):
    """
    Compute fraction of power in a cone around boresight (differentiable).

    Args:
        power: Shape (batch, theta_samples, num_phi) or (theta_samples, num_phi)
        half_angle_deg: Cone half-angle in degrees
        soft_edge: If True, use differentiable soft sigmoid edge
        edge_sharpness: Higher = sharper edge transition (default 5.0)

    Returns:
        cone_fraction: Shape (batch,) or scalar, in [0, 1]
    """
    torch = _get_torch()

    is_batched = power.dim() == 3
    if not is_batched:
        power = power.unsqueeze(0)

    device = power.device

    theta_rad = torch.linspace(
        0, math.pi / 2, theta_samples, device=device, dtype=torch.float64
    )
    theta_deg = torch.linspace(0, 90, theta_samples, device=device, dtype=torch.float64)
    dtheta = theta_rad[1] - theta_rad[0]
    dphi = math.radians(phi_step_deg)

    sin_theta = torch.sin(theta_rad).unsqueeze(0).unsqueeze(-1)  # (1, theta, 1)

    # Total power
    total_power = (power * sin_theta).sum(dim=(1, 2)) * dtheta * dphi

    # Cone mask - use soft sigmoid for gradient flow
    if soft_edge:
        # Soft sigmoid: 1 at theta=0, transitions to 0 around half_angle_deg
        cone_mask = torch.sigmoid(edge_sharpness * (half_angle_deg - theta_deg))
        cone_mask = cone_mask.unsqueeze(0).unsqueeze(-1)  # (1, theta, 1)
    else:
        # Hard mask (no gradient through boundary)
        cone_mask = (theta_deg <= half_angle_deg).float().unsqueeze(0).unsqueeze(-1)

    cone_power = (power * sin_theta * cone_mask).sum(dim=(1, 2)) * dtheta * dphi

    fraction = cone_power / (total_power + 1e-12)

    return fraction.squeeze(0) if not is_batched else fraction


def sll_torch(
    power,
    theta_samples: int = 181,
    phi_step_deg: float = 1.0,
    main_lobe_width_deg: float = 20.0,
):
    """
    Compute approximate sidelobe level (differentiable).

    Uses ratio of peak power to max power outside main lobe region.

    Args:
        power: Shape (batch, theta_samples, num_phi) or (theta_samples, num_phi)
        main_lobe_width_deg: Theta range to exclude as main lobe

    Returns:
        sll_db: Shape (batch,) or scalar (negative = sidelobes below main)
    """
    torch = _get_torch()

    is_batched = power.dim() == 3
    if not is_batched:
        power = power.unsqueeze(0)

    batch = power.shape[0]
    device = power.device

    theta_deg = torch.linspace(0, 90, theta_samples, device=device, dtype=torch.float64)

    # Main lobe: theta < main_lobe_width_deg
    main_mask = (theta_deg < main_lobe_width_deg).unsqueeze(0).unsqueeze(-1)
    side_mask = ~main_mask

    # Peak in main lobe
    main_power = power * main_mask.float() + (-1e12) * (~main_mask).float()
    peak_main = main_power.reshape(batch, -1).max(dim=1).values

    # Peak in sidelobes
    side_power = power * side_mask.float() + (-1e12) * (~side_mask).float()
    peak_side = side_power.reshape(batch, -1).max(dim=1).values

    # SLL = 10 * log10(peak_side / peak_main) - negative means sidelobes are lower
    sll_db = 10 * torch.log10((peak_side + 1e-12) / (peak_main + 1e-12))

    return sll_db.squeeze(0) if not is_batched else sll_db


def combined_objective_torch(
    power,
    theta_samples: int = 181,
    phi_step_deg: float = 1.0,
    weight_directivity: float = 1.0,
    weight_cone_power: float = 0.0,
    weight_sll: float = 0.0,
    cone_half_angle_deg: float = 15.0,
    main_lobe_width_deg: float = 20.0,
    penalty: float = 0.0,
):
    """
    Combined objective for optimization (to maximize).

    Args:
        power: Shape (batch, theta_samples, num_phi) or (theta_samples, num_phi)
        weight_*: Weights for each objective component
        penalty: Constraint penalty to subtract

    Returns:
        objective: Shape (batch,) or scalar

    Objective = w1*directivity_dbi + w2*cone_power*100 - w3*|sll_db| - penalty
    """
    torch = _get_torch()

    objective = torch.zeros(1, device=power.device, dtype=torch.float64)

    if weight_directivity > 0:
        d = directivity_torch(power, theta_samples, phi_step_deg)
        objective = objective + weight_directivity * d

    if weight_cone_power > 0:
        c = cone_power_torch(power, cone_half_angle_deg, theta_samples, phi_step_deg)
        objective = objective + weight_cone_power * c  # cone power ratio (0-1)

    if weight_sll > 0:
        s = sll_torch(power, theta_samples, phi_step_deg, main_lobe_width_deg)
        # SLL is negative, we want to minimize |SLL| which means maximize SLL (make it less negative)
        # But typically we want lower sidelobes, so we ADD sll (more negative = worse)
        # If you want to minimize sidelobes: subtract weight * |sll| or add weight * sll
        objective = objective + weight_sll * s  # more negative SLL = lower objective

    objective = objective - penalty

    return objective


def beam_steering_objective_torch(
    power,
    target_theta_deg: float,
    target_phi_deg: float,
    theta_samples: int = 181,
    phi_step_deg: float = 1.0,
):
    """
    Compute beam steering objective - maximize power at target direction.

    Args:
        power: Shape (batch, theta_samples, num_phi) or (theta_samples, num_phi)
        target_theta_deg: Target theta angle in degrees (0-90)
        target_phi_deg: Target phi angle in degrees (0-360)
        theta_samples: Number of theta samples
        phi_step_deg: Phi step in degrees

    Returns:
        objective: Power ratio at target direction (higher = better steering)
    """
    torch = _get_torch()

    is_batched = power.dim() == 3
    if not is_batched:
        power = power.unsqueeze(0)

    batch = power.shape[0]
    device = power.device
    num_phi = power.shape[2]

    # Find indices for target direction
    theta_idx = int(round(target_theta_deg / 90.0 * (theta_samples - 1)))
    theta_idx = min(max(theta_idx, 0), theta_samples - 1)

    phi_idx = int(round(target_phi_deg / phi_step_deg)) % num_phi

    # Get power at target direction
    target_power = power[:, theta_idx, phi_idx]  # (batch,)

    # Get peak power for normalization
    peak_power = power.reshape(batch, -1).max(dim=1).values

    # Steering efficiency: ratio of target power to peak power
    # 1.0 means peak is exactly at target, <1.0 means peak is elsewhere
    steering_efficiency = target_power / (peak_power + 1e-12)

    # Also compute gain at target direction (in dBi)
    theta_rad = torch.linspace(
        0, math.pi / 2, theta_samples, device=device, dtype=torch.float64
    )
    dtheta = theta_rad[1] - theta_rad[0]
    dphi = math.radians(phi_step_deg)
    sin_theta = torch.sin(theta_rad).unsqueeze(0).unsqueeze(-1)
    radiated_power = (power * sin_theta).sum(dim=(1, 2)) * dtheta * dphi
    gain_at_target = 4 * math.pi * target_power / (radiated_power + 1e-12)
    gain_dbi = 10 * torch.log10(gain_at_target + 1e-12)

    result = steering_efficiency.squeeze(0) if not is_batched else steering_efficiency
    return result, gain_dbi.squeeze(0) if not is_batched else gain_dbi


def find_peak_direction_torch(
    power,
    theta_samples: int = 181,
    phi_step_deg: float = 1.0,
):
    """
    Find the direction of peak radiation (differentiable-friendly).

    Returns:
        (peak_theta_deg, peak_phi_deg, peak_power)
    """
    torch = _get_torch()

    is_batched = power.dim() == 3
    if not is_batched:
        power = power.unsqueeze(0)

    batch = power.shape[0]
    device = power.device
    num_phi = power.shape[2]

    # Find peak index
    flat_power = power.reshape(batch, -1)
    peak_idx = flat_power.argmax(dim=1)  # (batch,)

    # Convert to theta, phi indices
    theta_idx = peak_idx // num_phi
    phi_idx = peak_idx % num_phi

    # Convert to degrees
    theta_deg = theta_idx.float() / (theta_samples - 1) * 90.0
    phi_deg = phi_idx.float() * phi_step_deg

    peak_power = flat_power.max(dim=1).values

    if not is_batched:
        return theta_deg.squeeze(0), phi_deg.squeeze(0), peak_power.squeeze(0)
    return theta_deg, phi_deg, peak_power
