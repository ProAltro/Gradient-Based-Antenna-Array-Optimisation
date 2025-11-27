"""Patch antenna design calculations for feed positioning and dimensions."""

import math
from typing import Tuple

from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import j1

C0 = 299_792_458.0


# =============================================================================
# Resistance Calculations (for feed position matching)
# =============================================================================


def calculate_circular_patch_resistance(
    radius: float,
    substrate_height: float,
    dielectric_constant: float,
    loss_tangent: float = 0.0009,
    copper_conductivity: float = 5.8e7,
) -> dict:
    """Estimate edge resistance for a circular patch using cavity model."""
    mu_0 = 4 * math.pi * 1e-7
    epsilon_0 = 8.854187817e-12

    log_term = math.log((math.pi * radius) / (2 * substrate_height)) + 1.7726
    fringe_factor = (
        1 + (2 * substrate_height / (math.pi * radius * dielectric_constant)) * log_term
    )
    effective_radius = radius * math.sqrt(fringe_factor)

    fr = (1.8412 * C0) / (
        2 * math.pi * effective_radius * math.sqrt(dielectric_constant)
    )
    omega_r = 2 * math.pi * fr
    k0 = (2 * math.pi * fr) / C0

    def integrand(theta: float) -> float:
        argument = k0 * radius * math.sin(theta)
        return (j1(argument) ** 2) * (math.sin(theta) ** 3)

    integral_i, _ = quad(integrand, 0, math.pi)
    g_r = integral_i / (120 * math.pi)

    r_s = math.sqrt((omega_r * mu_0) / (2 * copper_conductivity))
    eta = 120 * math.pi / math.sqrt(dielectric_constant)
    g_c = (math.pi * r_s) / (substrate_height * eta**2)

    capacitance = (
        epsilon_0 * dielectric_constant * math.pi * radius**2
    ) / substrate_height
    g_d = loss_tangent * omega_r * capacitance

    g_total = g_r + g_c + g_d
    r_edge = 1 / g_total

    return {
        "resistance": r_edge,
        "resonant_frequency": fr,
        "effective_radius": effective_radius,
        "k0": k0,
    }


def calculate_rectangular_patch_resistance(
    length: float,
    width: float,
    substrate_height: float,
    dielectric_constant: float,
    freq_hz: float,
) -> dict:
    """Estimate edge resistance for a rectangular patch using conductance model."""
    lambda_0 = C0 / freq_hz
    k0 = 2 * math.pi / lambda_0

    def integrand(theta: float) -> float:
        kw = k0 * width * math.cos(theta) / 2
        sinc_term = math.sin(kw) / kw if abs(kw) > 1e-10 else 1.0
        return sinc_term**2 * math.sin(theta) ** 3

    integral, _ = quad(integrand, 0, math.pi)
    g1 = integral / (120 * math.pi**2)

    r_edge = 1 / (2 * g1)

    return {
        "resistance": r_edge,
        "k0": k0,
        "lambda_0": lambda_0,
    }


# =============================================================================
# Feed Position Calculations
# =============================================================================


def calculate_circular_feed_position(
    radius: float,
    effective_radius: float,
    k0: float,
    r_edge: float,
    target_impedance: float = 50.0,
) -> float:
    """Calculate radial feed position for impedance matching using Bessel model."""
    j1_edge = j1(k0 * effective_radius)
    if abs(j1_edge) < 1e-10:
        j1_edge = 0.582

    target_ratio = math.sqrt(target_impedance / r_edge)

    def objective(rho: float) -> float:
        return abs(j1(k0 * rho) / j1_edge) - target_ratio

    try:
        rho_optimal = brentq(objective, 1e-6, radius * 0.95)
    except ValueError:
        rho_optimal = radius * 0.35
        print(f"Warning: Could not find exact match, using default ratio 0.35")

    return rho_optimal


def calculate_rectangular_feed_position(
    length: float,
    r_edge: float,
    target_impedance: float = 50.0,
    use_inset_model: bool = False,
) -> float:
    """Calculate feed position (from edge) using cos² (probe) or cos⁴ (inset) model."""
    ratio = target_impedance / r_edge

    if ratio > 1.0:
        print(
            f"Warning: Target impedance ({target_impedance}Ω) > edge resistance ({r_edge:.1f}Ω)"
        )
        return 0.0

    root = ratio**0.25 if use_inset_model else math.sqrt(ratio)
    y0 = (length / math.pi) * math.acos(root)
    return y0


# =============================================================================
# Patch Dimension Calculations
# =============================================================================


def calculate_rectangular_patch_dimensions(
    freq_hz: float,
    dielectric_constant: float,
    substrate_height: float,
) -> Tuple[float, float]:
    """Calculate rectangular patch (length, width) in meters for given frequency."""
    lambda_0 = C0 / freq_hz
    w = lambda_0 / (2 * math.sqrt((dielectric_constant + 1) / 2))

    eps_eff = (dielectric_constant + 1) / 2 + (dielectric_constant - 1) / 2 * (
        1 / math.sqrt(1 + 12 * substrate_height / w)
    )

    delta_l = (
        0.412
        * substrate_height
        * ((eps_eff + 0.3) * (w / substrate_height + 0.264))
        / ((eps_eff - 0.258) * (w / substrate_height + 0.8))
    )

    lambda_g = lambda_0 / math.sqrt(eps_eff)
    length = lambda_g / 2 - 2 * delta_l

    return length, w


def calculate_circular_patch_radius(
    freq_hz: float,
    dielectric_constant: float,
    substrate_height: float,
) -> float:
    """Calculate circular patch radius in meters for TM₁₁ mode with fringing correction."""
    a = (1.8412 * C0) / (2 * math.pi * freq_hz * math.sqrt(dielectric_constant))

    for _ in range(5):
        log_term = math.log((math.pi * a) / (2 * substrate_height)) + 1.7726
        fringe_factor = (
            1 + (2 * substrate_height / (math.pi * a * dielectric_constant)) * log_term
        )

        a_new = (1.8412 * C0) / (
            2 * math.pi * freq_hz * math.sqrt(dielectric_constant * fringe_factor)
        )

        if abs(a_new - a) < 1e-9:
            break
        a = a_new

    return a


# =============================================================================
# High-Level Feed Position Helpers
# =============================================================================


def get_feed_position_circular(
    radius_mm: float,
    substrate_height_mm: float,
    dielectric_constant: float,
    target_impedance: float = 50.0,
    loss_tangent: float = 0.0009,
) -> Tuple[float, dict]:
    """Get feed position (mm from center) and info dict for a circular patch."""
    radius_m = radius_mm * 1e-3
    height_m = substrate_height_mm * 1e-3

    patch_info = calculate_circular_patch_resistance(
        radius=radius_m,
        substrate_height=height_m,
        dielectric_constant=dielectric_constant,
        loss_tangent=loss_tangent,
    )

    feed_pos_m = calculate_circular_feed_position(
        radius=radius_m,
        effective_radius=patch_info["effective_radius"],
        k0=patch_info["k0"],
        r_edge=patch_info["resistance"],
        target_impedance=target_impedance,
    )

    return feed_pos_m * 1e3, patch_info


def get_feed_position_rectangular(
    length_mm: float,
    width_mm: float,
    substrate_height_mm: float,
    dielectric_constant: float,
    freq_hz: float,
    target_impedance: float = 50.0,
    use_inset_model: bool = False,
) -> Tuple[float, dict]:
    """Get feed position (mm from edge) and info dict for a rectangular patch."""
    length_m = length_mm * 1e-3
    width_m = width_mm * 1e-3
    height_m = substrate_height_mm * 1e-3

    patch_info = calculate_rectangular_patch_resistance(
        length=length_m,
        width=width_m,
        substrate_height=height_m,
        dielectric_constant=dielectric_constant,
        freq_hz=freq_hz,
    )

    feed_pos_m = calculate_rectangular_feed_position(
        length=length_m,
        r_edge=patch_info["resistance"],
        target_impedance=target_impedance,
        use_inset_model=use_inset_model,
    )

    return feed_pos_m * 1e3, patch_info
