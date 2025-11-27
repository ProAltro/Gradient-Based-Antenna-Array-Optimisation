"""
Array far-field evaluation with PyTorch (differentiable) or NumPy/CuPy backends.
Use backend='torch' for optimization, backend='numpy' for fast evaluation.
"""

from __future__ import annotations

import math
import importlib.util
from typing import Optional, Tuple, Union, TYPE_CHECKING, Any
import numpy as np

if TYPE_CHECKING:
    import torch as torch_type

    TorchTensor = torch_type.Tensor
else:
    TorchTensor = Any

C0 = 299_792_458.0

_torch_available = importlib.util.find_spec("torch") is not None
_cupy_available = importlib.util.find_spec("cupy") is not None
HAS_TORCH = _torch_available
HAS_CUDA = _cupy_available

_torch = None
_cp = None


def _get_torch():
    global _torch
    if _torch is None:
        import torch

        _torch = torch
    return _torch


def _get_cupy():
    global _cp
    if _cp is None:
        import cupy

        _cp = cupy
    return _cp


def _get_xp(use_cuda: bool = False):
    if use_cuda and _cupy_available:
        return _get_cupy(), True
    return np, False


# =============================================================================
# Element Patterns - NumPy/CuPy
# =============================================================================


def rectangular_patch_pattern_xp(xp, theta, phi, k0, L, W):
    """Rectangular patch (TM10) pattern."""
    sin_theta = xp.sin(theta) + 1e-12
    psi_L = (k0 * L / 2.0) * sin_theta * xp.cos(phi)
    psi_W = (k0 * W / 2.0) * sin_theta * xp.sin(phi)
    psi_W = xp.where(xp.abs(psi_W) < 1e-12, 1e-12, psi_W)
    field = xp.cos(psi_L) * (xp.sin(psi_W) / psi_W) * xp.cos(theta)
    return xp.abs(field)


def circular_patch_pattern_xp(xp, theta, phi, k0, a):
    """Circular patch (TM11) pattern."""
    sin_theta = xp.sin(theta)
    x = k0 * a * sin_theta
    x = xp.where(xp.abs(x) < 1e-12, 1e-12, x)
    if xp is np:
        from scipy.special import j1

        j1_val = j1(x)
    elif hasattr(xp, "special") and hasattr(xp.special, "j1"):
        j1_val = xp.special.j1(x)
    else:
        j1_val = xp.sin(x) / x
    field = (j1_val / x) * xp.cos(theta)
    return xp.abs(field)


# =============================================================================
# Element Patterns - PyTorch
# =============================================================================


def rectangular_patch_pattern_torch(theta, phi, k0, L, W):
    """Rectangular patch (TM10) pattern - differentiable."""
    torch = _get_torch()
    sin_theta = torch.sin(theta) + 1e-12
    psi_L = (k0 * L / 2.0) * sin_theta * torch.cos(phi)
    psi_W = (k0 * W / 2.0) * sin_theta * torch.sin(phi)
    # Safe division: avoid division by zero by clamping small values
    psi_W_safe = torch.where(
        torch.abs(psi_W) < 1e-12,
        torch.ones_like(psi_W) * 1e-12,  # Use small positive value
        psi_W,
    )
    sinc_W = torch.sin(psi_W_safe) / psi_W_safe
    field = torch.cos(psi_L) * sinc_W * torch.cos(theta)
    return torch.abs(field)


def circular_patch_pattern_torch(theta, phi, k0, a):
    """Circular patch (TM11) pattern - differentiable."""
    torch = _get_torch()
    sin_theta = torch.sin(theta)
    x = k0 * a * sin_theta
    # Safe division: use where instead of sign*clamp
    x_safe = torch.where(torch.abs(x) < 1e-12, torch.ones_like(x) * 1e-12, x)
    if hasattr(torch.special, "bessel_j1"):
        j1_val = torch.special.bessel_j1(x_safe)
    else:
        j1_val = torch.where(
            torch.abs(x_safe) < 0.5, x_safe / 2.0, torch.sin(x_safe) / x_safe
        )
    field = (j1_val / x_safe) * torch.cos(theta)
    return torch.abs(field)


# =============================================================================
# NumPy/CuPy Evaluation
# =============================================================================


def evaluate_array_xp(
    positions: np.ndarray,
    weights: Optional[np.ndarray] = None,
    *,
    freq_hz: float = 24.0e9,
    element_type: str = "isotropic",
    element_dims: Tuple[float, ...] = (0.0, 0.0),
    phi_step_deg: float = 1.0,
    theta_samples: int = 181,
    position_unit: str = "mm",
    use_cuda: bool = False,
) -> np.ndarray:
    """Evaluate far-field power using NumPy or CuPy."""
    xp, using_cuda = _get_xp(use_cuda)

    positions = np.asarray(positions)
    if positions.ndim == 2:
        positions = positions[np.newaxis, ...]

    scale = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}[position_unit.lower()]
    pos = xp.asarray(positions) * scale
    batch, num_elements = pos.shape[0], pos.shape[1]

    if weights is None:
        w = xp.ones((batch, num_elements), dtype=xp.complex128)
    else:
        w = xp.asarray(weights, dtype=xp.complex128)
        if w.ndim == 1:
            w = w[xp.newaxis, :]

    k0 = 2.0 * math.pi * freq_hz / C0
    theta_1d = xp.linspace(0.0, math.pi / 2, theta_samples)
    phi_1d = xp.deg2rad(xp.arange(0.0, 360.0, phi_step_deg))
    theta, phi = xp.meshgrid(theta_1d, phi_1d, indexing="ij")

    u = (xp.sin(theta) * xp.cos(phi)).flatten()
    v = (xp.sin(theta) * xp.sin(phi)).flatten()
    k_dir = xp.stack([u, v], axis=0)

    phase = k0 * xp.matmul(pos, k_dir)
    af = xp.sum(w[:, :, None] * xp.exp(1j * phase), axis=1)

    if element_type == "rect_patch":
        L, W = element_dims[0] * scale, element_dims[1] * scale
        ep = rectangular_patch_pattern_xp(xp, theta.flatten(), phi.flatten(), k0, L, W)
    elif element_type == "circ_patch":
        a = element_dims[0] * scale
        ep = circular_patch_pattern_xp(xp, theta.flatten(), phi.flatten(), k0, a)
    else:
        ep = xp.ones_like(u)

    total_field = af * ep[None, :]
    power = xp.abs(total_field) ** 2
    power_grid = power.reshape(batch, theta.shape[0], theta.shape[1])

    return _get_cupy().asnumpy(power_grid) if using_cuda else power_grid


# =============================================================================
# PyTorch Evaluation (Differentiable)
# =============================================================================


def evaluate_array_torch(
    positions,
    weights=None,
    *,
    freq_hz: float = 24.0e9,
    element_type: str = "isotropic",
    element_dims: Union[Tuple[float, ...], TorchTensor, None] = None,
    phi_step_deg: float = 1.0,
    theta_samples: int = 181,
    position_unit: str = "mm",
    device: Optional[str] = None,
):
    """Evaluate far-field power using PyTorch (differentiable).

    Parameters
    ----------
    positions : torch.Tensor
        Element positions
    weights : torch.Tensor, optional
        Complex element weights
    freq_hz : float
        Frequency in Hz
    element_type : str
        'isotropic', 'rect_patch', or 'circ_patch'
    element_dims : tuple of floats, torch.Tensor, or None
        Element dimensions. Can be:
        - Tuple of floats: (length, width) for rect or (radius,) for circ
        - torch.Tensor: shape (2,) for rect or (1,) for circ - MAINTAINS GRADIENTS
        - None: defaults to (0.0, 0.0)
    phi_step_deg : float
        Azimuthal step in degrees
    theta_samples : int
        Number of elevation samples
    position_unit : str
        'mm', 'cm', or 'm'
    device : str, optional
        PyTorch device
    """
    torch = _get_torch()

    if device is None:
        device = positions.device if hasattr(positions, "device") else "cpu"

    if positions.dim() == 2:
        positions = positions.unsqueeze(0)

    batch, num_elements, _ = positions.shape
    scale = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}[position_unit.lower()]
    pos = positions * scale

    if weights is None:
        w = torch.ones(batch, num_elements, dtype=torch.complex128, device=device)
    else:
        w = weights.to(dtype=torch.complex128, device=device)
        if w.dim() == 1:
            w = w.unsqueeze(0).expand(batch, -1)

    k0 = 2.0 * math.pi * freq_hz / C0

    theta_1d = torch.linspace(
        0.0, math.pi / 2, theta_samples, device=device, dtype=torch.float64
    )
    phi_1d = torch.deg2rad(
        torch.arange(0.0, 360.0, phi_step_deg, device=device, dtype=torch.float64)
    )
    theta, phi = torch.meshgrid(theta_1d, phi_1d, indexing="ij")

    u = (torch.sin(theta) * torch.cos(phi)).flatten()
    v = (torch.sin(theta) * torch.sin(phi)).flatten()
    k_dir = torch.stack([u, v], dim=0)

    phase = k0 * torch.matmul(pos, k_dir)
    af = torch.sum(w[:, :, None] * torch.exp(1j * phase), dim=1)

    theta_flat = theta.flatten()
    phi_flat = phi.flatten()

    # Handle element pattern with tensor or tuple dimensions
    if element_type == "rect_patch":
        if element_dims is None:
            element_dims = (0.0, 0.0)

        if isinstance(element_dims, torch.Tensor):
            # Tensor dimensions - maintains gradient flow!
            L = element_dims[0] * scale
            W = element_dims[1] * scale
        else:
            # Tuple of floats - legacy behavior
            L = element_dims[0] * scale
            W = element_dims[1] * scale

        ep = rectangular_patch_pattern_torch(theta_flat, phi_flat, k0, L, W)
    elif element_type == "circ_patch":
        if element_dims is None:
            element_dims = (0.0,)

        if isinstance(element_dims, torch.Tensor):
            # Tensor dimensions - maintains gradient flow!
            a = element_dims[0] * scale
        else:
            # Tuple of floats - legacy behavior
            a = element_dims[0] * scale

        ep = circular_patch_pattern_torch(theta_flat, phi_flat, k0, a)
    else:
        ep = torch.ones_like(u)

    total_field = af * ep[None, :]
    power = torch.abs(total_field) ** 2
    return power.reshape(batch, theta_samples, len(phi_1d))


# =============================================================================
# Unified Interface
# =============================================================================


def evaluate_array(
    positions,
    weights=None,
    *,
    freq_hz: float = 24.0e9,
    element_type: str = "isotropic",
    element_dims: Tuple[float, ...] = (0.0, 0.0),
    phi_step_deg: float = 1.0,
    theta_samples: int = 181,
    position_unit: str = "mm",
    backend: str = "numpy",
    use_cuda: bool = False,
    device: Optional[str] = None,
):
    """
    Evaluate far-field power for 2D planar antenna arrays.

    backend='torch' for differentiable optimization, 'numpy' for fast evaluation.
    Returns shape (batch, theta_samples, num_phi).
    """
    if backend == "torch":
        torch = _get_torch()
        if not isinstance(positions, torch.Tensor):
            positions = torch.tensor(positions, dtype=torch.float64)
        if weights is not None and not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.complex128)
        return evaluate_array_torch(
            positions,
            weights,
            freq_hz=freq_hz,
            element_type=element_type,
            element_dims=element_dims,
            phi_step_deg=phi_step_deg,
            theta_samples=theta_samples,
            position_unit=position_unit,
            device=device,
        )
    else:
        return evaluate_array_xp(
            positions,
            weights,
            freq_hz=freq_hz,
            element_type=element_type,
            element_dims=element_dims,
            phi_step_deg=phi_step_deg,
            theta_samples=theta_samples,
            position_unit=position_unit,
            use_cuda=use_cuda,
        )


__all__ = [
    "HAS_TORCH",
    "HAS_CUDA",
    "C0",
    "evaluate_array",
    "evaluate_array_torch",
    "evaluate_array_xp",
    "rectangular_patch_pattern_torch",
    "rectangular_patch_pattern_xp",
    "circular_patch_pattern_torch",
    "circular_patch_pattern_xp",
]
