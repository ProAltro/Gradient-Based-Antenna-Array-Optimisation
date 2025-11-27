"""Circular/Concentric Ring parameterized array."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import torch

from .base import ParamArray, ParamArrayConfig


@dataclass
class CircularParamArrayConfig(ParamArrayConfig):
    """Config for circular/concentric ring arrays."""

    elements_per_ring: Tuple[int, ...] = (1, 6, 12)  # Center + rings
    radius_bounds_mm: Tuple[float, float] = (5.0, 50.0)
    optimize_radii: bool = True
    optimize_rotations: bool = True  # Rotation offset per ring


class CircularParamArray(ParamArray):
    """
    Concentric ring array with optional variable radii and rotations.

    Elements are placed uniformly on concentric rings. The first "ring"
    can have 1 element (center) or more.

    Learnable params (depending on config):
    - radii (num_rings params) if optimize_radii
    - rotations (num_rings params) if optimize_rotations
    - amplitudes (N params) if optimize_amplitudes
    - phases (N params) if optimize_phases
    """

    def __init__(self, config: Optional[CircularParamArrayConfig] = None):
        self.circular_config = config or CircularParamArrayConfig()
        super().__init__(self.circular_config)

    @property
    def num_rings(self) -> int:
        return len(self.circular_config.elements_per_ring)

    @property
    def num_elements(self) -> int:
        return sum(self.circular_config.elements_per_ring)

    @property
    def num_params(self) -> int:
        n = 0
        if self.circular_config.optimize_radii:
            n += self.num_rings
        if self.circular_config.optimize_rotations:
            n += self.num_rings
        if self.config.optimize_amplitudes:
            n += self.num_elements
        if self.config.optimize_phases:
            n += self.num_elements
        return n

    @property
    def param_names(self) -> list[str]:
        names = []
        if self.circular_config.optimize_radii:
            names.extend([f"radius_{i}" for i in range(self.num_rings)])
        if self.circular_config.optimize_rotations:
            names.extend([f"rotation_{i}" for i in range(self.num_rings)])
        if self.config.optimize_amplitudes:
            names.extend([f"amp_{i}" for i in range(self.num_elements)])
        if self.config.optimize_phases:
            names.extend([f"phase_{i}" for i in range(self.num_elements)])
        return names

    def _define_bounds(self) -> torch.Tensor:
        bounds = []
        if self.circular_config.optimize_radii:
            for i in range(self.num_rings):
                bounds.append(list(self.circular_config.radius_bounds_mm))
        if self.circular_config.optimize_rotations:
            bounds.extend([[0.0, 2 * 3.141592653589793]] * self.num_rings)
        if self.config.optimize_amplitudes:
            bounds.extend([[0.0, 1.0]] * self.num_elements)
        if self.config.optimize_phases:
            bounds.extend([[0.0, 2 * 3.141592653589793]] * self.num_elements)
        return torch.tensor(bounds, dtype=torch.float64)

    def decode(self, alpha: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Decode normalized params to physical values."""
        is_batched = alpha.dim() == 2
        if not is_batched:
            alpha = alpha.unsqueeze(0)

        batch = alpha.shape[0]
        device = alpha.device
        bounds = self.bounds.to(device)

        idx = 0
        result = {}

        # Radii
        if self.circular_config.optimize_radii:
            radii_norm = alpha[:, idx : idx + self.num_rings]
            radii_min = bounds[idx : idx + self.num_rings, 0]
            radii_range = bounds[idx : idx + self.num_rings, 1] - radii_min
            radii = radii_min + radii_norm * radii_range  # (batch, num_rings)
            idx += self.num_rings
        else:
            # Default: linearly spaced radii
            r_min, r_max = self.circular_config.radius_bounds_mm
            radii = torch.linspace(
                r_min, r_max, self.num_rings, dtype=torch.float64, device=device
            )
            radii = radii.unsqueeze(0).expand(batch, -1)

        # Rotations
        if self.circular_config.optimize_rotations:
            rotations_norm = alpha[:, idx : idx + self.num_rings]
            rotations = rotations_norm * 2 * torch.pi  # (batch, num_rings)
            idx += self.num_rings
        else:
            rotations = torch.zeros(
                batch, self.num_rings, dtype=torch.float64, device=device
            )

        # Build positions
        positions_list = []
        for ring_idx, n_elem in enumerate(self.circular_config.elements_per_ring):
            r = radii[:, ring_idx]  # (batch,)
            rot = rotations[:, ring_idx]  # (batch,)

            if n_elem == 1:
                # Center element (or single element ring)
                if ring_idx == 0 and self.circular_config.elements_per_ring[0] == 1:
                    # True center
                    pos = torch.zeros(batch, 1, 2, dtype=torch.float64, device=device)
                else:
                    # Single element at radius
                    x = r * torch.cos(rot)
                    y = r * torch.sin(rot)
                    pos = torch.stack([x, y], dim=-1).unsqueeze(1)
            else:
                # Multiple elements uniformly on ring
                angles_base = torch.linspace(
                    0, 2 * torch.pi, n_elem + 1, dtype=torch.float64, device=device
                )[:-1]
                angles = angles_base.unsqueeze(0) + rot.unsqueeze(1)  # (batch, n_elem)
                x = r.unsqueeze(1) * torch.cos(angles)  # (batch, n_elem)
                y = r.unsqueeze(1) * torch.sin(angles)
                pos = torch.stack([x, y], dim=-1)  # (batch, n_elem, 2)

            positions_list.append(pos)

        positions = torch.cat(positions_list, dim=1)  # (batch, N, 2)
        result["positions"] = positions

        # Amplitudes
        if self.config.optimize_amplitudes:
            amp_norm = alpha[:, idx : idx + self.num_elements]
            result["amplitudes"] = amp_norm
            idx += self.num_elements

        # Phases
        if self.config.optimize_phases:
            phase_norm = alpha[:, idx : idx + self.num_elements]
            result["phases"] = phase_norm * 2 * torch.pi
            idx += self.num_elements

        if not is_batched:
            result = {k: v.squeeze(0) for k, v in result.items()}

        return result

    def default_init(self, device: str = "cpu") -> torch.Tensor:
        """Default: linearly spaced radii, zero rotations."""
        alpha = torch.zeros(self.num_params, dtype=torch.float64, device=device)
        idx = 0
        if self.circular_config.optimize_radii:
            # Linearly spaced normalized values
            alpha[idx : idx + self.num_rings] = torch.linspace(0.1, 0.9, self.num_rings)
            idx += self.num_rings
        if self.circular_config.optimize_rotations:
            alpha[idx : idx + self.num_rings] = 0.0
            idx += self.num_rings
        if self.config.optimize_amplitudes:
            alpha[idx : idx + self.num_elements] = 1.0
            idx += self.num_elements
        if self.config.optimize_phases:
            alpha[idx : idx + self.num_elements] = 0.0
        return alpha
