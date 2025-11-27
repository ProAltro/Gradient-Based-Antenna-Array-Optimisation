"""Hexagonal lattice parameterized array."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch

from .base import ParamArray, ParamArrayConfig


@dataclass
class HexagonalParamArrayConfig(ParamArrayConfig):
    """Config for hexagonal lattice arrays."""

    num_rings: int = 2  # 0 = center only (1 elem), 1 = 7 elem, 2 = 19 elem, etc.
    spacing_bounds_mm: Tuple[float, float] = (4.0, 15.0)
    rotation_bounds: Tuple[float, float] = (0.0, 6.283185307179586)  # 2*pi
    position_offset_bounds_mm: Tuple[float, float] = (
        -5.0,
        5.0,
    )  # Per-element x,y offsets
    optimize_spacing: bool = True
    optimize_rotation: bool = True  # Global rotation of lattice
    optimize_position_offsets: bool = False  # Per-element x,y offsets


class HexagonalParamArray(ParamArray):
    """
    Hexagonal (honeycomb) lattice array.

    Elements are arranged in a hexagonal pattern with configurable
    number of rings around the center element.

    Ring 0: 1 element (center)
    Ring 1: 6 elements (total 7)
    Ring 2: 12 elements (total 19)
    Ring n: 6*n elements

    Total elements for n rings: 1 + 3*n*(n+1)

    Learnable params (depending on config):
    - spacing (1 param) if optimize_spacing
    - rotation (1 param) if optimize_rotation
    - position_offsets (2*N params: x,y per element) if optimize_position_offsets
    - amplitudes (N params) if optimize_amplitudes
    - phases (N params) if optimize_phases
    """

    def __init__(self, config: Optional[HexagonalParamArrayConfig] = None):
        self.hex_config = config or HexagonalParamArrayConfig()
        self._element_positions_normalized = self._compute_hex_positions()
        super().__init__(self.hex_config)

    def _compute_hex_positions(self) -> torch.Tensor:
        """Compute normalized hexagonal positions (spacing=1)."""
        positions = [(0.0, 0.0)]  # Center

        for ring in range(1, self.hex_config.num_rings + 1):
            # 6 directions for hex grid
            directions = [
                (1.0, 0.0),
                (0.5, 0.8660254037844386),  # sqrt(3)/2
                (-0.5, 0.8660254037844386),
                (-1.0, 0.0),
                (-0.5, -0.8660254037844386),
                (0.5, -0.8660254037844386),
            ]

            for i, (dx, dy) in enumerate(directions):
                # Start position for this edge
                x = ring * dx
                y = ring * dy

                # Next direction (for walking along edge)
                next_dir = directions[(i + 2) % 6]

                for step in range(ring):
                    positions.append((x, y))
                    x += next_dir[0]
                    y += next_dir[1]

        return torch.tensor(positions, dtype=torch.float64)

    @property
    def num_elements(self) -> int:
        n = self.hex_config.num_rings
        return 1 + 3 * n * (n + 1) if n > 0 else 1

    @property
    def num_params(self) -> int:
        n = 0
        if self.hex_config.optimize_spacing:
            n += 1
        if self.hex_config.optimize_rotation:
            n += 1
        if self.hex_config.optimize_position_offsets:
            n += 2 * self.num_elements  # x and y offset per element
        if self.config.optimize_amplitudes:
            n += self.num_elements
        if self.config.optimize_phases:
            n += self.num_elements
        return n

    @property
    def param_names(self) -> list[str]:
        names = []
        if self.hex_config.optimize_spacing:
            names.append("spacing")
        if self.hex_config.optimize_rotation:
            names.append("rotation")
        if self.hex_config.optimize_position_offsets:
            for i in range(self.num_elements):
                names.extend([f"offset_x_{i}", f"offset_y_{i}"])
        if self.config.optimize_amplitudes:
            names.extend([f"amp_{i}" for i in range(self.num_elements)])
        if self.config.optimize_phases:
            names.extend([f"phase_{i}" for i in range(self.num_elements)])
        return names

    def _define_bounds(self) -> torch.Tensor:
        bounds = []
        if self.hex_config.optimize_spacing:
            bounds.append(list(self.hex_config.spacing_bounds_mm))
        if self.hex_config.optimize_rotation:
            bounds.append(list(self.hex_config.rotation_bounds))
        if self.hex_config.optimize_position_offsets:
            # x,y offset for each element
            for _ in range(self.num_elements):
                bounds.append(list(self.hex_config.position_offset_bounds_mm))  # x
                bounds.append(list(self.hex_config.position_offset_bounds_mm))  # y
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

        # Spacing
        if self.hex_config.optimize_spacing:
            spacing_norm = alpha[:, idx]
            spacing_min, spacing_max = self.hex_config.spacing_bounds_mm
            spacing = spacing_min + spacing_norm * (spacing_max - spacing_min)
            idx += 1
        else:
            mid = (
                self.hex_config.spacing_bounds_mm[0]
                + self.hex_config.spacing_bounds_mm[1]
            ) / 2
            spacing = torch.full((batch,), mid, dtype=torch.float64, device=device)

        # Rotation
        if self.hex_config.optimize_rotation:
            rotation_norm = alpha[:, idx]
            rotation = rotation_norm * 2 * torch.pi
            idx += 1
        else:
            rotation = torch.zeros(batch, dtype=torch.float64, device=device)

        # Build base positions
        base_pos = self._element_positions_normalized.to(device)  # (N, 2)

        # Apply rotation
        cos_r = torch.cos(rotation)  # (batch,)
        sin_r = torch.sin(rotation)

        # Rotation matrix applied to each position
        x = base_pos[:, 0].unsqueeze(0)  # (1, N)
        y = base_pos[:, 1].unsqueeze(0)

        x_rot = cos_r.unsqueeze(1) * x - sin_r.unsqueeze(1) * y  # (batch, N)
        y_rot = sin_r.unsqueeze(1) * x + cos_r.unsqueeze(1) * y

        # Scale by spacing
        x_scaled = spacing.unsqueeze(1) * x_rot
        y_scaled = spacing.unsqueeze(1) * y_rot

        # Apply per-element position offsets
        if self.hex_config.optimize_position_offsets:
            offset_min, offset_max = self.hex_config.position_offset_bounds_mm
            # Extract x,y offsets for all elements: (batch, 2*N)
            offsets_norm = alpha[:, idx : idx + 2 * self.num_elements]
            offsets = offset_min + offsets_norm * (offset_max - offset_min)
            # Reshape to (batch, N, 2)
            offsets = offsets.reshape(batch, self.num_elements, 2)
            x_scaled = x_scaled + offsets[:, :, 0]
            y_scaled = y_scaled + offsets[:, :, 1]
            idx += 2 * self.num_elements

        positions = torch.stack([x_scaled, y_scaled], dim=-1)  # (batch, N, 2)
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
        """Default: mid-spacing, zero rotation, zero offsets."""
        alpha = torch.zeros(self.num_params, dtype=torch.float64, device=device)
        idx = 0
        if self.hex_config.optimize_spacing:
            alpha[idx] = 0.5
            idx += 1
        if self.hex_config.optimize_rotation:
            alpha[idx] = 0.0
            idx += 1
        if self.hex_config.optimize_position_offsets:
            # Default: zero offsets (center of range = 0.5 in normalized space)
            alpha[idx : idx + 2 * self.num_elements] = 0.5
            idx += 2 * self.num_elements
        if self.config.optimize_amplitudes:
            alpha[idx : idx + self.num_elements] = 1.0
            idx += self.num_elements
        if self.config.optimize_phases:
            alpha[idx : idx + self.num_elements] = 0.0
        return alpha
