"""Rectangular grid parameterized array."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch

from .base import ParamArray, ParamArrayConfig


@dataclass
class GridParamArrayConfig(ParamArrayConfig):
    """Config for grid arrays."""

    rows: int = 4
    cols: int = 4
    spacing_bounds_mm: Tuple[float, float] = (4.0, 12.0)
    optimize_spacing: bool = True  # If False, use fixed spacing


class GridParamArray(ParamArray):
    """
    Rectangular grid array with optional variable spacing.

    Learnable params (depending on config):
    - spacing_x, spacing_y (2 params) if optimize_spacing
    - amplitudes (N params) if optimize_amplitudes
    - phases (N params) if optimize_phases
    """

    def __init__(self, config: Optional[GridParamArrayConfig] = None):
        self.grid_config = config or GridParamArrayConfig()
        super().__init__(self.grid_config)

    @property
    def num_elements(self) -> int:
        return self.grid_config.rows * self.grid_config.cols

    @property
    def num_params(self) -> int:
        n = 0
        if self.grid_config.optimize_spacing:
            n += 2  # spacing_x, spacing_y
        if self.config.optimize_amplitudes:
            n += self.num_elements
        if self.config.optimize_phases:
            n += self.num_elements
        return n

    @property
    def param_names(self) -> list[str]:
        names = []
        if self.grid_config.optimize_spacing:
            names.extend(["spacing_x", "spacing_y"])
        if self.config.optimize_amplitudes:
            names.extend([f"amp_{i}" for i in range(self.num_elements)])
        if self.config.optimize_phases:
            names.extend([f"phase_{i}" for i in range(self.num_elements)])
        return names

    def _define_bounds(self) -> torch.Tensor:
        bounds = []
        if self.grid_config.optimize_spacing:
            bounds.append(list(self.grid_config.spacing_bounds_mm))  # x
            bounds.append(list(self.grid_config.spacing_bounds_mm))  # y
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
        if self.grid_config.optimize_spacing:
            spacing_norm = alpha[:, idx : idx + 2]
            spacing_min = bounds[idx : idx + 2, 0]
            spacing_range = bounds[idx : idx + 2, 1] - bounds[idx : idx + 2, 0]
            spacing = spacing_min + spacing_norm * spacing_range
            spacing_x = spacing[:, 0]
            spacing_y = spacing[:, 1]
            idx += 2
        else:
            mid = (
                self.grid_config.spacing_bounds_mm[0]
                + self.grid_config.spacing_bounds_mm[1]
            ) / 2
            spacing_x = torch.full((batch,), mid, dtype=torch.float64, device=device)
            spacing_y = spacing_x.clone()

        # Build positions
        rows, cols = self.grid_config.rows, self.grid_config.cols
        ix = torch.arange(cols, dtype=torch.float64, device=device)
        iy = torch.arange(rows, dtype=torch.float64, device=device)
        gx, gy = torch.meshgrid(ix, iy, indexing="ij")
        gx, gy = gx.flatten(), gy.flatten()  # (N,)

        # Center the grid
        gx = gx - gx.mean()
        gy = gy - gy.mean()

        # Scale by spacing (batch, N)
        pos_x = spacing_x[:, None] * gx[None, :]
        pos_y = spacing_y[:, None] * gy[None, :]
        positions = torch.stack([pos_x, pos_y], dim=-1)  # (batch, N, 2)
        result["positions"] = positions

        # Amplitudes
        if self.config.optimize_amplitudes:
            amp_norm = alpha[:, idx : idx + self.num_elements]
            amp_min, amp_max = bounds[idx, 0], bounds[idx, 1]
            result["amplitudes"] = amp_min + amp_norm * (amp_max - amp_min)
            idx += self.num_elements

        # Phases
        if self.config.optimize_phases:
            phase_norm = alpha[:, idx : idx + self.num_elements]
            phase_min, phase_max = bounds[idx, 0], bounds[idx, 1]
            result["phases"] = phase_min + phase_norm * (phase_max - phase_min)
            idx += self.num_elements

        if not is_batched:
            result = {k: v.squeeze(0) for k, v in result.items()}

        return result

    def default_init(self, device: str = "cpu") -> torch.Tensor:
        """Default: mid-spacing, uniform amplitudes, zero phases."""
        alpha = torch.zeros(self.num_params, dtype=torch.float64, device=device)
        idx = 0
        if self.grid_config.optimize_spacing:
            alpha[idx : idx + 2] = 0.5  # mid spacing
            idx += 2
        if self.config.optimize_amplitudes:
            alpha[idx : idx + self.num_elements] = 1.0  # max amplitude
            idx += self.num_elements
        if self.config.optimize_phases:
            alpha[idx : idx + self.num_elements] = 0.0  # zero phase
        return alpha
