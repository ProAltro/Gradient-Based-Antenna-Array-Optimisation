"""Linear parameterized array."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch

from .base import ParamArray, ParamArrayConfig


@dataclass
class LinearParamArrayConfig(ParamArrayConfig):
    """Config for linear arrays."""

    n_elements: int = 8
    spacing_bounds_mm: Tuple[float, float] = (3.0, 15.0)
    orientation: str = "x"  # 'x' or 'y' axis
    spacing_mode: str = "uniform"  # 'uniform' or 'non_uniform'


class LinearParamArray(ParamArray):
    """
    Linear (1D) array along x or y axis.

    Supports uniform spacing (single parameter) or non-uniform spacing
    (N-1 parameters for gaps between elements).

    Learnable params (depending on config):
    - spacing (1 param) if uniform mode
    - spacings (N-1 params) if non_uniform mode
    - amplitudes (N params) if optimize_amplitudes
    - phases (N params) if optimize_phases
    """

    def __init__(self, config: Optional[LinearParamArrayConfig] = None):
        self.linear_config = config or LinearParamArrayConfig()
        super().__init__(self.linear_config)

    @property
    def num_elements(self) -> int:
        return self.linear_config.n_elements

    @property
    def num_spacing_params(self) -> int:
        if self.linear_config.spacing_mode == "uniform":
            return 1
        else:  # non_uniform
            return self.num_elements - 1

    @property
    def num_params(self) -> int:
        n = self.num_spacing_params
        if self.config.optimize_amplitudes:
            n += self.num_elements
        if self.config.optimize_phases:
            n += self.num_elements
        return n

    @property
    def param_names(self) -> list[str]:
        names = []
        if self.linear_config.spacing_mode == "uniform":
            names.append("spacing")
        else:
            names.extend([f"spacing_{i}" for i in range(self.num_elements - 1)])
        if self.config.optimize_amplitudes:
            names.extend([f"amp_{i}" for i in range(self.num_elements)])
        if self.config.optimize_phases:
            names.extend([f"phase_{i}" for i in range(self.num_elements)])
        return names

    def _define_bounds(self) -> torch.Tensor:
        bounds = []
        for _ in range(self.num_spacing_params):
            bounds.append(list(self.linear_config.spacing_bounds_mm))
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
        spacing_min, spacing_max = self.linear_config.spacing_bounds_mm

        if self.linear_config.spacing_mode == "uniform":
            spacing_norm = alpha[:, idx]  # (batch,)
            spacing = spacing_min + spacing_norm * (spacing_max - spacing_min)
            idx += 1

            # Build positions with uniform spacing
            element_indices = torch.arange(
                self.num_elements, dtype=torch.float64, device=device
            )
            # Center the array
            element_indices = element_indices - element_indices.mean()
            positions_1d = spacing.unsqueeze(1) * element_indices.unsqueeze(
                0
            )  # (batch, N)

        else:  # non_uniform
            spacings_norm = alpha[
                :, idx : idx + self.num_spacing_params
            ]  # (batch, N-1)
            spacings = spacing_min + spacings_norm * (spacing_max - spacing_min)
            idx += self.num_spacing_params

            # Build positions by cumulative sum
            # First element at 0, then add spacings
            zeros = torch.zeros(batch, 1, dtype=torch.float64, device=device)
            cumsum = torch.cumsum(spacings, dim=1)  # (batch, N-1)
            positions_1d = torch.cat([zeros, cumsum], dim=1)  # (batch, N)
            # Center the array
            positions_1d = positions_1d - positions_1d.mean(dim=1, keepdim=True)

        # Convert to 2D positions based on orientation
        zeros = torch.zeros_like(positions_1d)
        if self.linear_config.orientation == "x":
            positions = torch.stack([positions_1d, zeros], dim=-1)  # (batch, N, 2)
        else:  # y
            positions = torch.stack([zeros, positions_1d], dim=-1)

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
        """Default: mid-spacing, uniform amplitudes, zero phases."""
        alpha = torch.zeros(self.num_params, dtype=torch.float64, device=device)
        idx = 0

        # Mid-spacing for all spacing params
        alpha[idx : idx + self.num_spacing_params] = 0.5
        idx += self.num_spacing_params

        if self.config.optimize_amplitudes:
            alpha[idx : idx + self.num_elements] = 1.0
            idx += self.num_elements
        if self.config.optimize_phases:
            alpha[idx : idx + self.num_elements] = 0.0

        return alpha


@dataclass
class TaperedLinearParamArrayConfig(LinearParamArrayConfig):
    """Config for tapered linear arrays with amplitude tapering."""

    taper_type: str = "taylor"  # 'taylor', 'chebyshev', 'cosine', 'custom'
    sidelobe_level_db: float = -30.0  # For Taylor/Chebyshev


class TaperedLinearParamArray(LinearParamArray):
    """
    Linear array with built-in amplitude tapering options.

    Provides common tapering functions for sidelobe control.
    """

    def __init__(self, config: Optional[TaperedLinearParamArrayConfig] = None):
        self.taper_config = config or TaperedLinearParamArrayConfig()
        super().__init__(self.taper_config)

    def get_taper_weights(self, device: str = "cpu") -> torch.Tensor:
        """Get amplitude taper weights based on taper_type."""
        n = self.num_elements

        if self.taper_config.taper_type == "uniform":
            return torch.ones(n, dtype=torch.float64, device=device)

        elif self.taper_config.taper_type == "cosine":
            # Raised cosine taper
            idx = torch.arange(n, dtype=torch.float64, device=device)
            return 0.5 * (1 + torch.cos(torch.pi * (idx - (n - 1) / 2) / (n - 1)))

        elif self.taper_config.taper_type == "hamming":
            idx = torch.arange(n, dtype=torch.float64, device=device)
            return 0.54 - 0.46 * torch.cos(2 * torch.pi * idx / (n - 1))

        elif self.taper_config.taper_type == "hanning":
            idx = torch.arange(n, dtype=torch.float64, device=device)
            return 0.5 * (1 - torch.cos(2 * torch.pi * idx / (n - 1)))

        elif self.taper_config.taper_type == "taylor":
            # Simplified Taylor-like taper (approximation)
            # Full Taylor requires more complex computation
            sll = abs(self.taper_config.sidelobe_level_db)
            a = torch.arccosh(torch.tensor(10 ** (sll / 20.0))) / torch.pi
            idx = torch.arange(n, dtype=torch.float64, device=device)
            normalized = 2 * idx / (n - 1) - 1  # [-1, 1]
            return torch.cosh(
                a * torch.pi * torch.sqrt(1 - normalized**2)
            ) / torch.cosh(a * torch.pi)

        else:
            # Default to uniform
            return torch.ones(n, dtype=torch.float64, device=device)

    def decode(self, alpha: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Decode with optional automatic tapering."""
        result = super().decode(alpha)

        # If not optimizing amplitudes, apply taper
        if not self.config.optimize_amplitudes:
            device = alpha.device if alpha.dim() > 0 else "cpu"
            taper = self.get_taper_weights(device)
            is_batched = alpha.dim() == 2
            if is_batched:
                result["amplitudes"] = taper.unsqueeze(0).expand(alpha.shape[0], -1)
            else:
                result["amplitudes"] = taper

        return result
