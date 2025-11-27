"""Random/Sparse parameterized array with fully learnable positions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch

from .base import ParamArray, ParamArrayConfig


@dataclass
class RandomParamArrayConfig(ParamArrayConfig):
    """Config for random/sparse arrays."""

    n_elements: int = 16
    aperture_bounds_mm: Tuple[float, float] = (-50.0, 50.0)  # x and y range
    aperture_x_bounds_mm: Optional[Tuple[float, float]] = None  # Override for x
    aperture_y_bounds_mm: Optional[Tuple[float, float]] = None  # Override for y


class RandomParamArray(ParamArray):
    """
    Random/Sparse array with fully learnable element positions.

    Each element has independent x, y coordinates that can be optimized.
    This is the most flexible array geometry, suitable for:
    - Sparse array design
    - Thinned arrays
    - Aperiodic arrays
    - Custom geometries

    Learnable params:
    - x_i, y_i (2*N params) for each element position
    - amplitudes (N params) if optimize_amplitudes
    - phases (N params) if optimize_phases
    """

    def __init__(self, config: Optional[RandomParamArrayConfig] = None):
        self.random_config = config or RandomParamArrayConfig()
        super().__init__(self.random_config)

    @property
    def num_elements(self) -> int:
        return self.random_config.n_elements

    @property
    def num_params(self) -> int:
        n = 2 * self.num_elements  # x, y for each
        if self.config.optimize_amplitudes:
            n += self.num_elements
        if self.config.optimize_phases:
            n += self.num_elements
        return n

    @property
    def param_names(self) -> list[str]:
        names = []
        for i in range(self.num_elements):
            names.extend([f"x_{i}", f"y_{i}"])
        if self.config.optimize_amplitudes:
            names.extend([f"amp_{i}" for i in range(self.num_elements)])
        if self.config.optimize_phases:
            names.extend([f"phase_{i}" for i in range(self.num_elements)])
        return names

    def _get_x_bounds(self) -> Tuple[float, float]:
        if self.random_config.aperture_x_bounds_mm is not None:
            return self.random_config.aperture_x_bounds_mm
        return self.random_config.aperture_bounds_mm

    def _get_y_bounds(self) -> Tuple[float, float]:
        if self.random_config.aperture_y_bounds_mm is not None:
            return self.random_config.aperture_y_bounds_mm
        return self.random_config.aperture_bounds_mm

    def _define_bounds(self) -> torch.Tensor:
        bounds = []
        x_bounds = list(self._get_x_bounds())
        y_bounds = list(self._get_y_bounds())

        for _ in range(self.num_elements):
            bounds.append(x_bounds)
            bounds.append(y_bounds)

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

        # Positions (x, y interleaved)
        n_pos_params = 2 * self.num_elements
        pos_norm = alpha[:, idx : idx + n_pos_params]  # (batch, 2*N)
        pos_min = bounds[idx : idx + n_pos_params, 0]
        pos_range = bounds[idx : idx + n_pos_params, 1] - pos_min
        pos_physical = pos_min + pos_norm * pos_range  # (batch, 2*N)

        # Reshape to (batch, N, 2)
        positions = pos_physical.reshape(batch, self.num_elements, 2)
        result["positions"] = positions
        idx += n_pos_params

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
        """Default: centered positions (all at origin)."""
        alpha = torch.full((self.num_params,), 0.5, dtype=torch.float64, device=device)
        idx = 2 * self.num_elements
        if self.config.optimize_amplitudes:
            alpha[idx : idx + self.num_elements] = 1.0
            idx += self.num_elements
        if self.config.optimize_phases:
            alpha[idx : idx + self.num_elements] = 0.0
        return alpha

    def random_init(self, batch_size: int = 1, device: str = "cpu") -> torch.Tensor:
        """Generate random initial parameters with good coverage."""
        alpha = torch.rand(
            batch_size, self.num_params, dtype=torch.float64, device=device
        )

        # For positions, use stratified sampling for better coverage
        # This helps avoid initial configurations with overlapping elements
        n_pos = 2 * self.num_elements

        # Simple uniform random is usually fine, but we ensure amplitudes start high
        idx = n_pos
        if self.config.optimize_amplitudes:
            alpha[:, idx : idx + self.num_elements] = 0.8 + 0.2 * torch.rand(
                batch_size, self.num_elements, dtype=torch.float64, device=device
            )
            idx += self.num_elements

        return alpha

    def grid_init(self, device: str = "cpu") -> torch.Tensor:
        """Initialize on a regular grid (good starting point)."""
        alpha = torch.zeros(self.num_params, dtype=torch.float64, device=device)

        # Arrange elements on approximate grid
        n = self.num_elements
        cols = int(torch.ceil(torch.sqrt(torch.tensor(float(n)))))
        rows = int(torch.ceil(torch.tensor(float(n)) / cols))

        idx = 0
        for i in range(n):
            row = i // cols
            col = i % cols
            # Normalized positions [0, 1]
            alpha[idx] = (col + 0.5) / cols  # x
            alpha[idx + 1] = (row + 0.5) / rows  # y
            idx += 2

        idx = 2 * self.num_elements
        if self.config.optimize_amplitudes:
            alpha[idx : idx + self.num_elements] = 1.0
            idx += self.num_elements
        if self.config.optimize_phases:
            alpha[idx : idx + self.num_elements] = 0.0

        return alpha
