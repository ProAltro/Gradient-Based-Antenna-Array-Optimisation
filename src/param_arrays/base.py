"""Base class for parameterized antenna array geometries."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import torch
import numpy as np


@dataclass
class ParamArrayConfig:
    """Configuration for ParamArray optimization."""

    freq_hz: float = 24.0e9
    position_unit: str = "mm"
    element_type: str = "isotropic"
    element_dims: Tuple[float, ...] = (0.0,)

    # What to optimize
    optimize_positions: bool = True
    optimize_amplitudes: bool = False
    optimize_phases: bool = False

    # Constraints
    min_spacing_mm: float = 2.0
    penalty_weight: float = 1000.0

    # Objective weights (for multi-objective)
    weight_directivity: float = 1.0
    weight_sll: float = 0.0
    weight_cone_power: float = 0.0
    cone_half_angle_deg: float = 15.0


class ParamArray(ABC):
    """
    Abstract base class for parameterized antenna arrays.

    Subclasses define specific geometries (spiral, ring, grid, etc.)
    and how learnable parameters map to element positions.
    """

    def __init__(self, config: Optional[ParamArrayConfig] = None):
        self.config = config or ParamArrayConfig()
        self._bounds: Optional[torch.Tensor] = None  # (num_params, 2)

    @property
    @abstractmethod
    def num_params(self) -> int:
        """Total number of learnable parameters."""
        pass

    @property
    @abstractmethod
    def num_elements(self) -> int:
        """Number of antenna elements."""
        pass

    @property
    @abstractmethod
    def param_names(self) -> list[str]:
        """Names of each parameter (for logging/debugging)."""
        pass

    @property
    def bounds(self) -> torch.Tensor:
        """Parameter bounds as (num_params, 2) tensor [min, max]."""
        if self._bounds is None:
            self._bounds = self._define_bounds()
        return self._bounds

    @abstractmethod
    def _define_bounds(self) -> torch.Tensor:
        """Define min/max bounds for each parameter. Returns (num_params, 2)."""
        pass

    @abstractmethod
    def decode(self, alpha: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decode normalized [0,1] parameters to physical values.

        Args:
            alpha: Normalized parameters, shape (num_params,) or (batch, num_params)

        Returns:
            Dict with at least 'positions' key, shape (batch, num_elements, 2)
            May also include 'amplitudes', 'phases' if optimizing those.
        """
        pass

    def encode(self, physical_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode physical parameters to normalized [0,1] values.

        Args:
            physical_params: Dict with physical parameter values

        Returns:
            Normalized parameters, shape (num_params,) or (batch, num_params)
        """
        # Default implementation assumes linear scaling
        # Subclasses may override for non-linear encodings
        raise NotImplementedError("Subclass should implement encode() if needed")

    def build_positions(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Build element positions from normalized parameters.

        Args:
            alpha: Normalized parameters, shape (num_params,) or (batch, num_params)

        Returns:
            Positions tensor, shape (batch, num_elements, 2)
        """
        decoded = self.decode(alpha)
        return decoded["positions"]

    def get_weights(self, alpha: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Get complex weights (amplitude * exp(1j * phase)) if optimizing.

        Returns None if not optimizing amplitudes/phases.
        """
        if not (self.config.optimize_amplitudes or self.config.optimize_phases):
            return None

        decoded = self.decode(alpha)

        if self.config.optimize_amplitudes and self.config.optimize_phases:
            amp = decoded["amplitudes"]
            phase = decoded["phases"]
            return amp * torch.exp(1j * phase)
        elif self.config.optimize_amplitudes:
            return decoded["amplitudes"].to(torch.complex128)
        else:  # phases only
            return torch.exp(1j * decoded["phases"])

    def compute_spacing_penalty(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute penalty for elements too close together.

        Args:
            positions: Shape (batch, num_elements, 2)

        Returns:
            Penalty value (scalar or batch,)
        """
        # Pairwise distances
        diff = positions[:, :, None, :] - positions[:, None, :, :]  # (B, N, N, 2)
        dist = torch.sqrt((diff**2).sum(dim=-1) + 1e-12)  # (B, N, N)

        # Mask diagonal
        mask = ~torch.eye(self.num_elements, dtype=torch.bool, device=positions.device)
        dist_masked = dist[:, mask].reshape(positions.shape[0], -1)

        # Penalty for violations
        min_dist = dist_masked.min(dim=-1).values
        violation = torch.clamp(self.config.min_spacing_mm - min_dist, min=0.0)
        penalty = self.config.penalty_weight * violation**2

        return penalty

    def minimum_spacing(self, positions: torch.Tensor) -> torch.Tensor:
        """Get minimum element spacing. Shape (batch,)."""
        diff = positions[:, :, None, :] - positions[:, None, :, :]
        dist = torch.sqrt((diff**2).sum(dim=-1) + 1e-12)
        mask = ~torch.eye(self.num_elements, dtype=torch.bool, device=positions.device)
        dist_masked = dist[:, mask].reshape(positions.shape[0], -1)
        return dist_masked.min(dim=-1).values

    def random_init(self, batch_size: int = 1, device: str = "cpu") -> torch.Tensor:
        """Generate random initial parameters in [0, 1]."""
        return torch.rand(
            batch_size, self.num_params, dtype=torch.float64, device=device
        )

    def default_init(self, device: str = "cpu") -> torch.Tensor:
        """Default initialization (typically centered/uniform)."""
        # Default: mid-point of bounds → 0.5 in normalized space
        return torch.full((self.num_params,), 0.5, dtype=torch.float64, device=device)

    def clip(self, alpha: torch.Tensor) -> torch.Tensor:
        """Clip parameters to [0, 1] bounds."""
        return torch.clamp(alpha, 0.0, 1.0)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_elements={self.num_elements}, "
            f"num_params={self.num_params}, "
            f"optimize_amplitudes={self.config.optimize_amplitudes}, "
            f"optimize_phases={self.config.optimize_phases})"
        )
