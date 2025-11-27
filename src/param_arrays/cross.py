"""Cross/Plus shaped parameterized array."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch

from .base import ParamArray, ParamArrayConfig


@dataclass
class CrossParamArrayConfig(ParamArrayConfig):
    """Config for cross/plus shaped arrays."""

    elements_per_arm: int = 4  # Elements per arm (not counting center)
    spacing_bounds_mm: Tuple[float, float] = (3.0, 15.0)
    rotation_bounds: Tuple[float, float] = (0.0, 1.5707963267948966)  # 0 to pi/2
    shape: str = "plus"  # 'plus' (+) or 'x' (×)
    include_center: bool = True
    optimize_spacing: bool = True
    optimize_rotation: bool = True
    independent_arm_spacing: bool = False  # If True, each arm has its own spacing


class CrossParamArray(ParamArray):
    """
    Cross-shaped array (+ or ×).

    Elements are arranged along two perpendicular axes.
    - Plus (+): arms along x and y axes
    - X (×): arms at 45° angles

    Learnable params (depending on config):
    - spacing (1 or 4 params) if optimize_spacing
    - rotation (1 param) if optimize_rotation
    - amplitudes (N params) if optimize_amplitudes
    - phases (N params) if optimize_phases
    """

    def __init__(self, config: Optional[CrossParamArrayConfig] = None):
        self.cross_config = config or CrossParamArrayConfig()
        super().__init__(self.cross_config)

    @property
    def num_elements(self) -> int:
        n = 4 * self.cross_config.elements_per_arm
        if self.cross_config.include_center:
            n += 1
        return n

    @property
    def num_spacing_params(self) -> int:
        if not self.cross_config.optimize_spacing:
            return 0
        if self.cross_config.independent_arm_spacing:
            return 4  # One per arm
        return 1  # Single spacing for all arms

    @property
    def num_params(self) -> int:
        n = self.num_spacing_params
        if self.cross_config.optimize_rotation:
            n += 1
        if self.config.optimize_amplitudes:
            n += self.num_elements
        if self.config.optimize_phases:
            n += self.num_elements
        return n

    @property
    def param_names(self) -> list[str]:
        names = []
        if self.cross_config.optimize_spacing:
            if self.cross_config.independent_arm_spacing:
                names.extend([f"spacing_arm_{i}" for i in range(4)])
            else:
                names.append("spacing")
        if self.cross_config.optimize_rotation:
            names.append("rotation")
        if self.config.optimize_amplitudes:
            names.extend([f"amp_{i}" for i in range(self.num_elements)])
        if self.config.optimize_phases:
            names.extend([f"phase_{i}" for i in range(self.num_elements)])
        return names

    def _define_bounds(self) -> torch.Tensor:
        bounds = []
        if self.cross_config.optimize_spacing:
            for _ in range(self.num_spacing_params):
                bounds.append(list(self.cross_config.spacing_bounds_mm))
        if self.cross_config.optimize_rotation:
            bounds.append(list(self.cross_config.rotation_bounds))
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
        spacing_min, spacing_max = self.cross_config.spacing_bounds_mm
        if self.cross_config.optimize_spacing:
            spacing_norm = alpha[:, idx : idx + self.num_spacing_params]
            spacings = spacing_min + spacing_norm * (spacing_max - spacing_min)
            idx += self.num_spacing_params

            if self.cross_config.independent_arm_spacing:
                arm_spacings = spacings  # (batch, 4)
            else:
                arm_spacings = spacings.expand(batch, 4)  # (batch, 4)
        else:
            mid = (spacing_min + spacing_max) / 2
            arm_spacings = torch.full(
                (batch, 4), mid, dtype=torch.float64, device=device
            )

        # Rotation
        if self.cross_config.optimize_rotation:
            rotation_norm = alpha[:, idx]
            rot_min, rot_max = self.cross_config.rotation_bounds
            rotation = rot_min + rotation_norm * (rot_max - rot_min)
            idx += 1
        else:
            rotation = torch.zeros(batch, dtype=torch.float64, device=device)

        # Base angles for the 4 arms
        if self.cross_config.shape == "plus":
            base_angles = torch.tensor(
                [0, torch.pi / 2, torch.pi, 3 * torch.pi / 2],
                dtype=torch.float64,
                device=device,
            )
        else:  # 'x'
            base_angles = torch.tensor(
                [torch.pi / 4, 3 * torch.pi / 4, 5 * torch.pi / 4, 7 * torch.pi / 4],
                dtype=torch.float64,
                device=device,
            )

        # Apply rotation
        arm_angles = base_angles.unsqueeze(0) + rotation.unsqueeze(1)  # (batch, 4)

        # Build positions
        positions_list = []

        # Center element
        if self.cross_config.include_center:
            center = torch.zeros(batch, 1, 2, dtype=torch.float64, device=device)
            positions_list.append(center)

        # Arm elements
        for arm_idx in range(4):
            angle = arm_angles[:, arm_idx]  # (batch,)
            spacing = arm_spacings[:, arm_idx]  # (batch,)

            for elem_idx in range(1, self.cross_config.elements_per_arm + 1):
                r = spacing * elem_idx
                x = r * torch.cos(angle)
                y = r * torch.sin(angle)
                pos = torch.stack([x, y], dim=-1).unsqueeze(1)  # (batch, 1, 2)
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
        """Default: mid-spacing, zero rotation."""
        alpha = torch.zeros(self.num_params, dtype=torch.float64, device=device)
        idx = 0

        if self.cross_config.optimize_spacing:
            alpha[idx : idx + self.num_spacing_params] = 0.5
            idx += self.num_spacing_params
        if self.cross_config.optimize_rotation:
            alpha[idx] = 0.0
            idx += 1
        if self.config.optimize_amplitudes:
            alpha[idx : idx + self.num_elements] = 1.0
            idx += self.num_elements
        if self.config.optimize_phases:
            alpha[idx : idx + self.num_elements] = 0.0

        return alpha

    def get_arm_indices(self) -> Dict[str, list]:
        """Get element indices for each arm (useful for arm-specific weighting)."""
        indices = {}
        idx = 0

        if self.cross_config.include_center:
            indices["center"] = [0]
            idx = 1
        else:
            indices["center"] = []

        arm_names = (
            ["right", "up", "left", "down"]
            if self.cross_config.shape == "plus"
            else ["up_right", "up_left", "down_left", "down_right"]
        )

        for arm_name in arm_names:
            arm_indices = list(range(idx, idx + self.cross_config.elements_per_arm))
            indices[arm_name] = arm_indices
            idx += self.cross_config.elements_per_arm

        return indices
