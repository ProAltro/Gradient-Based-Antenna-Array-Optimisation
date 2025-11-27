import torch
from .base import ParamArray


class SpiralParamArray(ParamArray):
    """Parameterized Fermat spiral antenna array with golden angle spacing.

    Parameters (normalized [0,1]):
    - radius_scale: Scaling factor for the Fermat spiral radii
    - angle_offset_i: Per-element angle offsets (deviation from golden ratio steps)

    For element i:
    - theta_i = i * golden_angle + angle_offset_i
    - r_i = radius_scale * sqrt(i + 1)

    Positions: [r_i * cos(theta_i), r_i * sin(theta_i)]

    Golden angle ≈ 137.5° (2π / φ² where φ = (1+√5)/2)
    """

    def __init__(
        self,
        n_elements=16,
        radius_scale_bounds=(0.1, 10.0),
        angle_offset_bounds=(-0.5, 0.5),  # Per-element angle offset in radians
        config=None,
    ):
        self.n_elements = n_elements
        self.radius_scale_bounds = radius_scale_bounds
        self.angle_offset_bounds = angle_offset_bounds
        # Golden angle
        phi = (1 + torch.sqrt(torch.tensor(5.0))) / 2
        self.golden_angle = 2 * torch.pi / (phi * phi)
        super().__init__(config)

    @property
    def num_params(self) -> int:
        n = 1 + self.n_elements  # radius_scale, per-element angle_offsets
        if self.config.optimize_amplitudes:
            n += self.n_elements
        if self.config.optimize_phases:
            n += self.n_elements
        return n

    @property
    def num_elements(self) -> int:
        return self.n_elements

    @property
    def param_names(self) -> list[str]:
        names = ["radius_scale"] + [f"angle_offset_{i}" for i in range(self.n_elements)]
        if self.config.optimize_amplitudes:
            names.extend([f"amplitude_{i}" for i in range(self.n_elements)])
        if self.config.optimize_phases:
            names.extend([f"phase_{i}" for i in range(self.n_elements)])
        return names

    def _define_bounds(self) -> torch.Tensor:
        bounds = [self.radius_scale_bounds]
        bounds += [self.angle_offset_bounds] * self.n_elements
        if self.config.optimize_amplitudes:
            bounds.extend([[0.0, 1.0]] * self.n_elements)
        if self.config.optimize_phases:
            bounds.extend([[0.0, 2 * torch.pi]] * self.n_elements)
        return torch.tensor(bounds, dtype=torch.float64)  # (num_params, 2)

    def decode(self, alpha: torch.Tensor) -> dict:
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(0)

        batch_size = alpha.shape[0]
        device = alpha.device

        idx = 0
        result = {}

        # Position parameters
        radius_scale_alpha = alpha[:, idx]
        idx += 1
        angle_offset_alphas = alpha[:, idx : idx + self.n_elements]
        idx += self.n_elements

        # Denormalize position parameters
        radius_min, radius_max = self.radius_scale_bounds
        offset_min, offset_max = self.angle_offset_bounds
        radius_scale = radius_min + radius_scale_alpha * (radius_max - radius_min)
        angle_offsets = offset_min + angle_offset_alphas * (offset_max - offset_min)

        # Compute angles and radii for each element
        i = torch.arange(self.n_elements, dtype=torch.float64, device=device)
        theta_base = i * self.golden_angle
        theta = theta_base.unsqueeze(0) + angle_offsets  # (batch, n_elements)
        r = radius_scale.unsqueeze(-1) * torch.sqrt(i + 1)  # (batch, n_elements)

        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        positions = torch.stack([x, y], dim=-1)
        result["positions"] = positions

        if self.config.optimize_amplitudes:
            amp_norm = alpha[:, idx : idx + self.n_elements]
            result["amplitudes"] = amp_norm
            idx += self.n_elements

        if self.config.optimize_phases:
            phase_norm = alpha[:, idx : idx + self.n_elements]
            result["phases"] = phase_norm * 2 * torch.pi
            idx += self.n_elements

        if alpha.shape[0] == 1:
            result = {k: v.squeeze(0) for k, v in result.items()}

        return result
