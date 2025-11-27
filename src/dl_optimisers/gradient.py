from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, List
import torch

from ..param_arrays.base import ParamArray
from ..sim_helpers.arrays import evaluate_array_torch
from ..sim_helpers.results import combined_objective_torch


@dataclass
class GradientConfig:
    max_iters: int = 100
    learning_rate: float = 0.01
    optimizer: str = "adam"

    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    grad_tol: float = 1e-6
    improvement_tol: float = 1e-4
    stall_iters: int = 20

    weight_directivity: float = 1.0
    weight_cone_power: float = 0.0
    weight_sll: float = 0.0
    cone_half_angle_deg: float = 15.0

    device: str = "cpu"


class GradientOptimizer:
    def __init__(
        self, param_array: ParamArray, config: Optional[GradientConfig] = None
    ):
        self.param_array = param_array
        self.config = config or GradientConfig()
        self.history: List[Dict[str, float]] = []

    def _create_optimizer(self, params: List[torch.Tensor]) -> torch.optim.Optimizer:
        cfg = self.config
        if cfg.optimizer == "adam":
            return torch.optim.Adam(
                params, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps
            )
        elif cfg.optimizer == "sgd":
            return torch.optim.SGD(params, lr=cfg.learning_rate)
        elif cfg.optimizer == "lbfgs":
            return torch.optim.LBFGS(params, lr=cfg.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    def evaluate(self, alpha: torch.Tensor) -> tuple[torch.Tensor, Dict[str, float]]:
        cfg = self.config
        pa = self.param_array
        pac = pa.config

        positions = pa.build_positions(alpha)
        weights = pa.get_weights(alpha)

        if positions.dim() == 2:
            positions = positions.unsqueeze(0)
        if weights is not None and weights.dim() == 1:
            weights = weights.unsqueeze(0)

        power = evaluate_array_torch(
            positions,
            weights,
            freq_hz=pac.freq_hz,
            element_type=pac.element_type,
            element_dims=pac.element_dims,
            position_unit=pac.position_unit,
            device=cfg.device,
        )

        objective = combined_objective_torch(
            power,
            weight_directivity=cfg.weight_directivity,
            weight_cone_power=cfg.weight_cone_power,
            weight_sll=cfg.weight_sll,
            cone_half_angle_deg=cfg.cone_half_angle_deg,
            penalty=0.0,
        )

        with torch.no_grad():
            from ..sim_helpers.results import (
                directivity_torch,
                cone_power_torch,
                sll_torch,
                find_peak_direction_torch,
            )

            cone_pwr = cone_power_torch(power, cfg.cone_half_angle_deg).item()
            peak_theta, peak_phi, _ = find_peak_direction_torch(power)

            metrics = {
                "objective": objective.item(),
                "directivity_dbi": directivity_torch(power).item(),
                "cone_power": cone_pwr,
                "cone_power_ratio": cone_pwr,
                "cone_power_percent": cone_pwr * 100,
                "sll_db": sll_torch(power).item(),
                "min_spacing_mm": pa.minimum_spacing(positions).item(),
                "peak_theta_deg": (
                    peak_theta.item() if hasattr(peak_theta, "item") else peak_theta
                ),
                "peak_phi_deg": (
                    peak_phi.item() if hasattr(peak_phi, "item") else peak_phi
                ),
            }

        return objective, metrics

    def run(
        self,
        initial: Optional[torch.Tensor] = None,
        callback: Optional[Callable[[int, torch.Tensor, Dict], None]] = None,
    ) -> Dict[str, Any]:
        cfg = self.config
        pa = self.param_array

        if initial is None:
            alpha = pa.default_init(device=cfg.device)
        else:
            alpha = initial.to(cfg.device)

        alpha = alpha.clone().requires_grad_(True)
        optimizer = self._create_optimizer([alpha])

        best_alpha = alpha.detach().clone()
        best_objective = float("-inf")
        best_metrics = None
        self.history = []
        stall_count = 0

        for iteration in range(cfg.max_iters):
            optimizer.zero_grad()

            objective, metrics = self.evaluate(alpha)

            loss = -objective
            loss.backward()

            grad_norm = alpha.grad.norm().item() if alpha.grad is not None else 0.0
            metrics["grad_norm"] = grad_norm
            metrics["iteration"] = iteration

            if (
                best_metrics is None
                or metrics["objective"] > best_objective + cfg.improvement_tol
            ):
                best_objective = metrics["objective"]
                best_alpha = alpha.detach().clone()
                best_metrics = metrics.copy()
                stall_count = 0
            else:
                stall_count += 1

            self.history.append(metrics)

            if callback:
                callback(iteration, alpha.detach(), metrics)

            if grad_norm < cfg.grad_tol:
                print(f"Converged: grad_norm {grad_norm:.2e} < {cfg.grad_tol:.2e}")
                break

            if stall_count >= cfg.stall_iters:
                print(f"Stalled: no improvement for {cfg.stall_iters} iterations")
                break

            optimizer.step()

            with torch.no_grad():
                alpha.data = pa.clip(alpha.data)

        with torch.no_grad():
            decoded = pa.decode(best_alpha)
            positions = decoded["positions"]
            weights = pa.get_weights(best_alpha)

        return {
            "alpha": best_alpha,
            "positions": positions,
            "weights": weights,
            "metrics": best_metrics,
            "history": self.history,
        }
