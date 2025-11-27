from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable, List, Tuple
import torch

from ..param_arrays.base import ParamArray
from ..sim_helpers.arrays import evaluate_array_torch
from ..sim_helpers.results import combined_objective_torch


ObjectiveFn = Callable[
    [torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, Any],
    Tuple[torch.Tensor, List[Dict[str, float]]],
]


@dataclass
class ParticleSwarmConfig:
    swarm_size: int = 50
    max_iterations: int = 100

    inertia_weight: float = 0.7
    cognitive_coeff: float = 1.5
    social_coeff: float = 1.5

    inertia_start: float = 0.9
    inertia_end: float = 0.4
    use_inertia_decay: bool = True

    max_velocity: float = 0.2

    stall_iterations: int = 30
    improvement_tol: float = 1e-4

    weight_directivity: float = 1.0
    weight_cone_power: float = 0.0
    weight_sll: float = 0.0
    cone_half_angle_deg: float = 15.0

    objective_fn: Optional[ObjectiveFn] = None

    target_theta_deg: float = 0.0
    target_phi_deg: float = 0.0

    device: str = "cpu"


class ParticleSwarmOptimizer:
    def __init__(
        self, param_array: ParamArray, config: Optional[ParticleSwarmConfig] = None
    ):
        self.param_array = param_array
        self.config = config or ParticleSwarmConfig()
        self.history: List[Dict[str, float]] = []

    def _evaluate_swarm(
        self, swarm_params: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Dict[str, float]]]:
        cfg = self.config
        pa = self.param_array
        pac = pa.config
        swarm_size = swarm_params.shape[0]

        positions = pa.build_positions(swarm_params)

        if pac.optimize_amplitudes or pac.optimize_phases:
            weights_list = [pa.get_weights(swarm_params[i]) for i in range(swarm_size)]
            weights = torch.stack(weights_list, dim=0)
        else:
            weights = None

        power = evaluate_array_torch(
            positions,
            weights,
            freq_hz=pac.freq_hz,
            element_type=pac.element_type,
            element_dims=pac.element_dims,
            position_unit=pac.position_unit,
            device=cfg.device,
        )

        penalties = pa.compute_spacing_penalty(positions)

        if cfg.objective_fn is not None:
            fitness, metrics_list = cfg.objective_fn(
                power, positions, weights, penalties, cfg
            )
        else:
            fitness = combined_objective_torch(
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
                )

                dir_dbi = directivity_torch(power)
                cone_pwr = cone_power_torch(power, cfg.cone_half_angle_deg)
                sll_db = sll_torch(power)
                min_spacings = pa.minimum_spacing(positions)

                metrics_list = []
                for i in range(swarm_size):
                    metrics = {
                        "objective": fitness[i].item(),
                        "directivity_dbi": dir_dbi[i].item(),
                        "cone_power": cone_pwr[i].item(),
                        "sll_db": sll_db[i].item(),
                        "min_spacing_mm": min_spacings[i].item(),
                    }
                    metrics_list.append(metrics)

        return fitness, metrics_list

    def run(
        self,
        initial_positions: Optional[torch.Tensor] = None,
        callback: Optional[Callable[[int, torch.Tensor, Dict], None]] = None,
    ) -> Dict[str, Any]:
        cfg = self.config
        pa = self.param_array
        num_params = pa.num_params

        if initial_positions is None:
            swarm_pos = pa.random_init(cfg.swarm_size, device=cfg.device)
        else:
            swarm_pos = initial_positions.to(cfg.device)

        swarm_pos = pa.clip(swarm_pos)

        swarm_vel = (
            torch.rand(
                cfg.swarm_size, num_params, device=cfg.device, dtype=torch.float64
            )
            * 2
            - 1
        ) * cfg.max_velocity

        fitness, metrics_list = self._evaluate_swarm(swarm_pos)

        pbest_pos = swarm_pos.clone()
        pbest_fitness = fitness.clone()
        pbest_metrics = metrics_list.copy()

        gbest_idx = fitness.argmax().item()
        gbest_pos = swarm_pos[gbest_idx].clone()
        gbest_fitness = fitness[gbest_idx].item()
        gbest_metrics = metrics_list[gbest_idx]

        self.history = []
        stall_count = 0

        for iteration in range(cfg.max_iterations):
            if cfg.use_inertia_decay:
                w = cfg.inertia_start - (cfg.inertia_start - cfg.inertia_end) * (
                    iteration / cfg.max_iterations
                )
            else:
                w = cfg.inertia_weight

            for i in range(cfg.swarm_size):
                r1 = torch.rand(num_params, device=cfg.device, dtype=torch.float64)
                r2 = torch.rand(num_params, device=cfg.device, dtype=torch.float64)

                cognitive = cfg.cognitive_coeff * r1 * (pbest_pos[i] - swarm_pos[i])
                social = cfg.social_coeff * r2 * (gbest_pos - swarm_pos[i])
                swarm_vel[i] = w * swarm_vel[i] + cognitive + social

                swarm_vel[i] = torch.clamp(
                    swarm_vel[i], -cfg.max_velocity, cfg.max_velocity
                )

                swarm_pos[i] = swarm_pos[i] + swarm_vel[i]

            swarm_pos = pa.clip(swarm_pos)

            fitness, metrics_list = self._evaluate_swarm(swarm_pos)

            for i in range(cfg.swarm_size):
                if fitness[i] > pbest_fitness[i]:
                    pbest_fitness[i] = fitness[i]
                    pbest_pos[i] = swarm_pos[i].clone()
                    pbest_metrics[i] = metrics_list[i]

            iter_best_idx = fitness.argmax().item()
            iter_best_fitness = fitness[iter_best_idx].item()

            if iter_best_fitness > gbest_fitness + cfg.improvement_tol:
                gbest_fitness = iter_best_fitness
                gbest_pos = swarm_pos[iter_best_idx].clone()
                gbest_metrics = metrics_list[iter_best_idx].copy()
                stall_count = 0
            else:
                stall_count += 1

            gbest_metrics["iteration"] = iteration

            iter_stats = {
                "iteration": iteration,
                "best_fitness": gbest_fitness,
                "mean_fitness": fitness.mean().item(),
                "std_fitness": fitness.std().item(),
                "best_directivity_dbi": gbest_metrics["directivity_dbi"],
                "inertia_weight": w,
            }
            self.history.append(iter_stats)

            if callback:
                callback(iteration, gbest_pos, gbest_metrics)

            if stall_count >= cfg.stall_iterations:
                print(
                    f"Converged: no improvement for {cfg.stall_iterations} iterations"
                )
                break

        with torch.no_grad():
            decoded = pa.decode(gbest_pos)
            positions = decoded["positions"]
            weights = pa.get_weights(gbest_pos)

        return {
            "alpha": gbest_pos,
            "positions": positions,
            "weights": weights,
            "metrics": gbest_metrics,
            "history": self.history,
        }
