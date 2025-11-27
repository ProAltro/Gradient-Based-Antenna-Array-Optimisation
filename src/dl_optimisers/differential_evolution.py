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
class DifferentialEvolutionConfig:
    population_size: int = 50
    max_generations: int = 100

    mutation_factor: float = 0.8
    crossover_prob: float = 0.9
    strategy: str = "best/1/bin"

    stall_generations: int = 30
    improvement_tol: float = 1e-4

    weight_directivity: float = 1.0
    weight_cone_power: float = 0.0
    weight_sll: float = 0.0
    cone_half_angle_deg: float = 15.0

    objective_fn: Optional[ObjectiveFn] = None

    target_theta_deg: float = 0.0
    target_phi_deg: float = 0.0

    device: str = "cpu"


class DifferentialEvolutionOptimizer:
    def __init__(
        self,
        param_array: ParamArray,
        config: Optional[DifferentialEvolutionConfig] = None,
    ):
        self.param_array = param_array
        self.config = config or DifferentialEvolutionConfig()
        self.history: List[Dict[str, float]] = []

    def _evaluate_batch(
        self, population: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Dict[str, float]]]:
        cfg = self.config
        pa = self.param_array
        pac = pa.config
        pop_size = population.shape[0]

        positions = pa.build_positions(population)

        if pac.optimize_amplitudes or pac.optimize_phases:
            weights_list = [pa.get_weights(population[i]) for i in range(pop_size)]
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
                for i in range(pop_size):
                    metrics = {
                        "objective": fitness[i].item(),
                        "directivity_dbi": dir_dbi[i].item(),
                        "cone_power": cone_pwr[i].item(),
                        "sll_db": sll_db[i].item(),
                        "min_spacing_mm": min_spacings[i].item(),
                    }
                    metrics_list.append(metrics)

        return fitness, metrics_list

    def _evaluate_population(
        self, population: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Dict[str, float]]]:
        return self._evaluate_batch(population)

    def _mutation(
        self,
        population: torch.Tensor,
        fitness: torch.Tensor,
        target_idx: int,
    ) -> torch.Tensor:
        cfg = self.config
        pop_size = population.shape[0]
        F = cfg.mutation_factor

        available = [i for i in range(pop_size) if i != target_idx]

        if cfg.strategy.startswith("best"):
            best_idx = fitness.argmax().item()
            base = population[best_idx]
        else:
            base_idx = available[torch.randint(0, len(available), (1,)).item()]
            base = population[base_idx]
            available.remove(base_idx)

        if "/2/" in cfg.strategy and len(available) >= 4:
            indices = torch.randperm(len(available))[:4].tolist()
            diff1 = (
                population[available[indices[0]]] - population[available[indices[1]]]
            )
            diff2 = (
                population[available[indices[2]]] - population[available[indices[3]]]
            )
            mutant = base + F * (diff1 + diff2)
        else:
            indices = torch.randperm(len(available))[:2].tolist()
            diff = population[available[indices[0]]] - population[available[indices[1]]]
            mutant = base + F * diff

        return mutant

    def _crossover(self, target: torch.Tensor, mutant: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        num_params = target.shape[0]

        j_rand = torch.randint(0, num_params, (1,)).item()

        trial = target.clone()
        for j in range(num_params):
            if torch.rand(1).item() < cfg.crossover_prob or j == j_rand:
                trial[j] = mutant[j]

        return trial

    def run(
        self,
        initial_population: Optional[torch.Tensor] = None,
        callback: Optional[Callable[[int, torch.Tensor, Dict], None]] = None,
    ) -> Dict[str, Any]:
        cfg = self.config
        pa = self.param_array

        if initial_population is None:
            population = pa.random_init(cfg.population_size, device=cfg.device)
        else:
            population = initial_population.to(cfg.device)

        population = pa.clip(population)

        fitness, metrics_list = self._evaluate_population(population)

        best_idx = fitness.argmax().item()
        best_alpha = population[best_idx].clone()
        best_fitness = fitness[best_idx].item()
        best_metrics = metrics_list[best_idx]
        self.history = []
        stall_count = 0

        for generation in range(cfg.max_generations):
            trials = []
            for i in range(cfg.population_size):
                mutant = self._mutation(population, fitness, i)
                trial = self._crossover(population[i], mutant)
                trial = pa.clip(trial)
                trials.append(trial)

            trials = torch.stack(trials, dim=0)

            trial_fitness, trial_metrics_list = self._evaluate_batch(trials)

            for i in range(cfg.population_size):
                if trial_fitness[i] >= fitness[i]:
                    population[i] = trials[i]
                    fitness[i] = trial_fitness[i]
                    metrics_list[i] = trial_metrics_list[i]

            gen_best_idx = fitness.argmax().item()
            gen_best_fitness = fitness[gen_best_idx].item()
            gen_best_metrics = metrics_list[gen_best_idx]
            gen_best_metrics["generation"] = generation

            if gen_best_fitness > best_fitness + cfg.improvement_tol:
                best_fitness = gen_best_fitness
                best_alpha = population[gen_best_idx].clone()
                best_metrics = gen_best_metrics.copy()
                stall_count = 0
            else:
                stall_count += 1

            gen_stats = {
                "generation": generation,
                "best_fitness": gen_best_fitness,
                "mean_fitness": fitness.mean().item(),
                "std_fitness": fitness.std().item(),
                "best_directivity_dbi": gen_best_metrics["directivity_dbi"],
            }
            self.history.append(gen_stats)

            if callback:
                callback(generation, best_alpha, gen_best_metrics)

            if stall_count >= cfg.stall_generations:
                print(
                    f"Converged: no improvement for {cfg.stall_generations} generations"
                )
                break

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
