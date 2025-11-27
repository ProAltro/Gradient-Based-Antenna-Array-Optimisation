from __future__ import annotations

from dataclasses import dataclass, field
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
class GeneticConfig:
    population_size: int = 50
    max_generations: int = 100
    elite_count: int = 2

    tournament_size: int = 3

    crossover_prob: float = 0.9
    crossover_type: str = "blend"
    blend_alpha: float = 0.5

    mutation_prob: float = 0.1
    mutation_scale: float = 0.1

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


class GeneticOptimizer:
    def __init__(self, param_array: ParamArray, config: Optional[GeneticConfig] = None):
        self.param_array = param_array
        self.config = config or GeneticConfig()
        self.history: List[Dict[str, float]] = []

    def _evaluate_population(
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

    def _tournament_selection(
        self, population: torch.Tensor, fitness: torch.Tensor
    ) -> torch.Tensor:
        cfg = self.config
        pop_size = population.shape[0]
        num_params = population.shape[1]
        selected = torch.zeros_like(population)

        for i in range(pop_size):
            indices = torch.randint(0, pop_size, (cfg.tournament_size,))
            tournament_fitness = fitness[indices]
            winner_idx = indices[tournament_fitness.argmax()]
            selected[i] = population[winner_idx]

        return selected

    def _crossover(
        self, parent1: torch.Tensor, parent2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config

        if torch.rand(1).item() > cfg.crossover_prob:
            return parent1.clone(), parent2.clone()

        if cfg.crossover_type == "blend":
            alpha = cfg.blend_alpha
            d = torch.abs(parent1 - parent2)
            low = torch.min(parent1, parent2) - alpha * d
            high = torch.max(parent1, parent2) + alpha * d
            child1 = low + torch.rand_like(parent1) * (high - low)
            child2 = low + torch.rand_like(parent2) * (high - low)

        elif cfg.crossover_type == "uniform":
            mask = torch.rand_like(parent1) < 0.5
            child1 = torch.where(mask, parent1, parent2)
            child2 = torch.where(mask, parent2, parent1)

        elif cfg.crossover_type == "single_point":
            point = torch.randint(1, parent1.shape[0], (1,)).item()
            child1 = torch.cat([parent1[:point], parent2[point:]])
            child2 = torch.cat([parent2[:point], parent1[point:]])

        else:
            raise ValueError(f"Unknown crossover type: {cfg.crossover_type}")

        return child1, child2

    def _mutate(self, individual: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        mutated = individual.clone()

        mutation_mask = torch.rand_like(individual) < cfg.mutation_prob
        noise = torch.randn_like(individual) * cfg.mutation_scale
        mutated = torch.where(mutation_mask, individual + noise, individual)

        return mutated

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

        best_alpha = None
        best_fitness = float("-inf")
        best_metrics = {}
        self.history = []
        stall_count = 0

        for generation in range(cfg.max_generations):
            fitness, metrics_list = self._evaluate_population(population)

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

            sorted_indices = fitness.argsort(descending=True)
            new_population = [
                population[sorted_indices[i]].clone() for i in range(cfg.elite_count)
            ]

            parents = self._tournament_selection(population, fitness)
            parent_idx = 0
            while len(new_population) < cfg.population_size:
                p1 = parents[parent_idx % cfg.population_size]
                p2 = parents[(parent_idx + 1) % cfg.population_size]
                child1, child2 = self._crossover(p1, p2)
                new_population.append(self._mutate(child1))
                if len(new_population) < cfg.population_size:
                    new_population.append(self._mutate(child2))
                parent_idx += 2

            population = torch.stack(new_population)
            population = pa.clip(population)

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
