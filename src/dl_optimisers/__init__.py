import torch
from typing import Dict, List, Tuple, Optional, Any

from .gradient import GradientOptimizer, GradientConfig
from .genetic import GeneticOptimizer, GeneticConfig
from .differential_evolution import (
    DifferentialEvolutionOptimizer,
    DifferentialEvolutionConfig,
)
from .particle_swarm import ParticleSwarmOptimizer, ParticleSwarmConfig


def get_device(prefer_cuda: bool = True) -> str:
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def auto_device() -> str:
    return get_device(prefer_cuda=True)


def cone_power_objective(
    power: torch.Tensor,
    positions: torch.Tensor,
    weights: Optional[torch.Tensor],
    penalties: torch.Tensor,
    config: Any,
) -> Tuple[torch.Tensor, List[Dict[str, float]]]:
    from ..sim_helpers.results import (
        directivity_torch,
        cone_power_torch,
        find_peak_direction_torch,
    )

    batch_size = power.shape[0]
    cone_half_angle = getattr(config, "cone_half_angle_deg", 15.0)

    cone_ratio = cone_power_torch(power, cone_half_angle)
    dir_dbi = directivity_torch(power)
    peak_theta, peak_phi, _ = find_peak_direction_torch(power)

    fitness = cone_ratio

    metrics_list = []
    for i in range(batch_size):
        metrics = {
            "objective": fitness[i].item(),
            "cone_power_ratio": cone_ratio[i].item(),
            "cone_power_percent": cone_ratio[i].item() * 100,
            "directivity_dbi": dir_dbi[i].item(),
            "peak_theta_deg": peak_theta[i].item(),
            "peak_phi_deg": peak_phi[i].item(),
        }
        metrics_list.append(metrics)

    return fitness, metrics_list


def directivity_objective(
    power: torch.Tensor,
    positions: torch.Tensor,
    weights: Optional[torch.Tensor],
    penalties: torch.Tensor,
    config: Any,
) -> Tuple[torch.Tensor, List[Dict[str, float]]]:
    from ..sim_helpers.results import (
        directivity_torch,
        sll_torch,
        find_peak_direction_torch,
    )

    batch_size = power.shape[0]
    weight_sll = getattr(config, "weight_sll", 0.0)

    dir_dbi = directivity_torch(power)
    sll_db = sll_torch(power)
    peak_theta, peak_phi, _ = find_peak_direction_torch(power)

    fitness = dir_dbi + weight_sll * sll_db

    metrics_list = []
    for i in range(batch_size):
        metrics = {
            "objective": fitness[i].item(),
            "directivity_dbi": dir_dbi[i].item(),
            "sll_db": sll_db[i].item(),
            "peak_theta_deg": peak_theta[i].item(),
            "peak_phi_deg": peak_phi[i].item(),
        }
        metrics_list.append(metrics)

    return fitness, metrics_list


def beam_steering_objective(
    power: torch.Tensor,
    positions: torch.Tensor,
    weights: Optional[torch.Tensor],
    penalties: torch.Tensor,
    config: Any,
) -> Tuple[torch.Tensor, List[Dict[str, float]]]:
    from ..sim_helpers.results import (
        beam_steering_objective_torch,
        directivity_torch,
        find_peak_direction_torch,
    )

    batch_size = power.shape[0]
    target_theta = getattr(config, "target_theta_deg", 0.0)
    target_phi = getattr(config, "target_phi_deg", 0.0)

    steering_eff, gain_at_target = beam_steering_objective_torch(
        power, target_theta, target_phi
    )
    dir_dbi = directivity_torch(power)
    peak_theta, peak_phi, _ = find_peak_direction_torch(power)

    fitness = 100 * steering_eff + gain_at_target

    metrics_list = []
    for i in range(batch_size):
        metrics = {
            "objective": fitness[i].item(),
            "steering_efficiency": steering_eff[i].item(),
            "gain_at_target_dbi": gain_at_target[i].item(),
            "directivity_dbi": dir_dbi[i].item(),
            "peak_theta_deg": peak_theta[i].item(),
            "peak_phi_deg": peak_phi[i].item(),
            "target_theta_deg": target_theta,
            "target_phi_deg": target_phi,
        }
        metrics_list.append(metrics)

    return fitness, metrics_list


__all__ = [
    "get_device",
    "auto_device",
    "cone_power_objective",
    "directivity_objective",
    "beam_steering_objective",
    "GradientOptimizer",
    "GradientConfig",
    "GeneticOptimizer",
    "GeneticConfig",
    "DifferentialEvolutionOptimizer",
    "DifferentialEvolutionConfig",
    "ParticleSwarmOptimizer",
    "ParticleSwarmConfig",
]
