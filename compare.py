import json
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Callable, Optional

from src.param_arrays.grid import GridParamArray, GridParamArrayConfig
from src.param_arrays.hexagonal import HexagonalParamArray, HexagonalParamArrayConfig
from src.param_arrays.spiral import SpiralParamArray
from src.param_arrays.base import ParamArrayConfig
from src.dl_optimisers import (
    GradientOptimizer,
    GradientConfig,
    GeneticOptimizer,
    GeneticConfig,
    ParticleSwarmOptimizer,
    ParticleSwarmConfig,
    DifferentialEvolutionOptimizer,
    DifferentialEvolutionConfig,
    cone_power_objective,
    directivity_objective,
    beam_steering_objective,
)
from src.sim_helpers.arrays import C0

RESULTS_DIR = Path("results/compare")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_ITERS = 100
POP_SIZE = 40

FREQUENCIES = {
    "grid": 10e9,
    "hexagonal": 28e9,
    "spiral": 6.5e9,
}

OBJECTIVES = ["directivity", "cone_power", "beam_steering"]

TARGET_THETA_DEG = 30.0
TARGET_PHI_DEG = 45.0
CONE_HALF_ANGLE_DEG = 25.0


@dataclass
class TaskConfig:
    array_type: str
    objective: str
    optimizer_name: str
    freq_hz: float


def create_grid_array(freq_hz: float):
    wavelength_mm = C0 / freq_hz * 1000
    spacing_mm = wavelength_mm * 0.5
    patch_radius_mm = wavelength_mm * 0.3
    config = GridParamArrayConfig(
        freq_hz=freq_hz,
        position_unit="mm",
        element_type="circ_patch",
        element_dims=(patch_radius_mm,),
        rows=4,
        cols=4,
        spacing_bounds_mm=(spacing_mm * 0.8, spacing_mm * 1.2),
        optimize_spacing=True,
        optimize_amplitudes=True,
        optimize_phases=True,
        min_spacing_mm=spacing_mm * 0.8,
        penalty_weight=0.0,
    )
    return GridParamArray(config=config)


def create_hexagonal_array(freq_hz: float):
    wavelength_mm = C0 / freq_hz * 1000
    spacing_mm = wavelength_mm * 0.5
    patch_radius_mm = wavelength_mm * 0.3
    min_spacing = max(spacing_mm * 0.4, 2 * patch_radius_mm + 1.0)
    config = HexagonalParamArrayConfig(
        freq_hz=freq_hz,
        position_unit="mm",
        element_type="circ_patch",
        element_dims=(patch_radius_mm,),
        num_rings=2,  # 2 rings = 19 elements (closest to 16)
        spacing_bounds_mm=(spacing_mm * 0.8, spacing_mm * 1.2),
        rotation_bounds=(0.0, 2 * 3.14159),
        position_offset_bounds_mm=(-spacing_mm * 0.15, spacing_mm * 0.15),
        optimize_spacing=True,
        optimize_rotation=True,
        optimize_position_offsets=True,
        optimize_amplitudes=False,
        optimize_phases=False,
        min_spacing_mm=min_spacing,
        penalty_weight=0.0,
    )
    return HexagonalParamArray(config=config)


def create_spiral_array(freq_hz: float):
    wavelength_mm = C0 / freq_hz * 1000
    spacing_ref_mm = wavelength_mm * 0.5
    patch_radius_mm = wavelength_mm * 0.3
    min_spacing = max(spacing_ref_mm * 0.4, 2 * patch_radius_mm + 1.0)
    config = ParamArrayConfig(
        freq_hz=freq_hz,
        position_unit="mm",
        element_type="circ_patch",
        element_dims=(patch_radius_mm,),
        optimize_amplitudes=True,
        optimize_phases=True,
        min_spacing_mm=min_spacing,
        penalty_weight=0.0,
    )
    return SpiralParamArray(
        n_elements=16,
        radius_scale_bounds=(spacing_ref_mm * 0.5, spacing_ref_mm * 1.5),
        angle_offset_bounds=(-0.5, 0.5),
        config=config,
    )


def create_array(array_type: str, freq_hz: float):
    if array_type == "grid":
        return create_grid_array(freq_hz)
    elif array_type == "hexagonal":
        return create_hexagonal_array(freq_hz)
    elif array_type == "spiral":
        return create_spiral_array(freq_hz)
    else:
        raise ValueError(f"Unknown array type: {array_type}")


def get_objective_fn(objective: str):
    if objective == "directivity":
        return directivity_objective
    elif objective == "cone_power":
        return cone_power_objective
    elif objective == "beam_steering":
        return beam_steering_objective
    else:
        raise ValueError(f"Unknown objective: {objective}")


def run_single_optimization(task: TaskConfig) -> dict:
    array_type = task.array_type
    objective = task.objective
    optimizer_name = task.optimizer_name
    freq_hz = task.freq_hz

    pa = create_array(array_type, freq_hz)
    objective_fn = get_objective_fn(objective)

    start_time = time.time()

    if optimizer_name == "Gradient":
        if objective == "directivity":
            config = GradientConfig(
                max_iters=MAX_ITERS,
                learning_rate=0.01,
                optimizer="adam",
                weight_directivity=1.0,
                weight_cone_power=0.0,
                weight_sll=0.0,
                weight_beam_steering=0.0,
                device="cpu",
            )
        elif objective == "cone_power":
            config = GradientConfig(
                max_iters=MAX_ITERS,
                learning_rate=0.01,
                optimizer="adam",
                weight_directivity=0.0,
                weight_cone_power=1.0,
                weight_sll=0.0,
                weight_beam_steering=0.0,
                cone_half_angle_deg=CONE_HALF_ANGLE_DEG,
                device="cpu",
            )
        else:
            config = GradientConfig(
                max_iters=MAX_ITERS,
                learning_rate=0.05,
                optimizer="adam",
                weight_directivity=0.0,
                weight_cone_power=0.0,
                weight_sll=0.0,
                weight_beam_steering=1.0,
                target_theta_deg=TARGET_THETA_DEG,
                target_phi_deg=TARGET_PHI_DEG,
                device="cpu",
            )
        opt = GradientOptimizer(pa, config)

    elif optimizer_name == "GA":
        config = GeneticConfig(
            population_size=POP_SIZE,
            max_generations=MAX_ITERS,
            cone_half_angle_deg=CONE_HALF_ANGLE_DEG,
            target_theta_deg=TARGET_THETA_DEG,
            target_phi_deg=TARGET_PHI_DEG,
            objective_fn=objective_fn,
            device="cpu",
        )
        opt = GeneticOptimizer(pa, config)

    elif optimizer_name == "PSO":
        config = ParticleSwarmConfig(
            swarm_size=POP_SIZE,
            max_iterations=MAX_ITERS,
            cone_half_angle_deg=CONE_HALF_ANGLE_DEG,
            target_theta_deg=TARGET_THETA_DEG,
            target_phi_deg=TARGET_PHI_DEG,
            objective_fn=objective_fn,
            device="cpu",
        )
        opt = ParticleSwarmOptimizer(pa, config)

    elif optimizer_name == "DE":
        config = DifferentialEvolutionConfig(
            population_size=POP_SIZE,
            max_generations=MAX_ITERS,
            cone_half_angle_deg=CONE_HALF_ANGLE_DEG,
            target_theta_deg=TARGET_THETA_DEG,
            target_phi_deg=TARGET_PHI_DEG,
            objective_fn=objective_fn,
            device="cpu",
        )
        opt = DifferentialEvolutionOptimizer(pa, config)

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    result = opt.run()
    elapsed = time.time() - start_time

    metrics = result["metrics"]

    return {
        "array_type": array_type,
        "objective": objective,
        "optimizer": optimizer_name,
        "freq_hz": freq_hz,
        "num_elements": pa.num_elements,
        "num_params": pa.num_params,
        "time_s": elapsed,
        "final_objective": metrics.get("objective", 0),
        "directivity_dbi": metrics.get("directivity_dbi", 0),
        "cone_power_percent": metrics.get(
            "cone_power_percent", metrics.get("cone_power", 0) * 100
        ),
        "steering_efficiency": metrics.get("steering_efficiency", 0),
        "gain_at_target_dbi": metrics.get("gain_at_target_dbi", 0),
        "peak_theta_deg": metrics.get("peak_theta_deg", 0),
        "peak_phi_deg": metrics.get("peak_phi_deg", 0),
    }


def run_task_wrapper(task_dict: dict) -> dict:
    task = TaskConfig(**task_dict)
    try:
        return run_single_optimization(task)
    except Exception as e:
        return {
            "array_type": task.array_type,
            "objective": task.objective,
            "optimizer": task.optimizer_name,
            "freq_hz": task.freq_hz,
            "error": str(e),
        }


def main():
    print("=" * 70)
    print("Comprehensive Optimizer Comparison")
    print("=" * 70)
    print(f"Arrays: grid (4x4=16 elem), hexagonal (2 rings=19 elem), spiral (16 elem)")
    print(f"Objectives: {OBJECTIVES}")
    print(f"Optimizers: Gradient (Adam), GA, PSO, DE")
    print(f"Max iterations: {MAX_ITERS}, Population: {POP_SIZE}")
    print(f"Results: {RESULTS_DIR}")
    print()

    tasks = []
    optimizers = ["Gradient", "GA", "PSO", "DE"]

    for array_type in ["grid", "hexagonal", "spiral"]:
        freq_hz = FREQUENCIES[array_type]
        for objective in OBJECTIVES:
            for opt_name in optimizers:
                tasks.append(
                    {
                        "array_type": array_type,
                        "objective": objective,
                        "optimizer_name": opt_name,
                        "freq_hz": freq_hz,
                    }
                )

    total_tasks = len(tasks)
    print(f"Total tasks: {total_tasks}")
    print()

    start_time = time.time()

    results = []
    for i, task_dict in enumerate(tasks, 1):
        task = TaskConfig(**task_dict)
        print(
            f"[{i}/{total_tasks}] {task.array_type}/{task.objective}/{task.optimizer_name}...",
            end=" ",
            flush=True,
        )
        try:
            result = run_single_optimization(task)
            results.append(result)
            print(
                f"done (obj={result['final_objective']:.4f}, {result['time_s']:.1f}s)"
            )
        except Exception as e:
            results.append(
                {
                    "array_type": task.array_type,
                    "objective": task.objective,
                    "optimizer": task.optimizer_name,
                    "freq_hz": task.freq_hz,
                    "error": str(e),
                }
            )
            print(f"FAILED: {e}")

    total_time = time.time() - start_time

    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    print(f"\nCompleted {len(successful)}/{total_tasks} tasks in {total_time:.1f}s")
    if failed:
        print(f"Failed: {len(failed)}")
        for f in failed:
            print(
                f"  {f['array_type']}/{f['objective']}/{f['optimizer']}: {f['error']}"
            )

    df = pd.DataFrame(successful)
    df.to_csv(RESULTS_DIR / "all_results.csv", index=False)
    print(f"\nSaved: {RESULTS_DIR / 'all_results.csv'}")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_time_s": total_time,
        "num_tasks": total_tasks,
        "successful": len(successful),
        "failed": len(failed),
        "config": {
            "max_iters": MAX_ITERS,
            "pop_size": POP_SIZE,
            "frequencies": FREQUENCIES,
            "objectives": OBJECTIVES,
            "target_theta_deg": TARGET_THETA_DEG,
            "target_phi_deg": TARGET_PHI_DEG,
            "cone_half_angle_deg": CONE_HALF_ANGLE_DEG,
        },
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("RESULTS BY ARRAY TYPE AND OBJECTIVE")
    print("=" * 70)

    for array_type in ["grid", "hexagonal", "spiral"]:
        print(f"\n{array_type.upper()} Array ({FREQUENCIES[array_type]/1e9:.1f} GHz)")
        print("-" * 60)

        for objective in OBJECTIVES:
            subset = [
                r
                for r in successful
                if r["array_type"] == array_type and r["objective"] == objective
            ]
            if not subset:
                continue

            print(f"  {objective}:")
            best = max(subset, key=lambda x: x["final_objective"])
            for r in sorted(subset, key=lambda x: x["final_objective"], reverse=True):
                marker = " *" if r["optimizer"] == best["optimizer"] else ""
                print(
                    f"    {r['optimizer']:<10} obj={r['final_objective']:8.4f}  time={r['time_s']:6.2f}s{marker}"
                )

    print("\n" + "=" * 70)
    print("BEST OPTIMIZER PER CONFIGURATION")
    print("=" * 70)

    best_results = []
    for array_type in ["grid", "hexagonal", "spiral"]:
        for objective in OBJECTIVES:
            subset = [
                r
                for r in successful
                if r["array_type"] == array_type and r["objective"] == objective
            ]
            if subset:
                best = max(subset, key=lambda x: x["final_objective"])
                best_results.append(best)
                print(
                    f"{array_type:12} + {objective:15} -> {best['optimizer']:<10} (obj={best['final_objective']:.4f})"
                )

    df_best = pd.DataFrame(best_results)
    df_best.to_csv(RESULTS_DIR / "best_results.csv", index=False)
    print(f"\nSaved: {RESULTS_DIR / 'best_results.csv'}")


if __name__ == "__main__":
    main()
