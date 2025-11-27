import os
import json
import time
import math
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

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
    auto_device,
    cone_power_objective,
)
from src.sim_helpers.arrays import evaluate_array_torch, C0
from src.sim_helpers.results import (
    cone_power_torch,
    find_peak_direction_torch,
    directivity_torch,
)
from src.cst_constructors.cst_patch_constructor import create_circular_array

RESULTS_DIR = Path("results/case_study_1_spiral_wpt")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_ELEMENTS = 16
FREQ_HZ = 6.5e9
WAVELENGTH_MM = C0 / FREQ_HZ * 1000
SPACING_REF_MM = WAVELENGTH_MM * 0.5
CONE_HALF_ANGLE_DEG = 25.0
PATCH_RADIUS_MM = WAVELENGTH_MM * 0.3
SUBSTRATE_THICKNESS_MM = 1.27
MAX_ITERS = 150
DEVICE = auto_device()

print(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")


def create_param_array():
    min_spacing = max(SPACING_REF_MM * 0.4, 2 * PATCH_RADIUS_MM + 1.0)
    base_config = ParamArrayConfig(
        freq_hz=FREQ_HZ,
        position_unit="mm",
        element_type="circ_patch",
        element_dims=(PATCH_RADIUS_MM,),
        optimize_amplitudes=True,
        optimize_phases=False,
        min_spacing_mm=min_spacing,
        penalty_weight=100.0,
    )
    return SpiralParamArray(
        n_elements=N_ELEMENTS,
        radius_scale_bounds=(SPACING_REF_MM * 0.5, SPACING_REF_MM * 1.5),
        angle_offset_bounds=(-0.5, 0.5),  # Per-element angle deviation in radians
        config=base_config,
    )


def run_optimizer(name, optimizer, param_array):
    print(f"\n{'='*60}")
    print(f"Running {name}...")
    print(f"{'='*60}")

    start_time = time.time()
    result = optimizer.run()
    elapsed = time.time() - start_time

    result["elapsed_time"] = elapsed
    result["optimizer_name"] = name
    result["num_evaluations"] = (
        len(optimizer.history)
        if hasattr(optimizer, "history")
        else len(result.get("history", []))
    )
    result["num_iterations"] = result["num_evaluations"]

    m = result["metrics"]
    cone_pct = m.get("cone_power_percent", m.get("cone_power", 0) * 100)
    print(f"  Completed in {elapsed:.2f}s ({result['num_iterations']} iterations)")
    print(f"  Cone Power: {cone_pct:.1f}%")
    print(f"  Directivity: {m.get('directivity_dbi', 0):.2f} dBi")
    print(
        f"  Peak at: theta={m.get('peak_theta_deg', 0):.1f} deg, phi={m.get('peak_phi_deg', 0):.1f} deg"
    )
    print(f"  Objective: {m.get('objective', 0):.4f}")

    return result


def main():
    print("=" * 70)
    print("Case Study 1: Spiral Array for Wireless Power Transfer")
    print("=" * 70)

    pa_test = create_param_array()
    print(f"Array: {N_ELEMENTS}-element Fermat Spiral")
    print(f"Frequency: {FREQ_HZ/1e9:.1f} GHz (ISM band)")
    print(f"Wavelength: {WAVELENGTH_MM:.2f} mm")
    print(f"Cone Half-Angle: {CONE_HALF_ANGLE_DEG}°")
    print(f"Optimizable params: {pa_test.num_params}")
    print(f"  - Radius scale")
    print(f"  - Per-element angle offsets ({N_ELEMENTS} params)")
    print(f"  - Geometry only (no amplitude/phase optimization)")
    print(f"Results: {RESULTS_DIR}")

    all_results = {}

    # Gradient-based optimizer (Adam) - use cone power as primary objective
    # Objective: cone_ratio (0-1), matches other optimizers
    pa_grad = create_param_array()
    grad_config = GradientConfig(
        max_iters=MAX_ITERS,
        learning_rate=0.01,
        optimizer="adam",
        weight_directivity=0.0,
        weight_cone_power=1.0,
        weight_sll=0.0,
        cone_half_angle_deg=CONE_HALF_ANGLE_DEG,
        device=DEVICE,
    )
    grad_opt = GradientOptimizer(pa_grad, grad_config)
    all_results["Gradient (Adam)"] = run_optimizer("Gradient (Adam)", grad_opt, pa_grad)

    pa_ga = create_param_array()
    ga_config = GeneticConfig(
        population_size=50,
        max_generations=MAX_ITERS,
        cone_half_angle_deg=CONE_HALF_ANGLE_DEG,
        objective_fn=cone_power_objective,
        device=DEVICE,
    )
    ga_opt = GeneticOptimizer(pa_ga, ga_config)
    all_results["Genetic Algorithm"] = run_optimizer("Genetic Algorithm", ga_opt, pa_ga)

    pa_pso = create_param_array()
    pso_config = ParticleSwarmConfig(
        swarm_size=50,
        max_iterations=MAX_ITERS,
        cone_half_angle_deg=CONE_HALF_ANGLE_DEG,
        objective_fn=cone_power_objective,
        device=DEVICE,
    )
    pso_opt = ParticleSwarmOptimizer(pa_pso, pso_config)
    all_results["PSO"] = run_optimizer("PSO", pso_opt, pa_pso)

    pa_de = create_param_array()
    de_config = DifferentialEvolutionConfig(
        population_size=50,
        max_generations=MAX_ITERS,
        cone_half_angle_deg=CONE_HALF_ANGLE_DEG,
        objective_fn=cone_power_objective,
        device=DEVICE,
    )
    de_opt = DifferentialEvolutionOptimizer(pa_de, de_config)
    all_results["Differential Evolution"] = run_optimizer(
        "Differential Evolution", de_opt, pa_de
    )

    save_results(all_results)
    generate_plots(all_results)

    print("\n" + "=" * 70)
    print("Creating CST Model from Gradient-Optimized Array")
    print("=" * 70)

    create_cst_model(all_results["Gradient (Adam)"])

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)

    print_summary_table(all_results)


def save_results(all_results):
    summary_data = []
    for name, res in all_results.items():
        m = res["metrics"]
        cone_pct = m.get("cone_power_percent", m.get("cone_power", 0) * 100)
        summary_data.append(
            {
                "Optimizer": name,
                "Objective": m.get("objective", 0),
                "Cone_Power_%": cone_pct,
                "Directivity_dBi": m.get("directivity_dbi", 0),
                "Peak_Theta_deg": m.get("peak_theta_deg", 0),
                "Peak_Phi_deg": m.get("peak_phi_deg", 0),
                "Cone_Half_Angle_deg": CONE_HALF_ANGLE_DEG,
                "Time_s": res["elapsed_time"],
                "Iterations": res["num_iterations"],
                "Evaluations": res["num_evaluations"],
            }
        )

    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(RESULTS_DIR / "summary_results.csv", index=False)
    print(f"\nSaved: {RESULTS_DIR / 'summary_results.csv'}")

    for name, res in all_results.items():
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")

        def to_serializable(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            if isinstance(x, np.ndarray):
                if np.iscomplexobj(x):
                    return {"real": x.real.tolist(), "imag": x.imag.tolist()}
                return x.tolist()
            if isinstance(x, (np.floating, np.integer)):
                return float(x)
            return x

        weights = res["weights"]
        if weights is not None:
            if isinstance(weights, torch.Tensor):
                weights_np = weights.detach().cpu().numpy()
            else:
                weights_np = np.array(weights)
            amplitudes = np.abs(weights_np)
            phases_rad = np.angle(weights_np)
            phases_deg = np.degrees(phases_rad)
        else:
            amplitudes = None
            phases_rad = None
            phases_deg = None

        params = {
            "optimizer": name,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "n_elements": N_ELEMENTS,
                "freq_hz": FREQ_HZ,
                "wavelength_mm": WAVELENGTH_MM,
                "cone_half_angle_deg": CONE_HALF_ANGLE_DEG,
                "max_iterations": MAX_ITERS,
            },
            "results": {
                "alpha": to_serializable(res["alpha"]),
                "positions_mm": to_serializable(res["positions"]),
                "weights": to_serializable(res["weights"]),
                "amplitudes": to_serializable(amplitudes),
                "phases_rad": to_serializable(phases_rad),
                "phases_deg": to_serializable(phases_deg),
            },
            "metrics": {k: to_serializable(v) for k, v in res["metrics"].items()},
            "performance": {
                "elapsed_time_s": res["elapsed_time"],
                "num_iterations": res["num_iterations"],
                "num_evaluations": res["num_evaluations"],
            },
        }

        with open(RESULTS_DIR / f"params_{safe_name}.json", "w") as f:
            json.dump(params, f, indent=2)

        history = res.get("history", [])
        if history:
            df_hist = pd.DataFrame(history)
            df_hist.to_csv(RESULTS_DIR / f"history_{safe_name}.csv", index=False)

    print(f"Saved parameter files and histories to {RESULTS_DIR}")


def generate_plots(all_results):
    def get_cone_power_ratio(m):
        if "cone_power_ratio" in m:
            return m["cone_power_ratio"]
        elif "cone_power" in m:
            return m["cone_power"]
        return 0

    best_name = max(
        all_results.keys(),
        key=lambda k: get_cone_power_ratio(all_results[k]["metrics"]),
    )
    best_res = all_results[best_name]

    colors = {
        "Gradient (Adam)": "#2196F3",
        "Genetic Algorithm": "#4CAF50",
        "PSO": "#FF9800",
        "Differential Evolution": "#F44336",
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    for name, res in all_results.items():
        hist = res.get("history", [])
        if hist and len(hist) > 1:
            if "objective" in hist[0]:
                y = [h["objective"] for h in hist]
            elif "best_fitness" in hist[0]:
                y = [h["best_fitness"] for h in hist]
            else:
                continue
            ax.plot(y, label=name, color=colors.get(name, "gray"), linewidth=2)

    ax.set_xlabel("Iteration / Generation", fontsize=12)
    ax.set_ylabel("Objective Value", fontsize=12)
    ax.set_title(
        f"Convergence Comparison - Spiral WPT (Cone {CONE_HALF_ANGLE_DEG}°)",
        fontsize=14,
    )
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "convergence_comparison.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 8))
    for name, res in all_results.items():
        time_s = res["elapsed_time"]
        cone = res["metrics"].get("cone_power_percent", 0)
        ax.scatter(
            time_s,
            cone,
            s=200,
            c=colors.get(name, "gray"),
            label=name,
            edgecolors="black",
            linewidth=1.5,
        )
        ax.annotate(
            name.split()[0],
            (time_s, cone),
            fontsize=8,
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax.set_xlabel("Computation Time (seconds)", fontsize=12)
    ax.set_ylabel("Cone Power (%)", fontsize=12)
    ax.set_title("Performance vs Computation Time", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "time_vs_performance.png", dpi=150)
    plt.close()

    pa = create_param_array()
    pos_data = best_res["positions"]
    positions_t = (
        pos_data.clone().detach()
        if isinstance(pos_data, torch.Tensor)
        else torch.tensor(pos_data, dtype=torch.float64)
    )
    wt_data = best_res["weights"]
    weights_t = (
        wt_data.clone().detach()
        if isinstance(wt_data, torch.Tensor)
        else (
            torch.tensor(wt_data, dtype=torch.complex128)
            if wt_data is not None
            else None
        )
    )

    if positions_t.dim() == 2:
        positions_t = positions_t.unsqueeze(0)
    if weights_t is not None and weights_t.dim() == 1:
        weights_t = weights_t.unsqueeze(0)

    power = evaluate_array_torch(
        positions_t,
        weights_t,
        freq_hz=FREQ_HZ,
        element_type="circ_patch",
        element_dims=(PATCH_RADIUS_MM,),
        position_unit="mm",
    )
    power_np = power.squeeze().detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 8))
    power_db = 10 * np.log10(power_np / power_np.max() + 1e-12)

    im = ax.imshow(
        power_db, extent=[0, 360, 90, 0], aspect="auto", cmap="jet", vmin=-30, vmax=0
    )

    theta_grid = np.linspace(0, 90, power_np.shape[0])
    cone_idx = np.argmin(np.abs(theta_grid - CONE_HALF_ANGLE_DEG))
    ax.axhline(
        y=CONE_HALF_ANGLE_DEG,
        color="white",
        linestyle="--",
        linewidth=2,
        label=f"Cone {CONE_HALF_ANGLE_DEG}°",
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Normalized Power (dB)", fontsize=12)
    ax.set_xlabel("Phi (degrees)", fontsize=12)
    ax.set_ylabel("Theta (degrees)", fontsize=12)
    ax.set_title(
        f"Radiation Pattern - {best_name}\n"
        f"Cone Power: {best_res['metrics'].get('cone_power_percent', 0):.1f}%",
        fontsize=14,
    )
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "pattern_2d_heatmap.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 8))
    positions = best_res["positions"]
    if isinstance(positions, torch.Tensor):
        positions = positions.detach().cpu().numpy()

    weights = best_res["weights"]
    if isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()

    if weights is not None:
        amplitudes = np.abs(weights)
        scatter = ax.scatter(
            positions[:, 0],
            positions[:, 1],
            c=amplitudes,
            cmap="viridis",
            s=200,
            edgecolors="black",
            linewidth=2,
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Amplitude", fontsize=12)
    else:
        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            s=200,
            c="blue",
            edgecolors="black",
            linewidth=2,
        )

    theta = np.linspace(0, 6 * np.pi, 200)
    r = np.sqrt(theta) * np.max(np.linalg.norm(positions, axis=1)) / np.sqrt(6 * np.pi)
    x_spiral = r * np.cos(theta)
    y_spiral = r * np.sin(theta)
    ax.plot(x_spiral, y_spiral, "k--", alpha=0.3, label="Fermat spiral")

    ax.set_xlabel("X (mm)", fontsize=12)
    ax.set_ylabel("Y (mm)", fontsize=12)
    ax.set_title(f"Optimized Spiral Array Layout - {best_name}", fontsize=14)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "spiral_layout.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    names = list(all_results.keys())
    bar_colors = [colors.get(n, "gray") for n in names]

    cone_pct = [all_results[n]["metrics"].get("cone_power_percent", 0) for n in names]
    axes[0, 0].bar(range(len(names)), cone_pct, color=bar_colors)
    axes[0, 0].set_xticks(range(len(names)))
    axes[0, 0].set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    axes[0, 0].set_ylabel("Cone Power (%)", fontsize=11)
    axes[0, 0].set_title(f"Cone Power ({CONE_HALF_ANGLE_DEG}° cone)", fontsize=12)
    axes[0, 0].set_ylim([0, 105])

    dirs = [all_results[n]["metrics"].get("directivity_dbi", 0) for n in names]
    axes[0, 1].bar(range(len(names)), dirs, color=bar_colors)
    axes[0, 1].set_xticks(range(len(names)))
    axes[0, 1].set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    axes[0, 1].set_ylabel("Directivity (dBi)", fontsize=11)
    axes[0, 1].set_title("Boresight Directivity", fontsize=12)

    times = [all_results[n]["elapsed_time"] for n in names]
    axes[1, 0].bar(range(len(names)), times, color=bar_colors)
    axes[1, 0].set_xticks(range(len(names)))
    axes[1, 0].set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    axes[1, 0].set_ylabel("Time (seconds)", fontsize=11)
    axes[1, 0].set_title("Computation Time", fontsize=12)
    axes[1, 0].set_yscale("log")

    objs = [all_results[n]["metrics"].get("objective", 0) for n in names]
    axes[1, 1].bar(range(len(names)), objs, color=bar_colors)
    axes[1, 1].set_xticks(range(len(names)))
    axes[1, 1].set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    axes[1, 1].set_ylabel("Objective Value", fontsize=11)
    axes[1, 1].set_title("Final Objective", fontsize=12)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "comparison_bars.png", dpi=150)
    plt.close()

    print(f"Saved plots to {RESULTS_DIR}")


def create_cst_model(gradient_result):
    return "Wait"
    positions = gradient_result["positions"]
    if isinstance(positions, torch.Tensor):
        positions = positions.detach().cpu().numpy()

    positions_list = [(float(p[0]), float(p[1])) for p in positions]
    patch_radius = PATCH_RADIUS_MM

    try:
        cst_path = create_circular_array(
            project_name=f"case_study_1_spiral_wpt_{N_ELEMENTS}elem",
            positions=positions_list,
            radii=patch_radius,
            freq_hz=FREQ_HZ,
            substrate_thickness=SUBSTRATE_THICKNESS_MM,
            dielectric_constant=2.2,
            loss_tangent=0.0009,
            copper_thickness=0.035,
            board_margin=5.0,
            target_impedance=50.0,
        )
        print(f"CST model created: {cst_path}")

        with open(RESULTS_DIR / "cst_model_info.json", "w") as f:
            json.dump(
                {
                    "cst_path": cst_path,
                    "array_config": {
                        "n_elements": N_ELEMENTS,
                        "freq_hz": FREQ_HZ,
                        "patch_radius_mm": patch_radius,
                        "cone_half_angle_deg": CONE_HALF_ANGLE_DEG,
                    },
                    "positions_mm": positions_list,
                    "weights": (
                        {
                            "amplitudes": np.abs(
                                gradient_result["weights"].detach().cpu().numpy()
                            ).tolist(),
                            "phases_deg": np.degrees(
                                np.angle(
                                    gradient_result["weights"].detach().cpu().numpy()
                                )
                            ).tolist(),
                        }
                        if gradient_result["weights"] is not None
                        else None
                    ),
                },
                f,
                indent=2,
            )

    except Exception as e:
        print(f"  CST model creation failed: {e}")
        print("  (This is expected if CST is not installed)")


def print_summary_table(all_results):
    print("\n" + "=" * 90)
    print("SUMMARY COMPARISON TABLE")
    print("=" * 90)
    print(
        f"{'Optimizer':<30} {'Time (s)':<10} {'Cone Power %':<14} {'Directivity':<12} {'Objective':<12}"
    )
    print("-" * 90)

    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1]["metrics"].get("cone_power_ratio", 0),
        reverse=True,
    )

    for name, res in sorted_results:
        m = res["metrics"]
        time_s = res["elapsed_time"]
        cone = m.get("cone_power_percent", 0)
        dir_dbi = m.get("directivity_dbi", 0)
        obj = m.get("objective", 0)

        print(f"{name:<30} {time_s:<10.3f} {cone:<14.1f} {dir_dbi:<12.2f} {obj:<12.4f}")

    print("=" * 90)

    best_cone = sorted_results[0]
    fastest = min(all_results.items(), key=lambda x: x[1]["elapsed_time"])

    print(
        f"\nBest Cone Power: {best_cone[0]} ({best_cone[1]['metrics'].get('cone_power_percent', 0):.1f}%)"
    )
    print(f"Fastest Optimizer: {fastest[0]} ({fastest[1]['elapsed_time']:.3f}s)")


if __name__ == "__main__":
    main()
