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

from src.param_arrays.grid import GridParamArray, GridParamArrayConfig
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
    beam_steering_objective,
)
from src.sim_helpers.arrays import evaluate_array_torch, C0
from src.sim_helpers.results import (
    find_peak_direction_torch,
    directivity_torch,
    beam_steering_objective_torch,
)
from src.cst_constructors.cst_patch_constructor import create_circular_array

RESULTS_DIR = Path("results/case_study_3_grid_beamsteering")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ROWS = 4
COLS = 4
N_ELEMENTS = ROWS * COLS
FREQ_HZ = 10e9
WAVELENGTH_MM = C0 / FREQ_HZ * 1000
SPACING_MM = WAVELENGTH_MM * 0.5
TARGET_THETA_DEG = 30.0
TARGET_PHI_DEG = 45.0
PATCH_RADIUS_MM = WAVELENGTH_MM * 0.3
SUBSTRATE_THICKNESS_MM = 0.787
MAX_ITERS = 150
DEVICE = auto_device()

print(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")


def create_param_array():
    grid_config = GridParamArrayConfig(
        freq_hz=FREQ_HZ,
        position_unit="mm",
        element_type="circ_patch",
        element_dims=(PATCH_RADIUS_MM,),
        rows=ROWS,
        cols=COLS,
        spacing_bounds_mm=(SPACING_MM * 0.8, SPACING_MM * 1.2),
        optimize_spacing=True,
        optimize_amplitudes=False,
        optimize_phases=True,
        min_spacing_mm=SPACING_MM * 0.8,
        penalty_weight=0.0,
    )
    return GridParamArray(config=grid_config)


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
    print(f"  Completed in {elapsed:.2f}s ({result['num_iterations']} iterations)")
    print(f"  Steering Efficiency: {m.get('steering_efficiency', 0)*100:.1f}%")
    print(f"  Gain at Target: {m.get('gain_at_target_dbi', 0):.2f} dBi")
    print(
        f"  Peak at: θ={m.get('peak_theta_deg', 0):.1f}°, φ={m.get('peak_phi_deg', 0):.1f}°"
    )
    print(f"  Target: θ={TARGET_THETA_DEG}°, φ={TARGET_PHI_DEG}°")
    print(f"  Objective: {m.get('objective', 0):.4f}")

    return result


def main():
    print("=" * 70)
    print("Case Study 3: 4x4 Grid Array for Beam Steering (Phase + Spacing)")
    print("=" * 70)

    pa_test = create_param_array()
    print(f"Array: {ROWS}x{COLS} = {N_ELEMENTS}-element Grid")
    print(f"Frequency: {FREQ_HZ/1e9:.1f} GHz (X-band)")
    print(f"Wavelength: {WAVELENGTH_MM:.2f} mm")
    print(f"Spacing: {SPACING_MM:.2f} mm (λ/2) ±20%")
    print(f"Target Direction: θ={TARGET_THETA_DEG}°, φ={TARGET_PHI_DEG}°")
    print(f"Optimizable params: {pa_test.num_params} (spacing + phases)")
    print(f"  - Spacing x,y (2 params)")
    print(f"  - Per-element phases ({N_ELEMENTS} params)")
    print(f"  - Total: {pa_test.num_params} parameters")
    print(f"Results: {RESULTS_DIR}")

    all_results = {}

    pa_grad = create_param_array()
    grad_config = GradientConfig(
        max_iters=MAX_ITERS,
        learning_rate=0.05,
        optimizer="adam",
        weight_directivity=0.0,
        weight_cone_power=0.0,
        weight_sll=0.0,
        weight_beam_steering=1.0,
        target_theta_deg=TARGET_THETA_DEG,
        target_phi_deg=TARGET_PHI_DEG,
        device=DEVICE,
    )
    grad_opt = GradientOptimizer(pa_grad, grad_config)
    all_results["Gradient (Adam)"] = run_optimizer("Gradient (Adam)", grad_opt, pa_grad)

    pa_ga = create_param_array()
    ga_config = GeneticConfig(
        population_size=50,
        max_generations=MAX_ITERS,
        target_theta_deg=TARGET_THETA_DEG,
        target_phi_deg=TARGET_PHI_DEG,
        objective_fn=beam_steering_objective,
        device=DEVICE,
    )
    ga_opt = GeneticOptimizer(pa_ga, ga_config)
    all_results["Genetic Algorithm"] = run_optimizer("Genetic Algorithm", ga_opt, pa_ga)

    pa_pso = create_param_array()
    pso_config = ParticleSwarmConfig(
        swarm_size=50,
        max_iterations=MAX_ITERS,
        target_theta_deg=TARGET_THETA_DEG,
        target_phi_deg=TARGET_PHI_DEG,
        objective_fn=beam_steering_objective,
        device=DEVICE,
    )
    pso_opt = ParticleSwarmOptimizer(pa_pso, pso_config)
    all_results["PSO"] = run_optimizer("PSO", pso_opt, pa_pso)

    pa_de = create_param_array()
    de_config = DifferentialEvolutionConfig(
        population_size=50,
        max_generations=MAX_ITERS,
        target_theta_deg=TARGET_THETA_DEG,
        target_phi_deg=TARGET_PHI_DEG,
        objective_fn=beam_steering_objective,
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
        summary_data.append(
            {
                "Optimizer": name,
                "Objective": m.get("objective", 0),
                "Steering_Efficiency": m.get("steering_efficiency", 0),
                "Gain_at_Target_dBi": m.get("gain_at_target_dbi", 0),
                "Directivity_dBi": m.get("directivity_dbi", 0),
                "Peak_Theta_deg": m.get("peak_theta_deg", 0),
                "Peak_Phi_deg": m.get("peak_phi_deg", 0),
                "Target_Theta_deg": TARGET_THETA_DEG,
                "Target_Phi_deg": TARGET_PHI_DEG,
                "Num_Elements": N_ELEMENTS,
                "Num_Params": res["alpha"].numel() if res["alpha"] is not None else 0,
                "Time_s": res["elapsed_time"],
                "Iterations": res["num_iterations"],
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
            phases_rad = np.angle(weights_np)
            phases_deg = np.degrees(phases_rad)
        else:
            phases_rad = None
            phases_deg = None

        params = {
            "optimizer": name,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "rows": ROWS,
                "cols": COLS,
                "n_elements": N_ELEMENTS,
                "freq_hz": FREQ_HZ,
                "wavelength_mm": WAVELENGTH_MM,
                "spacing_mm": SPACING_MM,
                "target_theta_deg": TARGET_THETA_DEG,
                "target_phi_deg": TARGET_PHI_DEG,
                "max_iterations": MAX_ITERS,
            },
            "results": {
                "alpha": to_serializable(res["alpha"]),
                "positions_mm": to_serializable(res["positions"]),
                "weights": to_serializable(res["weights"]),
                "phases_rad": to_serializable(phases_rad),
                "phases_deg": to_serializable(phases_deg),
            },
            "metrics": {k: to_serializable(v) for k, v in res["metrics"].items()},
            "performance": {
                "elapsed_time_s": res["elapsed_time"],
                "num_iterations": res["num_iterations"],
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
    best_name = max(
        all_results.keys(),
        key=lambda k: all_results[k]["metrics"].get("steering_efficiency", 0),
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
        f"Convergence Comparison - {ROWS}x{COLS} Grid Beam Steering\n"
        f"Target: θ={TARGET_THETA_DEG}°, φ={TARGET_PHI_DEG}° (Phase + Spacing)",
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
        steer_eff = res["metrics"].get("steering_efficiency", 0) * 100
        ax.scatter(
            time_s,
            steer_eff,
            s=200,
            c=colors.get(name, "gray"),
            label=name,
            edgecolors="black",
            linewidth=1.5,
        )
        ax.annotate(
            name.split()[0],
            (time_s, steer_eff),
            fontsize=8,
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax.set_xlabel("Computation Time (seconds)", fontsize=12)
    ax.set_ylabel("Steering Efficiency (%)", fontsize=12)
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

    ax.plot(
        TARGET_PHI_DEG,
        TARGET_THETA_DEG,
        "w*",
        markersize=20,
        markeredgecolor="black",
        label="Target",
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Normalized Power (dB)", fontsize=12)
    ax.set_xlabel("Phi (degrees)", fontsize=12)
    ax.set_ylabel("Theta (degrees)", fontsize=12)
    ax.set_title(
        f"Radiation Pattern - {best_name}\n"
        f"Target: θ={TARGET_THETA_DEG}°, φ={TARGET_PHI_DEG}° | "
        f"Steering Eff: {best_res['metrics'].get('steering_efficiency', 0)*100:.1f}%",
        fontsize=14,
    )
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "pattern_2d_heatmap.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 8))
    positions = best_res["positions"]
    if isinstance(positions, torch.Tensor):
        positions = positions.detach().cpu().numpy()

    weights = best_res["weights"]
    if isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()

    if weights is not None:
        phases_deg = np.degrees(np.angle(weights))
        scatter = ax.scatter(
            positions[:, 0],
            positions[:, 1],
            c=phases_deg,
            cmap="hsv",
            s=500,
            edgecolors="black",
            linewidth=2,
            vmin=-180,
            vmax=180,
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Phase (degrees)", fontsize=12)

        for i, (x, y) in enumerate(positions):
            ax.annotate(
                f"{phases_deg[i]:.0f}°",
                (x, y),
                fontsize=8,
                ha="center",
                va="center",
                fontweight="bold",
                color="white",
            )
    else:
        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            s=500,
            c="blue",
            edgecolors="black",
            linewidth=2,
        )

    ax.set_xlabel("X (mm)", fontsize=12)
    ax.set_ylabel("Y (mm)", fontsize=12)
    ax.set_title(
        f"Optimized {ROWS}x{COLS} Grid Phase Distribution - {best_name}\n"
        f"Steering to θ={TARGET_THETA_DEG}°, φ={TARGET_PHI_DEG}° (Phase + Spacing)",
        fontsize=14,
    )
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "grid_phase_layout.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    names = list(all_results.keys())
    bar_colors = [colors.get(n, "gray") for n in names]

    steer_eff = [
        all_results[n]["metrics"].get("steering_efficiency", 0) * 100 for n in names
    ]
    axes[0, 0].bar(range(len(names)), steer_eff, color=bar_colors)
    axes[0, 0].set_xticks(range(len(names)))
    axes[0, 0].set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    axes[0, 0].set_ylabel("Steering Efficiency (%)", fontsize=11)
    axes[0, 0].set_title("Steering Efficiency", fontsize=12)
    axes[0, 0].set_ylim([0, 105])

    gains = [all_results[n]["metrics"].get("gain_at_target_dbi", 0) for n in names]
    axes[0, 1].bar(range(len(names)), gains, color=bar_colors)
    axes[0, 1].set_xticks(range(len(names)))
    axes[0, 1].set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    axes[0, 1].set_ylabel("Gain at Target (dBi)", fontsize=11)
    axes[0, 1].set_title(
        f"Gain at θ={TARGET_THETA_DEG}°, φ={TARGET_PHI_DEG}°", fontsize=12
    )

    times = [all_results[n]["elapsed_time"] for n in names]
    axes[1, 0].bar(range(len(names)), times, color=bar_colors)
    axes[1, 0].set_xticks(range(len(names)))
    axes[1, 0].set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    axes[1, 0].set_ylabel("Time (seconds)", fontsize=11)
    axes[1, 0].set_title("Computation Time", fontsize=12)
    axes[1, 0].set_yscale("log")

    theta_errors = [
        abs(all_results[n]["metrics"].get("peak_theta_deg", 0) - TARGET_THETA_DEG)
        for n in names
    ]
    phi_errors = [
        abs(all_results[n]["metrics"].get("peak_phi_deg", 0) - TARGET_PHI_DEG)
        for n in names
    ]
    x = np.arange(len(names))
    width = 0.35
    axes[1, 1].bar(
        x - width / 2,
        theta_errors,
        width,
        label="θ error",
        color=[c for c in bar_colors],
    )
    axes[1, 1].bar(
        x + width / 2,
        phi_errors,
        width,
        label="φ error",
        color=[c for c in bar_colors],
        alpha=0.6,
    )
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    axes[1, 1].set_ylabel("Pointing Error (degrees)", fontsize=11)
    axes[1, 1].set_title("Beam Pointing Error", fontsize=12)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "comparison_bars.png", dpi=150)
    plt.close()

    print(f"Saved plots to {RESULTS_DIR}")


def create_cst_model(gradient_result):
    positions = gradient_result["positions"]
    if isinstance(positions, torch.Tensor):
        positions = positions.detach().cpu().numpy()

    positions_list = [(float(p[0]), float(p[1])) for p in positions]

    weights = gradient_result["weights"]
    if isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()

    phases_deg = np.degrees(np.angle(weights)) if weights is not None else None

    try:
        cst_path = create_circular_array(
            project_name=f"case_study_3_grid_beamsteering_{ROWS}x{COLS}_phase_spacing",
            positions=positions_list,
            radii=PATCH_RADIUS_MM,
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
                        "rows": ROWS,
                        "cols": COLS,
                        "n_elements": N_ELEMENTS,
                        "freq_hz": FREQ_HZ,
                        "patch_radius_mm": PATCH_RADIUS_MM,
                        "spacing_mm": SPACING_MM,
                        "target_theta_deg": TARGET_THETA_DEG,
                        "target_phi_deg": TARGET_PHI_DEG,
                    },
                    "positions_mm": positions_list,
                    "phases_deg": (
                        phases_deg.tolist() if phases_deg is not None else None
                    ),
                },
                f,
                indent=2,
            )

    except Exception as e:
        print(f"  CST model creation failed: {e}")
        print("  (This is expected if CST is not installed)")


def print_summary_table(all_results):
    print("\n" + "=" * 100)
    print("SUMMARY COMPARISON TABLE")
    print("=" * 100)
    print(
        f"{'Optimizer':<30} {'Time (s)':<10} {'Steer Eff %':<12} {'Gain (dBi)':<12} {'θ Peak':<10} {'φ Peak':<10}"
    )
    print("-" * 100)

    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1]["metrics"].get("steering_efficiency", 0),
        reverse=True,
    )

    for name, res in sorted_results:
        m = res["metrics"]
        time_s = res["elapsed_time"]
        steer_eff = m.get("steering_efficiency", 0) * 100
        gain = m.get("gain_at_target_dbi", 0)
        theta_peak = m.get("peak_theta_deg", 0)
        phi_peak = m.get("peak_phi_deg", 0)

        print(
            f"{name:<30} {time_s:<10.3f} {steer_eff:<12.1f} {gain:<12.2f} {theta_peak:<10.1f} {phi_peak:<10.1f}"
        )

    print("=" * 100)
    print(f"Target Direction: θ={TARGET_THETA_DEG}°, φ={TARGET_PHI_DEG}°")

    best_steer = sorted_results[0]
    fastest = min(all_results.items(), key=lambda x: x[1]["elapsed_time"])

    print(
        f"\nBest Steering: {best_steer[0]} ({best_steer[1]['metrics'].get('steering_efficiency', 0)*100:.1f}%)"
    )
    print(f"Fastest Optimizer: {fastest[0]} ({fastest[1]['elapsed_time']:.3f}s)")


if __name__ == "__main__":
    main()
