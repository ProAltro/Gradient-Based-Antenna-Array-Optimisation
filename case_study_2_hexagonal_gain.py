"""
Case Study 2: Hexagonal Array for High-Gain Applications (mmWave)
===================================================================

Problem: Maximize directivity/gain for high-gain antenna applications
Array: 19-element hexagonal lattice (2 rings around center)
Frequency: 28 GHz (5G mmWave band)
Optimizable: Position offsets ONLY (fixed base hexagonal pattern)

Compares:
- Gradient-based: Adam
- Evolutionary: Genetic Algorithm, PSO, Differential Evolution
"""

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

from src.param_arrays.hexagonal import HexagonalParamArray, HexagonalParamArrayConfig
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
    directivity_objective,
)
from src.sim_helpers.arrays import evaluate_array_torch, C0
from src.sim_helpers.results import (
    find_peak_direction_torch,
    directivity_torch,
)
from src.cst_constructors.cst_patch_constructor import create_circular_array

RESULTS_DIR = Path("results/case_study_2_hexagonal_gain")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Array Configuration - 19 elements (2 rings hexagonal)
NUM_RINGS = 2  # 1 + 6 + 12 = 19 elements
N_ELEMENTS = 1 + 3 * NUM_RINGS * (NUM_RINGS + 1)  # = 19
FREQ_HZ = 28e9  # 28 GHz (5G mmWave band)
WAVELENGTH_MM = C0 / FREQ_HZ * 1000  # ~10.7 mm
SPACING_MM = WAVELENGTH_MM * 0.5  # Half-wavelength nominal spacing

# Circular patch radius for mmWave (~0.3 * lambda)
PATCH_RADIUS_MM = WAVELENGTH_MM * 0.3

# Substrate thickness for mmWave (~0.025 * lambda, Rogers RT5880)
SUBSTRATE_THICKNESS_MM = 0.254

MAX_ITERS = 150
DEVICE = auto_device()

print(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")


# =============================================================================
# Setup Functions
# =============================================================================


def create_param_array():
    """Create hexagonal array for directivity optimization (position-only)."""
    min_spacing = max(SPACING_MM * 0.4, 2 * PATCH_RADIUS_MM + 1.0)
    hex_config = HexagonalParamArrayConfig(
        freq_hz=FREQ_HZ,
        position_unit="mm",
        element_type="circ_patch",
        element_dims=(PATCH_RADIUS_MM,),  # Circular patch radius
        num_rings=NUM_RINGS,  # 19 elements
        spacing_bounds_mm=(SPACING_MM * 0.8, SPACING_MM * 1.3),
        rotation_bounds=(0.0, 2 * math.pi),
        position_offset_bounds_mm=(-SPACING_MM * 0.15, SPACING_MM * 0.15),
        optimize_spacing=True,
        optimize_rotation=True,
        optimize_position_offsets=True,  # Per-element x,y offsets
        optimize_amplitudes=False,  # No amplitude optimization
        optimize_phases=False,  # No phase optimization
        min_spacing_mm=min_spacing,
        penalty_weight=100.0,
    )
    return HexagonalParamArray(config=hex_config)


def run_optimizer(name, optimizer, param_array):
    """Run optimizer and return results with timing."""
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
    print(f"  Directivity: {m.get('directivity_dbi', 0):.2f} dBi")
    print(
        f"  Peak at: θ={m.get('peak_theta_deg', 0):.1f}°, φ={m.get('peak_phi_deg', 0):.1f}°"
    )
    print(f"  Objective: {m.get('objective', 0):.4f}")

    return result


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("Case Study 2: Hexagonal Array for High-Gain Applications")
    print("=" * 70)

    pa_test = create_param_array()
    print(f"Array: {N_ELEMENTS}-element Hexagonal ({NUM_RINGS} rings)")
    print(f"Frequency: {FREQ_HZ/1e9:.1f} GHz (ISM band)")
    print(f"Wavelength: {WAVELENGTH_MM:.2f} mm")
    print(f"Nominal Spacing: {SPACING_MM:.2f} mm (λ/2)")
    print(f"Optimizable params: {pa_test.num_params}")
    print(f"  - Spacing, rotation (2 params)")
    print(f"  - Per-element x,y offsets ({2*N_ELEMENTS} = {2*N_ELEMENTS} params)")
    print(f"  - Total: {pa_test.num_params} position parameters")
    print(f"Results: {RESULTS_DIR}")

    all_results = {}

    # Gradient-based optimizer (Adam)
    pa_grad = create_param_array()
    grad_config = GradientConfig(
        max_iters=MAX_ITERS,
        learning_rate=0.01,
        optimizer="adam",
        weight_directivity=1.0,
        weight_cone_power=0.0,
        weight_sll=0.0,
        device=DEVICE,
    )
    grad_opt = GradientOptimizer(pa_grad, grad_config)
    all_results["Gradient (Adam)"] = run_optimizer("Gradient (Adam)", grad_opt, pa_grad)

    # Genetic Algorithm
    pa_ga = create_param_array()
    ga_config = GeneticConfig(
        population_size=50,
        max_generations=MAX_ITERS,
        objective_fn=directivity_objective,
        device=DEVICE,
    )
    ga_opt = GeneticOptimizer(pa_ga, ga_config)
    all_results["Genetic Algorithm"] = run_optimizer("Genetic Algorithm", ga_opt, pa_ga)

    # PSO
    pa_pso = create_param_array()
    pso_config = ParticleSwarmConfig(
        swarm_size=50,
        max_iterations=MAX_ITERS,
        objective_fn=directivity_objective,
        device=DEVICE,
    )
    pso_opt = ParticleSwarmOptimizer(pa_pso, pso_config)
    all_results["PSO"] = run_optimizer("PSO", pso_opt, pa_pso)

    # Differential Evolution
    pa_de = create_param_array()
    de_config = DifferentialEvolutionConfig(
        population_size=50,
        max_generations=MAX_ITERS,
        objective_fn=directivity_objective,
        device=DEVICE,
    )
    de_opt = DifferentialEvolutionOptimizer(pa_de, de_config)
    all_results["Differential Evolution"] = run_optimizer(
        "Differential Evolution", de_opt, pa_de
    )

    # =========================================================================
    # Save Results & Generate Plots
    # =========================================================================

    save_results(all_results)
    generate_plots(all_results)

    # =========================================================================
    # Create CST Model for Best Gradient Result
    # =========================================================================

    print("\n" + "=" * 70)
    print("Creating CST Model from Gradient-Optimized Array")
    print("=" * 70)

    create_cst_model(all_results["Gradient (Adam)"])

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)

    # Print summary table
    print_summary_table(all_results)


def save_results(all_results):
    """Save all results to files with comprehensive metrics."""

    summary_data = []
    for name, res in all_results.items():
        m = res["metrics"]
        summary_data.append(
            {
                "Optimizer": name,
                "Objective": m.get("objective", 0),
                "Directivity_dBi": m.get("directivity_dbi", 0),
                "Peak_Theta_deg": m.get("peak_theta_deg", 0),
                "Peak_Phi_deg": m.get("peak_phi_deg", 0),
                "Num_Elements": N_ELEMENTS,
                "Num_Params": res["alpha"].numel() if res["alpha"] is not None else 0,
                "Time_s": res["elapsed_time"],
                "Iterations": res["num_iterations"],
                "Evaluations": res["num_evaluations"],
            }
        )

    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(RESULTS_DIR / "summary_results.csv", index=False)
    print(f"\nSaved: {RESULTS_DIR / 'summary_results.csv'}")

    # Detailed parameters for each optimizer
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

        params = {
            "optimizer": name,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "n_elements": N_ELEMENTS,
                "num_rings": NUM_RINGS,
                "freq_hz": FREQ_HZ,
                "wavelength_mm": WAVELENGTH_MM,
                "spacing_mm": SPACING_MM,
                "max_iterations": MAX_ITERS,
            },
            "results": {
                "alpha": to_serializable(res["alpha"]),
                "positions_mm": to_serializable(res["positions"]),
                "weights": to_serializable(res["weights"]),
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

        # Save convergence history
        history = res.get("history", [])
        if history:
            df_hist = pd.DataFrame(history)
            df_hist.to_csv(RESULTS_DIR / f"history_{safe_name}.csv", index=False)

    print(f"Saved parameter files and histories to {RESULTS_DIR}")


def generate_plots(all_results):
    """Generate comparison plots."""

    best_name = max(
        all_results.keys(),
        key=lambda k: all_results[k]["metrics"].get("directivity_dbi", 0),
    )
    best_res = all_results[best_name]

    # Color scheme for different optimizer categories
    colors = {
        "Gradient (Adam)": "#2196F3",  # Blue - DL method
        "Genetic Algorithm": "#4CAF50",  # Green - Evolutionary
        "PSO": "#FF9800",  # Orange - Evolutionary
        "Differential Evolution": "#F44336",  # Red - Evolutionary
    }

    # --- 1. Convergence Comparison ---
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
        f"Convergence Comparison - Hexagonal Array ({N_ELEMENTS} elements)", fontsize=14
    )
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "convergence_comparison.png", dpi=150)
    plt.close()

    # --- 2. Time vs Performance Scatter ---
    fig, ax = plt.subplots(figsize=(10, 8))
    for name, res in all_results.items():
        time_s = res["elapsed_time"]
        dir_dbi = res["metrics"].get("directivity_dbi", 0)
        ax.scatter(
            time_s,
            dir_dbi,
            s=200,
            c=colors.get(name, "gray"),
            label=name,
            edgecolors="black",
            linewidth=1.5,
        )
        ax.annotate(
            name.split()[0],
            (time_s, dir_dbi),
            fontsize=8,
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax.set_xlabel("Computation Time (seconds)", fontsize=12)
    ax.set_ylabel("Directivity (dBi)", fontsize=12)
    ax.set_title("Performance vs Computation Time", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "time_vs_performance.png", dpi=150)
    plt.close()

    # --- 3. Best Radiation Pattern (2D Heatmap) ---
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

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Normalized Power (dB)", fontsize=12)
    ax.set_xlabel("Phi (degrees)", fontsize=12)
    ax.set_ylabel("Theta (degrees)", fontsize=12)
    ax.set_title(
        f"Radiation Pattern - {best_name}\n"
        f"Directivity: {best_res['metrics'].get('directivity_dbi', 0):.2f} dBi",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "pattern_2d_heatmap.png", dpi=150)
    plt.close()

    # --- 4. Hexagonal Array Layout ---
    fig, ax = plt.subplots(figsize=(10, 10))
    positions = best_res["positions"]
    if isinstance(positions, torch.Tensor):
        positions = positions.detach().cpu().numpy()

    # Plot optimized positions
    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        s=300,
        c="#2196F3",
        edgecolors="black",
        linewidth=2,
        label="Optimized",
        zorder=3,
    )

    # Number elements
    for i, (x, y) in enumerate(positions):
        ax.annotate(
            str(i),
            (x, y),
            fontsize=10,
            ha="center",
            va="center",
            fontweight="bold",
            color="white",
        )

    # Draw hexagonal grid reference (nominal positions)
    pa_ref = create_param_array()
    ref_alpha = pa_ref.default_init()
    with torch.no_grad():
        ref_decoded = pa_ref.decode(ref_alpha)
        ref_positions = ref_decoded["positions"].numpy()

    ax.scatter(
        ref_positions[:, 0],
        ref_positions[:, 1],
        s=150,
        c="none",
        edgecolors="gray",
        linewidth=2,
        linestyle="--",
        label="Nominal (hex)",
        zorder=2,
    )

    # Draw displacement vectors
    for opt_pos, ref_pos in zip(positions, ref_positions):
        ax.arrow(
            ref_pos[0],
            ref_pos[1],
            opt_pos[0] - ref_pos[0],
            opt_pos[1] - ref_pos[1],
            head_width=1,
            head_length=0.5,
            fc="red",
            ec="red",
            alpha=0.5,
        )

    ax.set_xlabel("X (mm)", fontsize=12)
    ax.set_ylabel("Y (mm)", fontsize=12)
    ax.set_title(f"Optimized Hexagonal Array Layout - {best_name}", fontsize=14)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "hexagonal_layout.png", dpi=150)
    plt.close()

    # --- 5. Comparison Bar Charts ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    names = list(all_results.keys())
    bar_colors = [colors.get(n, "gray") for n in names]

    # Directivity
    dirs = [all_results[n]["metrics"].get("directivity_dbi", 0) for n in names]
    axes[0, 0].bar(range(len(names)), dirs, color=bar_colors)
    axes[0, 0].set_xticks(range(len(names)))
    axes[0, 0].set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    axes[0, 0].set_ylabel("Directivity (dBi)", fontsize=11)
    axes[0, 0].set_title("Boresight Directivity", fontsize=12)

    # Peak Theta
    thetas = [all_results[n]["metrics"].get("peak_theta_deg", 0) for n in names]
    axes[0, 1].bar(range(len(names)), thetas, color=bar_colors)
    axes[0, 1].set_xticks(range(len(names)))
    axes[0, 1].set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    axes[0, 1].set_ylabel("Peak Theta (deg)", fontsize=11)
    axes[0, 1].set_title("Peak Direction (Theta)", fontsize=12)

    # Computation Time (log scale)
    times = [all_results[n]["elapsed_time"] for n in names]
    axes[1, 0].bar(range(len(names)), times, color=bar_colors)
    axes[1, 0].set_xticks(range(len(names)))
    axes[1, 0].set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    axes[1, 0].set_ylabel("Time (seconds)", fontsize=11)
    axes[1, 0].set_title("Computation Time", fontsize=12)
    axes[1, 0].set_yscale("log")

    # Objective Value
    objs = [all_results[n]["metrics"].get("objective", 0) for n in names]
    axes[1, 1].bar(range(len(names)), objs, color=bar_colors)
    axes[1, 1].set_xticks(range(len(names)))
    axes[1, 1].set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    axes[1, 1].set_ylabel("Objective Value", fontsize=11)
    axes[1, 1].set_title("Final Objective (Directivity)", fontsize=12)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "comparison_bars.png", dpi=150)
    plt.close()

    print(f"Saved plots to {RESULTS_DIR}")


def create_cst_model(gradient_result):
    """Create CST model from gradient-optimized array."""

    positions = gradient_result["positions"]
    if isinstance(positions, torch.Tensor):
        positions = positions.detach().cpu().numpy()

    # Convert to list of tuples for CST constructor
    positions_list = [(float(p[0]), float(p[1])) for p in positions]

    try:
        cst_path = create_circular_array(
            project_name=f"case_study_2_hexagonal_gain_{N_ELEMENTS}elem",
            positions=positions_list,
            radii=PATCH_RADIUS_MM,
            freq_hz=FREQ_HZ,
            substrate_thickness=SUBSTRATE_THICKNESS_MM,  # mmWave substrate
            dielectric_constant=2.2,
            loss_tangent=0.0009,
            copper_thickness=0.017,  # Thinner copper for mmWave
            board_margin=2.0,  # Smaller margin for compact mmWave
            target_impedance=50.0,
        )
        print(f"CST model created: {cst_path}")

        # Save CST model path to results
        with open(RESULTS_DIR / "cst_model_info.json", "w") as f:
            json.dump(
                {
                    "cst_path": cst_path,
                    "array_config": {
                        "n_elements": N_ELEMENTS,
                        "num_rings": NUM_RINGS,
                        "freq_hz": FREQ_HZ,
                        "patch_radius_mm": PATCH_RADIUS_MM,
                    },
                    "positions_mm": positions_list,
                },
                f,
                indent=2,
            )

    except Exception as e:
        print(f"  CST model creation failed: {e}")
        print("  (This is expected if CST is not installed)")


def print_summary_table(all_results):
    """Print a summary comparison table."""

    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Optimizer':<30} {'Time (s)':<10} {'Directivity':<12} {'Objective':<12}")
    print("-" * 80)

    # Sort by directivity
    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1]["metrics"].get("directivity_dbi", 0),
        reverse=True,
    )

    for name, res in sorted_results:
        m = res["metrics"]
        time_s = res["elapsed_time"]
        dir_dbi = m.get("directivity_dbi", 0)
        obj = m.get("objective", 0)

        print(f"{name:<30} {time_s:<10.3f} {dir_dbi:<12.2f} {obj:<12.4f}")

    print("=" * 80)

    # Highlight key findings
    best_dir = sorted_results[0]
    fastest = min(all_results.items(), key=lambda x: x[1]["elapsed_time"])

    print(
        f"\nBest Directivity: {best_dir[0]} ({best_dir[1]['metrics'].get('directivity_dbi', 0):.2f} dBi)"
    )
    print(f"Fastest Optimizer: {fastest[0]} ({fastest[1]['elapsed_time']:.3f}s)")


if __name__ == "__main__":
    main()
