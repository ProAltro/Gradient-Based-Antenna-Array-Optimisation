# Gradient-Based Antenna Array Optimisation

Differentiable physics approach to antenna array optimization using automatic differentiation.

## Quick Start

Run any case study:
```bash
python case_study_1_spiral_wpt.py
python case_study_2_hexagonal_gain.py
python case_study_3_grid_beamsteering.py
```

Compare optimizers:
```bash
python compare.py
```

## Project Structure

```
├── case_study_1_spiral_wpt.py      # Spiral array for wireless power transfer
├── case_study_2_hexagonal_gain.py  # Hexagonal array directivity maximization
├── case_study_3_grid_beamsteering.py # Grid array beam steering
├── compare.py                       # Optimizer comparison script
│
└── src/
    ├── cst_constructors/           # CST model construction
    │   ├── cst_patch_constructor.py
    │   └── patches.py
    │
    ├── dl_optimisers/              # Optimization algorithms
    │   ├── differential_evolution.py
    │   ├── genetic.py
    │   ├── gradient.py
    │   └── particle_swarm.py
    │
    ├── param_arrays/               # Parameterized array geometries
    │   ├── base.py
    │   ├── circular.py
    │   ├── cross.py
    │   ├── grid.py
    │   ├── hexagonal.py
    │   ├── linear.py
    │   ├── random_array.py
    │   └── spiral.py
    │
    └── sim_helpers/                # Simulation utilities
        ├── arrays.py
        ├── patches.py
        ├── results.py
        └── storage.py
```

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
