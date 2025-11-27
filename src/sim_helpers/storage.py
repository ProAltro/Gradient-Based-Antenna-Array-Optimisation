"""
Storage utilities for saving antenna array configurations and results.

Creates organized folder structures for each antenna design with:
- CSV files for element parameters (positions, dimensions, amplitudes, etc.)
- Result plots
- CST files (created separately)
"""

import os
import csv
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional
import numpy as np


def get_output_dir() -> str:
    """Get the base output directory."""
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base, "output")


def create_antenna_folder(name: str) -> str:
    """
    Create a folder for an antenna design in the output directory.

    Parameters
    ----------
    name : str
        Name of the antenna design

    Returns
    -------
    str
        Path to the created folder
    """
    folder = os.path.join(get_output_dir(), name)
    os.makedirs(folder, exist_ok=True)
    return folder


def save_element_positions(
    folder: str,
    positions: np.ndarray,
    filename: str = "element_positions.csv",
) -> str:
    """
    Save element positions to CSV.

    Parameters
    ----------
    folder : str
        Output folder path
    positions : np.ndarray
        Element positions, shape (n_elements, 2)
    filename : str
        CSV filename

    Returns
    -------
    str
        Path to saved file
    """
    filepath = os.path.join(folder, filename)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["element_id", "x_mm", "y_mm"])
        for i, (x, y) in enumerate(positions):
            writer.writerow([i, float(x), float(y)])
    return filepath


def save_element_dimensions(
    folder: str,
    dimensions: np.ndarray,
    element_type: str = "rectangular",
    filename: str = "element_dimensions.csv",
) -> str:
    """
    Save element dimensions to CSV.

    Parameters
    ----------
    folder : str
        Output folder path
    dimensions : np.ndarray
        For rectangular: shape (n_elements, 2) with (length, width)
        For circular: shape (n_elements,) with radii
    element_type : str
        'rectangular' or 'circular'
    filename : str
        CSV filename

    Returns
    -------
    str
        Path to saved file
    """
    filepath = os.path.join(folder, filename)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        if element_type == "rectangular":
            writer.writerow(["element_id", "length_mm", "width_mm"])
            for i, (l, w) in enumerate(dimensions):
                writer.writerow([i, float(l), float(w)])
        else:  # circular
            writer.writerow(["element_id", "radius_mm"])
            for i, r in enumerate(dimensions):
                writer.writerow([i, float(r)])
    return filepath


def save_element_weights(
    folder: str,
    amplitudes: Optional[np.ndarray] = None,
    phases: Optional[np.ndarray] = None,
    filename: str = "element_weights.csv",
) -> str:
    """
    Save element excitation weights (amplitudes and phases) to CSV.

    Parameters
    ----------
    folder : str
        Output folder path
    amplitudes : np.ndarray, optional
        Amplitude weights, shape (n_elements,)
    phases : np.ndarray, optional
        Phase weights in radians, shape (n_elements,)
    filename : str
        CSV filename

    Returns
    -------
    str
        Path to saved file
    """
    filepath = os.path.join(folder, filename)

    # Determine number of elements
    if amplitudes is not None:
        n_elements = len(amplitudes)
    elif phases is not None:
        n_elements = len(phases)
    else:
        return ""  # Nothing to save

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["element_id", "amplitude", "phase_rad", "phase_deg"])
        for i in range(n_elements):
            amp = float(amplitudes[i]) if amplitudes is not None else 1.0
            phase_rad = float(phases[i]) if phases is not None else 0.0
            phase_deg = np.degrees(phase_rad)
            writer.writerow([i, amp, phase_rad, phase_deg])
    return filepath


def save_design_parameters(
    folder: str,
    params: Dict[str, Any],
    filename: str = "design_parameters.json",
) -> str:
    """
    Save design parameters to JSON.

    Parameters
    ----------
    folder : str
        Output folder path
    params : dict
        Design parameters dictionary
    filename : str
        JSON filename

    Returns
    -------
    str
        Path to saved file
    """
    filepath = os.path.join(folder, filename)

    # Convert numpy arrays to lists for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj

    with open(filepath, "w") as f:
        json.dump(convert(params), f, indent=2)
    return filepath


def save_optimization_history(
    folder: str,
    history: List[Dict[str, float]],
    filename: str = "optimization_history.csv",
) -> str:
    """
    Save optimization history to CSV.

    Parameters
    ----------
    folder : str
        Output folder path
    history : list
        List of metric dictionaries from optimizer
    filename : str
        CSV filename

    Returns
    -------
    str
        Path to saved file
    """
    if not history:
        return ""

    filepath = os.path.join(folder, filename)
    keys = list(history[0].keys())

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    k: float(v) if isinstance(v, (int, float)) else v
                    for k, v in row.items()
                }
            )
    return filepath


def save_broadband_results(
    folder: str,
    frequencies: np.ndarray,
    s11_db: np.ndarray,
    realized_gain_dbi: np.ndarray,
    directivity_dbi: np.ndarray,
    mismatch_efficiency: np.ndarray,
    filename: str = "broadband_results.csv",
) -> str:
    """
    Save broadband analysis results to CSV.

    Parameters
    ----------
    folder : str
        Output folder path
    frequencies : np.ndarray
        Frequency array in Hz
    s11_db : np.ndarray
        S11 in dB
    realized_gain_dbi : np.ndarray
        Realized gain in dBi
    directivity_dbi : np.ndarray
        Directivity in dBi
    mismatch_efficiency : np.ndarray
        Mismatch efficiency (linear)
    filename : str
        CSV filename

    Returns
    -------
    str
        Path to saved file
    """
    filepath = os.path.join(folder, filename)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frequency_hz",
                "frequency_ghz",
                "s11_db",
                "realized_gain_dbi",
                "directivity_dbi",
                "mismatch_efficiency",
            ]
        )
        for i in range(len(frequencies)):
            writer.writerow(
                [
                    float(frequencies[i]),
                    float(frequencies[i] / 1e9),
                    float(s11_db[i]),
                    float(realized_gain_dbi[i]),
                    float(directivity_dbi[i]),
                    float(mismatch_efficiency[i]),
                ]
            )
    return filepath


@dataclass
class AntennaConfig:
    """Complete antenna configuration for storage and reconstruction."""

    name: str
    element_type: str  # 'rectangular' or 'circular'
    n_elements: int
    freq_hz: float
    substrate_thickness_mm: float
    dielectric_constant: float
    loss_tangent: float
    copper_thickness_mm: float
    positions_mm: List[Tuple[float, float]]
    dimensions_mm: List[Tuple[float, ...]]  # (L, W) for rect, (R,) for circular
    amplitudes: Optional[List[float]] = None
    phases_rad: Optional[List[float]] = None

    def save(self, folder: str) -> Dict[str, str]:
        """
        Save complete antenna configuration to folder.

        Returns dict of saved file paths.
        """
        saved_files = {}

        # Positions
        positions = np.array(self.positions_mm)
        saved_files["positions"] = save_element_positions(folder, positions)

        # Dimensions
        dimensions = np.array(self.dimensions_mm)
        saved_files["dimensions"] = save_element_dimensions(
            folder, dimensions, self.element_type
        )

        # Weights
        if self.amplitudes is not None or self.phases_rad is not None:
            amps = np.array(self.amplitudes) if self.amplitudes else None
            phases = np.array(self.phases_rad) if self.phases_rad else None
            saved_files["weights"] = save_element_weights(folder, amps, phases)

        # Design parameters
        params = {
            "name": self.name,
            "element_type": self.element_type,
            "n_elements": self.n_elements,
            "freq_hz": self.freq_hz,
            "freq_ghz": self.freq_hz / 1e9,
            "substrate_thickness_mm": self.substrate_thickness_mm,
            "dielectric_constant": self.dielectric_constant,
            "loss_tangent": self.loss_tangent,
            "copper_thickness_mm": self.copper_thickness_mm,
        }
        saved_files["parameters"] = save_design_parameters(folder, params)

        return saved_files


__all__ = [
    "get_output_dir",
    "create_antenna_folder",
    "save_element_positions",
    "save_element_dimensions",
    "save_element_weights",
    "save_design_parameters",
    "save_optimization_history",
    "save_broadband_results",
    "AntennaConfig",
]
