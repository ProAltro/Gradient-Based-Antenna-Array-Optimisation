"""CST patch antenna array constructor with automatic feed positioning."""

import math
import os
import shutil
from typing import List, Tuple

from cst_python_api import CST_MicrowaveStudio as CST
from cst_python_api import Port

from .patches import C0, get_feed_position_circular, get_feed_position_rectangular

COPPER_MATERIAL = "Copper (annealed)"

DEFAULTS = {
    "substrate_thickness": 0.508,
    "dielectric_constant": 2.2,
    "loss_tangent": 0.0009,
    "copper_thickness": 0.035,
    "board_margin": 2.0,
    "freq_hz": 24.0e9,
    "target_impedance": 50.0,
}


def _calculate_board_size(
    positions: List[Tuple[float, float]],
    element_extents: List[float] | float,
    margin: float = DEFAULTS["board_margin"],
) -> float:
    """Calculate square board size from element positions and extents."""
    if isinstance(element_extents, (int, float)):
        element_extents = [element_extents] * len(positions)

    x_extent = max(abs(p[0]) + e for p, e in zip(positions, element_extents)) + margin
    y_extent = max(abs(p[1]) + e for p, e in zip(positions, element_extents)) + margin
    return 2 * max(x_extent, y_extent)


def _add_brick(cst, name, component, material, origin, size):
    """Create a brick from origin and size."""
    x0, y0, z0 = origin
    sx, sy, sz = size
    cst.Build.Shape.addBrick(
        name=name,
        component=component,
        material=material,
        xMin=float(min(x0, x0 + sx)),
        xMax=float(max(x0, x0 + sx)),
        yMin=float(min(y0, y0 + sy)),
        yMax=float(max(y0, y0 + sy)),
        zMin=float(min(z0, z0 + sz)),
        zMax=float(max(z0, z0 + sz)),
    )


def _add_cylinder(cst, name, component, material, x, y, z_min, z_max, radius):
    """Create a vertical cylinder at (x, y)."""
    cst.Build.Shape.addCylinder(
        xMin=float(x),
        yMin=float(y),
        zMin=float(z_min),
        xMax=0.0,
        yMax=0.0,
        zMax=float(z_max),
        extRad=float(radius),
        intRad=0.0,
        name=name,
        component=component,
        material=material,
        orientation="z",
        nSegments=0,
    )


def _get_cst_instance(folder, filename):
    """Return CST instance, rebuilding COM cache if needed."""
    try:
        return CST(folder, filename)
    except AttributeError:
        from win32com.client import gencache

        cache_dir = gencache.GetGeneratePath()
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)
        gencache.EnsureDispatch("CSTStudio.Application")
        return CST(folder, filename)


def _configure_solver(cst, freq_hz, freq_range_ghz=None):
    """Configure solver, boundaries, mesh, and field monitors."""
    fr_ghz = freq_hz / 1e9
    f_min = freq_range_ghz[0] if freq_range_ghz else max(10.0, fr_ghz - 4.0)
    f_max = freq_range_ghz[1] if freq_range_ghz else fr_ghz + 4.0

    cst.Solver.setFrequencyRange(fMin=f_min, fMax=f_max)
    cst.Solver.setBoundaryCondition(
        xMin="expanded open",
        xMax="expanded open",
        yMin="expanded open",
        yMax="expanded open",
        zMin="electric",
        zMax="expanded open",
    )

    padding = (C0 / freq_hz) * 1e3 * 0.25
    cst.Solver.setBackgroundLimits(
        xMin=padding,
        xMax=padding,
        yMin=padding,
        yMax=padding,
        zMin=padding,
        zMax=padding,
    )
    cst.Solver.changeSolverType("HF Time Domain")

    mesh_vba = """With Mesh
    .SetCreator "High Frequency"
    .LinesPerWavelength "10"
    .MinimumStepSize "20"
    .MergeThinPECLayerFixpoints "True"
    .RatioLimit "20"
    .AutomeshRefineAtPecLines "True", "6"
    .FPBAAvoidNonRegUnite "True"
    .ConsiderSpaceForLowerMeshLimit "False"
    .MinimumStepNumber "5"
    .AnisotropicCurvatureRefinement "True"
    .AnisotropicCurvatureRefinementFSM "True"
End With
With MeshSettings
    .SetMeshType "Hex"
    .Set "RatioLimitGeometry", "20"
    .Set "EdgeRefinementOn", "1"
    .Set "EdgeRefinementRatio", "6"
End With
With Solver
    .Method "Hexahedral"
    .CalculationType "TD-S"
    .StimulationPort "All"
    .StimulationMode "All"
    .SteadyStateLimit "-40"
    .MeshAdaption "False"
    .AutoNormImpedance "True"
    .NormingImpedance "50"
End With
PostProcess1D.ActivateOperation "vswr", "true"
PostProcess1D.ActivateOperation "yz-matrices", "true"
With FarfieldPlot
    .ClearCuts
    .AddCut "lateral", "0", "1"
    .AddCut "lateral", "90", "1"
    .AddCut "polar", "90", "1"
End With"""
    cst._CST_MicrowaveStudio__MWS.AddToHistory("configure mesh", mesh_vba)

    # Farfield monitors
    for freq in [f_min + i * (f_max - f_min) / 6 for i in range(7)]:
        cst.Solver.addFieldMonitor("Farfield", freq)
    cst.Solver.addFieldMonitor("Efield", fr_ghz)
    cst.Solver.addFieldMonitor("Hfield", fr_ghz)


def _add_port(
    cst,
    patch_name,
    feed_x,
    feed_y,
    substrate_h,
    copper_h,
    impedance,
    port_num,
    gap_radius: float = 0.1,
):
    """Add discrete port with gap holes at feed position.

    Args:
        gap_radius: Radius of the feed hole (should be proportional to patch size).
    """
    gap_r = gap_radius

    # Cut holes in patch and ground
    _add_cylinder(
        cst,
        f"gap_patch_{port_num}",
        "ports",
        "Vacuum",
        feed_x,
        feed_y,
        substrate_h,
        substrate_h + copper_h,
        gap_r,
    )
    _add_cylinder(
        cst,
        f"gap_gnd_{port_num}",
        "ports",
        "Vacuum",
        feed_x,
        feed_y,
        -copper_h,
        0.0,
        gap_r,
    )

    try:
        cst.Build.Boolean.subtract(
            f"radiator:{patch_name}", f"ports:gap_patch_{port_num}"
        )
        cst.Build.Boolean.subtract("ground:ground_plane", f"ports:gap_gnd_{port_num}")
    except Exception as e:
        print(f"Warning: Boolean subtract failed for port {port_num}: {e}")

    cst.Solver.Port.addDiscretePort(
        xMin=feed_x,
        xMax=feed_x,
        yMin=feed_y,
        yMax=feed_y,
        zMin=0.0,
        zMax=substrate_h,
        type="SParameter",
        impedance=impedance,
        radius=gap_r * 0.5,  # Port radius proportional to gap radius
    )


# =============================================================================
# Position generators
# =============================================================================


def grid_positions(rows: int, cols: int, spacing: float) -> List[Tuple[float, float]]:
    """Generate centered grid positions."""
    return [
        ((c - (cols - 1) / 2) * spacing, (r - (rows - 1) / 2) * spacing)
        for r in range(rows)
        for c in range(cols)
    ]


def spiral_positions(count: int, step: float) -> List[Tuple[float, float]]:
    """Generate Fermat spiral positions."""
    golden = math.pi * (3 - math.sqrt(5))
    return [
        (
            step * math.sqrt(i) * math.cos(i * golden),
            step * math.sqrt(i) * math.sin(i * golden),
        )
        for i in range(1, count + 1)
    ]


# =============================================================================
# Main constructor functions
# =============================================================================


def create_circular_array(
    project_name: str,
    positions: List[Tuple[float, float]],
    radii: List[float] | float = 2.2,
    freq_hz: float = DEFAULTS["freq_hz"],
    substrate_thickness: float = DEFAULTS["substrate_thickness"],
    dielectric_constant: float = DEFAULTS["dielectric_constant"],
    loss_tangent: float = DEFAULTS["loss_tangent"],
    copper_thickness: float = DEFAULTS["copper_thickness"],
    board_size: float | None = None,
    board_margin: float = DEFAULTS["board_margin"],
    target_impedance: float = DEFAULTS["target_impedance"],
) -> str:
    """Create circular patch array. Radii can be uniform or per-element list."""
    n = len(positions)
    radii_list = [radii] * n if isinstance(radii, (int, float)) else list(radii)

    # Calculate feed offsets for each element size
    feed_data = [
        get_feed_position_circular(
            r, substrate_thickness, dielectric_constant, target_impedance, loss_tangent
        )
        for r in radii_list
    ]

    # Auto-calculate board size if not provided
    if board_size is None:
        board_size = _calculate_board_size(positions, radii_list, board_margin)
        print(f"  Board size auto-calculated: {board_size:.1f} mm")

    # Setup paths
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sims_dir = os.path.join(base, "output", "sims", project_name)
    os.makedirs(sims_dir, exist_ok=True)

    # Create CST project
    cst = _get_cst_instance(sims_dir, f"{project_name}.cst")
    cst.Solver.Port = Port(cst._CST_MicrowaveStudio__MWS)
    cst.Project.setUnits(leng="mm", freq="GHz", time="ns", temp="degC")

    # Material
    cst.Build.Material.addNormalMaterial(
        "Rogers_RO5880",
        eps=dielectric_constant,
        mu=1.0,
        colour=[0.54, 0.65, 0.51],
        tanD=loss_tangent,
    )

    # Substrate & ground
    half = board_size / 2
    _add_brick(
        cst,
        "substrate",
        "dielectric",
        "Rogers_RO5880",
        [-half, -half, 0],
        [board_size, board_size, substrate_thickness],
    )
    _add_brick(
        cst,
        "ground_plane",
        "ground",
        COPPER_MATERIAL,
        [-half, -half, -copper_thickness],
        [board_size, board_size, copper_thickness],
    )

    # Patches and ports
    z_bot, z_top = substrate_thickness, substrate_thickness + copper_thickness
    for i, ((x, y), r, (feed_offset, info)) in enumerate(
        zip(positions, radii_list, feed_data)
    ):
        name = f"patch_{i:02d}"
        _add_cylinder(cst, name, "radiator", COPPER_MATERIAL, x, y, z_bot, z_top, r)
        feed_x, feed_y = x, y - feed_offset
        # Gap radius proportional to patch radius (~4% of radius)
        gap_radius = r * 0.04
        _add_port(
            cst,
            name,
            feed_x,
            feed_y,
            substrate_thickness,
            copper_thickness,
            target_impedance,
            i + 1,
            gap_radius=gap_radius,
        )

    _configure_solver(cst, freq_hz)
    cst.saveFile()

    r_min, r_max = min(radii_list), max(radii_list)
    r_str = f"r={r_min}mm" if r_min == r_max else f"r={r_min}-{r_max}mm"
    print(f"\nCreated: {sims_dir}/{project_name}.cst")
    print(f"  Patches: {len(positions)} circular ({r_str})")
    print(
        f"  Edge R range: {min(d[1]['resistance'] for d in feed_data):.1f}-{max(d[1]['resistance'] for d in feed_data):.1f}Ω"
    )

    return os.path.join(sims_dir, f"{project_name}.cst")


def create_rectangular_array(
    project_name: str,
    positions: List[Tuple[float, float]],
    dimensions: List[Tuple[float, float]] | Tuple[float, float] = (4.0, 5.0),
    freq_hz: float = DEFAULTS["freq_hz"],
    substrate_thickness: float = DEFAULTS["substrate_thickness"],
    dielectric_constant: float = DEFAULTS["dielectric_constant"],
    loss_tangent: float = DEFAULTS["loss_tangent"],
    copper_thickness: float = DEFAULTS["copper_thickness"],
    board_size: float | None = None,
    board_margin: float = DEFAULTS["board_margin"],
    target_impedance: float = DEFAULTS["target_impedance"],
    use_inset_feed: bool = False,
) -> str:
    """Create rectangular patch array. Dimensions (length, width) can be uniform or per-element."""
    n = len(positions)
    if (
        isinstance(dimensions, tuple)
        and len(dimensions) == 2
        and isinstance(dimensions[0], (int, float))
    ):
        dims_list = [dimensions] * n
    else:
        dims_list = list(dimensions)

    # Calculate feed offsets for each element size
    feed_data = [
        get_feed_position_rectangular(
            l,
            w,
            substrate_thickness,
            dielectric_constant,
            freq_hz,
            target_impedance,
            use_inset_feed,
        )
        for l, w in dims_list
    ]

    # Auto-calculate board size if not provided
    if board_size is None:
        element_extents = [max(l / 2, w / 2) for l, w in dims_list]
        board_size = _calculate_board_size(positions, element_extents, board_margin)
        print(f"  Board size auto-calculated: {board_size:.1f} mm")

    # Setup paths
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sims_dir = os.path.join(base, "output", "sims", project_name)
    os.makedirs(sims_dir, exist_ok=True)

    # Create CST project
    cst = _get_cst_instance(sims_dir, f"{project_name}.cst")
    cst.Solver.Port = Port(cst._CST_MicrowaveStudio__MWS)
    cst.Project.setUnits(leng="mm", freq="GHz", time="ns", temp="degC")

    # Material
    cst.Build.Material.addNormalMaterial(
        "Rogers_RO5880",
        eps=dielectric_constant,
        mu=1.0,
        colour=[0.54, 0.65, 0.51],
        tanD=loss_tangent,
    )

    # Substrate & ground
    half = board_size / 2
    _add_brick(
        cst,
        "substrate",
        "dielectric",
        "Rogers_RO5880",
        [-half, -half, 0],
        [board_size, board_size, substrate_thickness],
    )
    _add_brick(
        cst,
        "ground_plane",
        "ground",
        COPPER_MATERIAL,
        [-half, -half, -copper_thickness],
        [board_size, board_size, copper_thickness],
    )

    # Patches and ports
    z_bot = substrate_thickness
    for i, ((x, y), (length, width), (feed_offset, info)) in enumerate(
        zip(positions, dims_list, feed_data)
    ):
        name = f"patch_{i:02d}"
        half_l, half_w = length / 2, width / 2
        _add_brick(
            cst,
            name,
            "radiator",
            COPPER_MATERIAL,
            [x - half_w, y - half_l, z_bot],
            [width, length, copper_thickness],
        )
        feed_x, feed_y = x, y - half_l + feed_offset
        # Gap radius proportional to smallest patch dimension (~4%)
        gap_radius = min(length, width) * 0.04
        _add_port(
            cst,
            name,
            feed_x,
            feed_y,
            substrate_thickness,
            copper_thickness,
            target_impedance,
            i + 1,
            gap_radius=gap_radius,
        )

    _configure_solver(cst, freq_hz)
    cst.saveFile()

    lengths = [d[0] for d in dims_list]
    widths = [d[1] for d in dims_list]
    dim_str = (
        f"{lengths[0]}x{widths[0]}mm" if len(set(dims_list)) == 1 else "variable sizes"
    )
    print(f"\nCreated: {sims_dir}/{project_name}.cst")
    print(f"  Patches: {len(positions)} rectangular ({dim_str})")
    print(
        f"  Edge R range: {min(d[1]['resistance'] for d in feed_data):.1f}-{max(d[1]['resistance'] for d in feed_data):.1f}Ω"
    )

    return os.path.join(sims_dir, f"{project_name}.cst")
