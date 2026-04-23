"""
Multi-Objective Smooth AnalytICal Placer (MOSAIC Placer)

Team Members: Arnav Patil
              Alexandre Singer

Optimizes a smooth surrogate of the challenge proxy cost:
    PC = (1.0 * WL) + (0.5 * Cong) + (0.5 * Dens)

This version optimizes hard and soft macros.

Usage:
    uv run evaluate submissions/mosaic/placer.py
    uv run evaluate submissions/mosaic/placer.py --all
    uv run evaluate submissions/mosaic/placer.py -b ibm03
"""

import math
import os
import random
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from macro_place.benchmark import Benchmark

# Define proxy weights for the optimization objectives
PROXY_WEIGHTS = {
    "wirelength": 1.0,  # Weight for wirelength cost
    "congestion": 0.5,  # Weight for congestion cost
    "density": 0.5,  # Weight for density cost
}

# Define directories for NanGate45 benchmarks
NG45_DIRS = {
    "ariane133": "Flows/NanGate45/ariane133/netlist/output_CT_Grouping",
    "ariane136": "Flows/NanGate45/ariane136/netlist/output_CT_Grouping",
    "mempool_tile": "Flows/NanGate45/mempool_tile/netlist/output_CT_Grouping",
    "nvdla": "Flows/NanGate45/nvdla/netlist/output_CT_Grouping",
}

# Function to determine candidate external roots for placement cost evaluation
def _candidate_external_roots() -> list[Path]:
    roots: list[Path] = []

    # Check if an environment variable is set for external root
    env_root = os.environ.get("MACROPLACE_EXTERNAL_ROOT")
    if env_root:
        roots.append(Path(env_root))

    # Default to the MacroPlacement directory
    roots.append(Path("external/MacroPlacement"))
    return roots

# Function to load the PlacementCost class from the external root
def _load_placement_cost_class(external_root: Path):
    plc_dir = external_root / "CodeElements" / "Plc_client"
    if str(plc_dir) not in sys.path:
        sys.path.insert(0, str(plc_dir))

    from plc_client_os import PlacementCost

    return PlacementCost

# Function to load placement cost data for a given benchmark
def _load_plc(benchmark_name: str):
    for external_root in _candidate_external_roots():
        testcase_root = external_root / "Testcases" / "ICCAD04" / benchmark_name
        
        if testcase_root.exists():
            netlist = testcase_root / "netlist.pb.txt"
            plc_file = testcase_root / "initial.plc"
            PlacementCost = _load_placement_cost_class(external_root)
            plc = PlacementCost(str(netlist))
            
            if plc_file.exists():
                plc.restore_placement(str(plc_file), ifInital=True, ifReadComment=True)
                
            return plc

        ng45_rel = NG45_DIRS.get(benchmark_name)
        
        if ng45_rel:
            ng45_root = external_root / ng45_rel
            netlist = ng45_root / "netlist.pb.txt"
            plc_file = ng45_root / "initial.plc"
            if netlist.exists():
                PlacementCost = _load_placement_cost_class(external_root)
                plc = PlacementCost(str(netlist))
                
                if plc_file.exists():
                    plc.restore_placement(str(plc_file), ifInital=True, ifReadComment=True)
                    
                return plc

    return None

# Ensure congestion arrays are initialized properly
def _ensure_congestion_arrays(plc) -> None:
    expected_size = plc.grid_col * plc.grid_row

    if len(plc.H_routing_cong) != expected_size:
        plc.V_routing_cong = [0] * expected_size
        plc.H_routing_cong = [0] * expected_size
        plc.V_macro_routing_cong = [0] * expected_size
        plc.H_macro_routing_cong = [0] * expected_size

# Set placement positions in the PlacementCost object
def _set_plc_placement(plc, placement: torch.Tensor, benchmark: Benchmark) -> None:
    placement_np = placement.detach().cpu().numpy()

    if not hasattr(plc, "_macro_pin_map"):
        pin_map = {}

        # Map macro pins to their indices
        for idx, mod in enumerate(plc.modules_w_pins):
            if mod.get_type() == "MACRO_PIN" and hasattr(mod, "get_macro_name"):
                pin_map.setdefault(mod.get_macro_name(), []).append(idx)
                
        plc._macro_pin_map = pin_map

    # Set positions for hard macros
    for i, macro_idx in enumerate(benchmark.hard_macro_indices):
        node = plc.modules_w_pins[macro_idx]
        x, y = placement_np[i]
        node.set_pos(x, y)

        for pin_idx in plc._macro_pin_map.get(node.get_name(), []):
            pin = plc.modules_w_pins[pin_idx]
            pin.set_pos(x + pin.x_offset, y + pin.y_offset)

    # Set positions for soft macros
    num_hard = benchmark.num_hard_macros
    for i, macro_idx in enumerate(benchmark.soft_macro_indices):
        node = plc.modules_w_pins[macro_idx]
        x, y = placement_np[num_hard + i]
        node.set_pos(x, y)

        for pin_idx in plc._macro_pin_map.get(node.get_name(), []):
            pin = plc.modules_w_pins[pin_idx]
            pin.set_pos(x + pin.x_offset, y + pin.y_offset)

    # Update flags to recalculate costs
    _ensure_congestion_arrays(plc)
    plc.FLAG_UPDATE_WIRELENGTH = True
    plc.FLAG_UPDATE_DENSITY = True
    plc.FLAG_UPDATE_CONGESTION = True

# Compute the exact proxy cost using the true evaluator
def _exact_proxy_cost(plc, placement: torch.Tensor, benchmark: Benchmark) -> tuple[float, int]:
    _set_plc_placement(plc, placement, benchmark)  # Set placement in the evaluator
    wire = float(plc.get_cost())  # Compute wirelength cost
    density = float(plc.get_density_cost())  # Compute density cost
    congestion = float(plc.get_congestion_cost())  # Compute congestion cost

    # Combine costs using proxy weights
    proxy = (
        PROXY_WEIGHTS["wirelength"] * wire
        + PROXY_WEIGHTS["density"] * density
        + PROXY_WEIGHTS["congestion"] * congestion
    )
    overlaps = _count_hard_overlaps(placement[: benchmark.num_hard_macros], benchmark)  # Count overlaps

    return proxy, overlaps

# Count overlaps between hard macros
def _count_hard_overlaps(hard_positions: torch.Tensor, benchmark: Benchmark) -> int:
    num_hard = benchmark.num_hard_macros
    sizes = benchmark.macro_sizes[:num_hard]
    x_min = hard_positions[:, 0] - sizes[:, 0] / 2
    x_max = hard_positions[:, 0] + sizes[:, 0] / 2
    y_min = hard_positions[:, 1] - sizes[:, 1] / 2
    y_max = hard_positions[:, 1] + sizes[:, 1] / 2

    overlaps = 0
    for i in range(num_hard):
        for j in range(i + 1, num_hard):
            # Check for overlap between macro i and macro j
            if not (x_min[i] >= x_max[j] or x_max[i] <= x_min[j] or 
                    y_min[i] >= y_max[j] or y_max[i] <= y_min[j]):
                overlaps += 1
                
    return overlaps

# Class implementing the MOSAIC placer
class MOSAICPlacer:
    def __init__(self, seed: int = 7, iterations: int = 40, learning_rate: float = 0.06,
                wire_tau: float = 0.15, route_sigma_scale: float = 0.85, interval_tau_scale: float = 0.60,
                tail_beta: float = 18.0, overlap_weight: float = 6.0,
                soft_iterations: int = 60, soft_learning_rate: float = 0.035):
        self.seed = seed
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.wire_tau = wire_tau
        self.route_sigma_scale = route_sigma_scale
        self.interval_tau_scale = interval_tau_scale
        self.tail_beta = tail_beta
        self.overlap_weight = overlap_weight
        self.soft_iterations = soft_iterations
        self.soft_learning_rate = soft_learning_rate

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        num_hard = benchmark.num_hard_macros
        if num_hard == 0:
            return benchmark.macro_positions.clone()

        # Initialize hard macro positions
        movable_mask = (~benchmark.macro_fixed[:num_hard]).clone()
        hard_init = benchmark.macro_positions[:num_hard].detach().cpu().numpy().astype(np.float64)
        hard_sizes_np = benchmark.macro_sizes[:num_hard].detach().cpu().numpy().astype(np.float64)
        hard_start = torch.tensor(hard_init, dtype=torch.float32)

        # Load placement cost evaluator
        plc = _load_plc(benchmark.name)
        if plc is None:
            full_positions = benchmark.macro_positions.clone()
            full_positions[:num_hard] = hard_start
            return full_positions

        # Precompute evaluator data for optimization
        objective = self._build_objective_data(benchmark, plc, movable_mask)
        refined_hard = self._optimize_hard_macros(benchmark, hard_start, movable_mask, objective)

        # Legalize placement after optimization
        legalized_refined = torch.tensor(
            self._legalize(
                refined_hard.detach().cpu().numpy().astype(np.float64),
                movable_mask.detach().cpu().numpy(),
                hard_sizes_np,
                float(benchmark.canvas_width),
                float(benchmark.canvas_height),
            ),
            dtype=torch.float32,
        )

        # Compare and select the best placement
        candidates = [hard_start, refined_hard.detach().cpu(), legalized_refined]
        best_hard = self._pick_best_candidate(candidates, benchmark, plc)

        if benchmark.num_soft_macros == 0:
            full_positions = benchmark.macro_positions.clone()
            full_positions[:num_hard] = best_hard.to(dtype=benchmark.macro_positions.dtype)
            return full_positions

        # Optimize soft macro positions
        soft_start = benchmark.macro_positions[num_hard:].detach().cpu().to(torch.float32)
        soft_movable_mask = (~benchmark.macro_fixed[num_hard:]).clone()

        soft_objective = self._build_soft_objective_data(
            benchmark,
            plc,
            best_hard.detach().cpu().to(torch.float32),
            soft_movable_mask,
        )

        refined_soft = self._optimize_soft_macros(
            benchmark,
            soft_start,
            soft_movable_mask,
            soft_objective,
        )

        # Combine hard and soft placements and return the result
        full_positions = self._pick_best_full_candidate(
            [soft_start, refined_soft.detach().cpu()],
            best_hard.detach().cpu(),
            benchmark,
            plc,
        )

        return full_positions


    def _pick_best_candidate(self, candidates, benchmark: Benchmark, plc):
        """
        Evaluates multiple candidate placements using the exact evaluator and returns the best.
        Heavily penalizes any overlaps so that legal placements always win over invalid ones.
        """
        num_hard = benchmark.num_hard_macros
        best_score = float("inf")
        best_hard = candidates[0]

        seen = set()
        for hard_pos in candidates:

            # Rounded hashing avoids rescoring duplicates created when the
            # optimizer or legalizer returns the same placement.
            key = tuple(torch.round(hard_pos.flatten(), decimals=4).tolist())
            if key in seen:
                continue
            seen.add(key)

            full = benchmark.macro_positions.clone()
            full[:num_hard] = hard_pos.to(dtype=full.dtype)

            # Overlaps are penalized heavily so exact legal placements always
            # win over slightly lower-cost but invalid ones.
            proxy, overlaps = _exact_proxy_cost(plc, full, benchmark)
            score = proxy if overlaps == 0 else proxy + 1000.0 * overlaps

            if score < best_score:
                best_score = score
                best_hard = hard_pos

        return best_hard
    
    def _pick_best_full_candidate(self, soft_candidates, hard_pos: torch.Tensor, benchmark: Benchmark, plc):
        """
        Evaluates full placements after soft-macro refinement while keeping the hard
        macros fixed at the chosen hard placement.
        """
        num_hard = benchmark.num_hard_macros
        best_score = float("inf")
        best_full = None
        seen = set()

        for soft_pos in soft_candidates:
            key = tuple(torch.round(soft_pos.flatten(), decimals=4).tolist())
            if key in seen:
                continue
            seen.add(key)

            full = benchmark.macro_positions.clone()
            full[:num_hard] = hard_pos.to(dtype=full.dtype)
            full[num_hard:] = soft_pos.to(dtype=full.dtype)

            proxy, overlaps = _exact_proxy_cost(plc, full, benchmark)
            score = proxy if overlaps == 0 else proxy + 1000.0 * overlaps

            if score < best_score:
                best_score = score
                best_full = full

        return best_full


    def _build_objective_data(self, benchmark: Benchmark, plc, movable_mask: torch.Tensor):
        """
        One-time preprocessor for evaluator data at the start of optimization. Constructs routing
        grid geometry, net connectivivy, and pin ownership. Filters to only active nets where at least
        one pin can move.
        Returns a dictionary of tensors used during optimization to evaluate the surrogate cost.
        """
        device = torch.device("cpu")
        num_hard = benchmark.num_hard_macros

        # Map evaluator macro names back to benchmark hard-macro indices so we
        # can express every pin as either:
        #   1. attached to an optimized hard macro, or
        #   2. fixed in absolute coordinates.
        hard_name_to_idx = {
            plc.modules_w_pins[plc_idx].get_name(): hard_idx
            for hard_idx, plc_idx in enumerate(plc.hard_macro_indices)
        }

        fixed_positions = benchmark.macro_positions.detach().cpu()
        fixed_soft_pos = fixed_positions[num_hard:].to(torch.float32)
        fixed_soft_sizes = benchmark.macro_sizes[num_hard:].detach().cpu().to(torch.float32)
        hard_sizes = benchmark.macro_sizes[:num_hard].detach().cpu().to(torch.float32)

        grid_rows = int(benchmark.grid_rows)
        grid_cols = int(benchmark.grid_cols)

        canvas_w = float(benchmark.canvas_width)
        canvas_h = float(benchmark.canvas_height)

        grid_w = canvas_w / grid_cols
        grid_h = canvas_h / grid_rows

        # The differentiable congestion surrogate lives on the same routing
        # grid as the evaluator, so we precompute the cell centers and bounds.
        row_centers = torch.arange(grid_rows, dtype=torch.float32, device=device) * grid_h + grid_h / 2
        col_centers = torch.arange(grid_cols, dtype=torch.float32, device=device) * grid_w + grid_w / 2

        row_idx = torch.arange(grid_rows, dtype=torch.float32, device=device)
        col_idx = torch.arange(grid_cols, dtype=torch.float32, device=device)
        grid_col, grid_row = torch.meshgrid(col_idx, row_idx, indexing="xy")

        cell_xmin = (grid_col.flatten() * grid_w).to(torch.float32)
        cell_xmax = cell_xmin + grid_w
        cell_ymin = (grid_row.flatten() * grid_h).to(torch.float32)
        cell_ymax = cell_ymin + grid_h

        pin_owner = []
        pin_offset_x = []
        pin_offset_y = []
        pin_fixed_x = []
        pin_fixed_y = []
        pin_net_ids = []
        net_weights = []

        src_owner = []
        src_offset_x = []
        src_offset_y = []
        src_fixed_x = []
        src_fixed_y = []

        sink_owner = []
        sink_offset_x = []
        sink_offset_y = []
        sink_fixed_x = []
        sink_fixed_y = []

        conn_weights = []

        active_net_id = 0
        for driver_name, sinks in plc.nets.items():
            # Use the evaluator's native nets directly so our wirelength and
            # congestion surrogates track the same connectivity.
            driver_idx = plc.mod_name_to_indices[driver_name]
            driver = plc.modules_w_pins[driver_idx]
            weight = float(driver.get_weight()) if hasattr(driver, "get_weight") else 1.0
            if weight <= 0:
                weight = 1.0

            # Each pin is encoded either by a hard-macro owner index plus a pin
            # offset, or as a fixed absolute location when it belongs to a port
            # or a non-optimized object.
            pin_specs = [self._pin_spec(driver_name, plc, hard_name_to_idx)]
            pin_specs.extend(self._pin_spec(sink_name, plc, hard_name_to_idx) for sink_name in sinks)

            # Skip nets that cannot influence the optimization because every pin
            # on the net is fixed from the hard-macro optimizer's perspective.
            if not any(spec["owner"] >= 0 and movable_mask[spec["owner"]] for spec in pin_specs):
                continue

            for spec in pin_specs:
                pin_owner.append(spec["owner"])
                pin_offset_x.append(spec["offset_x"])
                pin_offset_y.append(spec["offset_y"])
                pin_fixed_x.append(spec["fixed_x"])
                pin_fixed_y.append(spec["fixed_y"])
                pin_net_ids.append(active_net_id)

            driver_spec = pin_specs[0]
            for sink_spec in pin_specs[1:]:
                # Build source/sink tensors for the congestion surrogate. We
                # only need a pair if at least one endpoint can move.
                if not (
                    (driver_spec["owner"] >= 0 and movable_mask[driver_spec["owner"]])
                    or (sink_spec["owner"] >= 0 and movable_mask[sink_spec["owner"]])
                ):
                    continue

                src_owner.append(driver_spec["owner"])
                src_offset_x.append(driver_spec["offset_x"])
                src_offset_y.append(driver_spec["offset_y"])
                src_fixed_x.append(driver_spec["fixed_x"])
                src_fixed_y.append(driver_spec["fixed_y"])

                sink_owner.append(sink_spec["owner"])
                sink_offset_x.append(sink_spec["offset_x"])
                sink_offset_y.append(sink_spec["offset_y"])
                sink_fixed_x.append(sink_spec["fixed_x"])
                sink_fixed_y.append(sink_spec["fixed_y"])
                conn_weights.append(weight)

            net_weights.append(weight)
            active_net_id += 1

        # Soft macros remain fixed in this version, but they still occupy area
        # on the density grid and should therefore bias the optimizer away from
        # already-crowded regions.
        density_fixed_area = self._fixed_macro_overlap_area(
            fixed_soft_pos,
            fixed_soft_sizes,
            cell_xmin,
            cell_xmax,
            cell_ymin,
            cell_ymax,
        )

        return {
            "hard_sizes": hard_sizes,
            "cell_xmin": cell_xmin,
            "cell_xmax": cell_xmax,
            "cell_ymin": cell_ymin,
            "cell_ymax": cell_ymax,
            "row_centers": row_centers,
            "col_centers": col_centers,
            "grid_rows": grid_rows,
            "grid_cols": grid_cols,
            "grid_w": float(grid_w),
            "grid_h": float(grid_h),
            "grid_area": float(grid_w * grid_h),
            "cap_h": float(grid_h * benchmark.hroutes_per_micron),
            "cap_v": float(grid_w * benchmark.vroutes_per_micron),
            "hrouting_alloc": float(getattr(plc, "hrouting_alloc", 0.0)),
            "vrouting_alloc": float(getattr(plc, "vrouting_alloc", 0.0)),
            "smooth_range": int(getattr(plc, "smooth_range", 0)),
            "pin_owner": torch.tensor(pin_owner, dtype=torch.long, device=device),
            "pin_offset_x": torch.tensor(pin_offset_x, dtype=torch.float32, device=device),
            "pin_offset_y": torch.tensor(pin_offset_y, dtype=torch.float32, device=device),
            "pin_fixed_x": torch.tensor(pin_fixed_x, dtype=torch.float32, device=device),
            "pin_fixed_y": torch.tensor(pin_fixed_y, dtype=torch.float32, device=device),
            "pin_net_ids": torch.tensor(pin_net_ids, dtype=torch.long, device=device),
            "net_weights": torch.tensor(net_weights, dtype=torch.float32, device=device),
            "num_active_nets": active_net_id,
            "plc_net_count": max(int(getattr(plc, "net_cnt", active_net_id)), 1),
            "src_owner": torch.tensor(src_owner, dtype=torch.long, device=device),
            "src_offset_x": torch.tensor(src_offset_x, dtype=torch.float32, device=device),
            "src_offset_y": torch.tensor(src_offset_y, dtype=torch.float32, device=device),
            "src_fixed_x": torch.tensor(src_fixed_x, dtype=torch.float32, device=device),
            "src_fixed_y": torch.tensor(src_fixed_y, dtype=torch.float32, device=device),
            "sink_owner": torch.tensor(sink_owner, dtype=torch.long, device=device),
            "sink_offset_x": torch.tensor(sink_offset_x, dtype=torch.float32, device=device),
            "sink_offset_y": torch.tensor(sink_offset_y, dtype=torch.float32, device=device),
            "sink_fixed_x": torch.tensor(sink_fixed_x, dtype=torch.float32, device=device),
            "sink_fixed_y": torch.tensor(sink_fixed_y, dtype=torch.float32, device=device),
            "conn_weights": torch.tensor(conn_weights, dtype=torch.float32, device=device),
            "fixed_density_area": density_fixed_area.to(torch.float32),
        }
    
    def _build_soft_objective_data(self, benchmark: Benchmark, plc, hard_pos: torch.Tensor, movable_mask: torch.Tensor):
        """
        Builds a differentiable objective for soft-macro-only refinement.
        Hard macros are treated as fixed context; soft macros are the variables.
        """
        device = torch.device("cpu")
        num_hard = benchmark.num_hard_macros

        hard_pos = hard_pos.detach().to(torch.float32).to(device)
        movable_mask = movable_mask.to(device)

        hard_name_to_idx = {
            plc.modules_w_pins[plc_idx].get_name(): hard_idx
            for hard_idx, plc_idx in enumerate(plc.hard_macro_indices)
        }
        soft_name_to_idx = {
            plc.modules_w_pins[plc_idx].get_name(): soft_idx
            for soft_idx, plc_idx in enumerate(plc.soft_macro_indices)
        }

        soft_sizes = benchmark.macro_sizes[num_hard:].detach().to(torch.float32).to(device)
        hard_sizes = benchmark.macro_sizes[:num_hard].detach().to(torch.float32).to(device)

        grid_rows = int(benchmark.grid_rows)
        grid_cols = int(benchmark.grid_cols)
        
        canvas_w = float(benchmark.canvas_width)
        canvas_h = float(benchmark.canvas_height)
        
        grid_w = canvas_w / grid_cols
        grid_h = canvas_h / grid_rows

        row_centers = torch.arange(grid_rows, dtype=torch.float32, device=device) * grid_h + grid_h / 2
        col_centers = torch.arange(grid_cols, dtype=torch.float32, device=device) * grid_w + grid_w / 2

        row_idx = torch.arange(grid_rows, dtype=torch.float32, device=device)
        col_idx = torch.arange(grid_cols, dtype=torch.float32, device=device)
        grid_col, grid_row = torch.meshgrid(col_idx, row_idx, indexing="xy")

        cell_xmin = (grid_col.flatten() * grid_w).to(torch.float32)
        cell_xmax = cell_xmin + grid_w
        
        cell_ymin = (grid_row.flatten() * grid_h).to(torch.float32)
        cell_ymax = cell_ymin + grid_h

        pin_owner = []
        pin_offset_x = []
        pin_offset_y = []
        pin_fixed_x = []
        pin_fixed_y = []
        pin_net_ids = []
        net_weights = []

        src_owner = []
        src_offset_x = []
        src_offset_y = []
        src_fixed_x = []
        src_fixed_y = []

        sink_owner = []
        sink_offset_x = []
        sink_offset_y = []
        sink_fixed_x = []
        sink_fixed_y = []

        conn_weights = []

        active_net_id = 0
        for driver_name, sinks in plc.nets.items():
            driver_idx = plc.mod_name_to_indices[driver_name]
            driver = plc.modules_w_pins[driver_idx]
            weight = float(driver.get_weight()) if hasattr(driver, "get_weight") else 1.0
            if weight <= 0:
                weight = 1.0

            pin_specs = [self._soft_pin_spec(driver_name, plc, hard_name_to_idx, soft_name_to_idx, hard_pos)]
            pin_specs.extend(
                self._soft_pin_spec(sink_name, plc, hard_name_to_idx, soft_name_to_idx, hard_pos)
                for sink_name in sinks
            )

            if not any(spec["owner"] >= 0 and movable_mask[spec["owner"]] for spec in pin_specs):
                continue

            for spec in pin_specs:
                pin_owner.append(spec["owner"])
                pin_offset_x.append(spec["offset_x"])
                pin_offset_y.append(spec["offset_y"])
                pin_fixed_x.append(spec["fixed_x"])
                pin_fixed_y.append(spec["fixed_y"])
                pin_net_ids.append(active_net_id)

            driver_spec = pin_specs[0]
            for sink_spec in pin_specs[1:]:
                if not (
                    (driver_spec["owner"] >= 0 and movable_mask[driver_spec["owner"]])
                    or (sink_spec["owner"] >= 0 and movable_mask[sink_spec["owner"]])
                ):
                    continue

                src_owner.append(driver_spec["owner"])
                src_offset_x.append(driver_spec["offset_x"])
                src_offset_y.append(driver_spec["offset_y"])
                src_fixed_x.append(driver_spec["fixed_x"])
                src_fixed_y.append(driver_spec["fixed_y"])

                sink_owner.append(sink_spec["owner"])
                sink_offset_x.append(sink_spec["offset_x"])
                sink_offset_y.append(sink_spec["offset_y"])
                sink_fixed_x.append(sink_spec["fixed_x"])
                sink_fixed_y.append(sink_spec["fixed_y"])
                conn_weights.append(weight)

            net_weights.append(weight)
            active_net_id += 1

        hard_fixed_area = self._fixed_macro_overlap_area(
            hard_pos,
            hard_sizes,
            cell_xmin,
            cell_xmax,
            cell_ymin,
            cell_ymax,
        )

        if (~movable_mask).any():
            frozen_soft_pos = benchmark.macro_positions[num_hard:].detach().to(torch.float32).to(device)[~movable_mask]
            frozen_soft_sizes = soft_sizes[~movable_mask]
            frozen_soft_area = self._fixed_macro_overlap_area(
                frozen_soft_pos,
                frozen_soft_sizes,
                cell_xmin,
                cell_xmax,
                cell_ymin,
                cell_ymax,
            )
        else:
            frozen_soft_area = torch.zeros_like(cell_xmin)

        return {
            "hard_sizes": soft_sizes,
            "cell_xmin": cell_xmin,
            "cell_xmax": cell_xmax,
            "cell_ymin": cell_ymin,
            "cell_ymax": cell_ymax,
            "row_centers": row_centers,
            "col_centers": col_centers,
            "grid_rows": grid_rows,
            "grid_cols": grid_cols,
            "grid_w": float(grid_w),
            "grid_h": float(grid_h),
            "grid_area": float(grid_w * grid_h),
            "cap_h": float(grid_h * benchmark.hroutes_per_micron),
            "cap_v": float(grid_w * benchmark.vroutes_per_micron),
            "hrouting_alloc": float(getattr(plc, "hrouting_alloc", 0.0)),
            "vrouting_alloc": float(getattr(plc, "vrouting_alloc", 0.0)),
            "smooth_range": int(getattr(plc, "smooth_range", 0)),
            "pin_owner": torch.tensor(pin_owner, dtype=torch.long, device=device),
            "pin_offset_x": torch.tensor(pin_offset_x, dtype=torch.float32, device=device),
            "pin_offset_y": torch.tensor(pin_offset_y, dtype=torch.float32, device=device),
            "pin_fixed_x": torch.tensor(pin_fixed_x, dtype=torch.float32, device=device),
            "pin_fixed_y": torch.tensor(pin_fixed_y, dtype=torch.float32, device=device),
            "pin_net_ids": torch.tensor(pin_net_ids, dtype=torch.long, device=device),
            "net_weights": torch.tensor(net_weights, dtype=torch.float32, device=device),
            "num_active_nets": active_net_id,
            "plc_net_count": max(int(getattr(plc, "net_cnt", active_net_id)), 1),
            "src_owner": torch.tensor(src_owner, dtype=torch.long, device=device),
            "src_offset_x": torch.tensor(src_offset_x, dtype=torch.float32, device=device),
            "src_offset_y": torch.tensor(src_offset_y, dtype=torch.float32, device=device),
            "src_fixed_x": torch.tensor(src_fixed_x, dtype=torch.float32, device=device),
            "src_fixed_y": torch.tensor(src_fixed_y, dtype=torch.float32, device=device),
            "sink_owner": torch.tensor(sink_owner, dtype=torch.long, device=device),
            "sink_offset_x": torch.tensor(sink_offset_x, dtype=torch.float32, device=device),
            "sink_offset_y": torch.tensor(sink_offset_y, dtype=torch.float32, device=device),
            "sink_fixed_x": torch.tensor(sink_fixed_x, dtype=torch.float32, device=device),
            "sink_fixed_y": torch.tensor(sink_fixed_y, dtype=torch.float32, device=device),
            "conn_weights": torch.tensor(conn_weights, dtype=torch.float32, device=device),
            "fixed_density_area": (hard_fixed_area + frozen_soft_area).to(torch.float32),
        }


    def _pin_spec(self, pin_name: str, plc, hard_name_to_idx: dict[str, int]):
        """
        Converts an evaluator pin name into a differentiable representation.
        PORT and non-optimized pins are encoded as fixed absolute coordinates, while
        pins owned by hard macros are stored as an owner index + offset.
        """

        # Convert a PlacementCost pin name into a compact differentiable
        # representation. Hard-macro pins are stored as (owner, offset); every
        # other pin is reduced to a fixed point in absolute coordinates.
        pin_idx = plc.mod_name_to_indices[pin_name]
        pin = plc.modules_w_pins[pin_idx]

        if pin.get_type() == "PORT":
            x, y = pin.get_pos()
            return {
                "owner": -1,
                "offset_x": 0.0,
                "offset_y": 0.0,
                "fixed_x": float(x),
                "fixed_y": float(y),
            }

        parent_name = pin_name.split("/")[0]
        if hasattr(pin, "get_offset"):
            offset_x, offset_y = pin.get_offset()
        else:
            offset_x, offset_y = pin.x_offset, pin.y_offset

        if parent_name in hard_name_to_idx:
            return {
                "owner": hard_name_to_idx[parent_name],
                "offset_x": float(offset_x),
                "offset_y": float(offset_y),
                "fixed_x": 0.0,
                "fixed_y": 0.0,
            }

        parent_idx = plc.mod_name_to_indices[parent_name]
        parent = plc.modules_w_pins[parent_idx]
        px, py = parent.get_pos()
        
        return {
            "owner": -1,
            "offset_x": 0.0,
            "offset_y": 0.0,
            "fixed_x": float(px + offset_x),
            "fixed_y": float(py + offset_y),
        }
        
    def _soft_pin_spec(self, pin_name: str, plc, hard_name_to_idx: dict[str, int],
                   soft_name_to_idx: dict[str, int], hard_pos: torch.Tensor):
        """
        For soft-stage optimization:
        - soft-macro pins are encoded as (owner, offset),
        - hard-macro pins are treated as fixed absolute points using the chosen hard placement,
        - ports / other pins are fixed absolute points.
        """
        pin_idx = plc.mod_name_to_indices[pin_name]
        pin = plc.modules_w_pins[pin_idx]

        if pin.get_type() == "PORT":
            x, y = pin.get_pos()
            return {
                "owner": -1,
                "offset_x": 0.0,
                "offset_y": 0.0,
                "fixed_x": float(x),
                "fixed_y": float(y),
            }

        parent_name = pin_name.split("/")[0]
        if hasattr(pin, "get_offset"):
            offset_x, offset_y = pin.get_offset()
        else:
            offset_x, offset_y = pin.x_offset, pin.y_offset

        if parent_name in soft_name_to_idx:
            return {
                "owner": soft_name_to_idx[parent_name],
                "offset_x": float(offset_x),
                "offset_y": float(offset_y),
                "fixed_x": 0.0,
                "fixed_y": 0.0,
            }

        if parent_name in hard_name_to_idx:
            hard_idx = hard_name_to_idx[parent_name]
            px = float(hard_pos[hard_idx, 0].item())
            py = float(hard_pos[hard_idx, 1].item())
            return {
                "owner": -1,
                "offset_x": 0.0,
                "offset_y": 0.0,
                "fixed_x": float(px + offset_x),
                "fixed_y": float(py + offset_y),
            }

        parent_idx = plc.mod_name_to_indices[parent_name]
        parent = plc.modules_w_pins[parent_idx]
        px, py = parent.get_pos()
        return {
            "owner": -1,
            "offset_x": 0.0,
            "offset_y": 0.0,
            "fixed_x": float(px + offset_x),
            "fixed_y": float(py + offset_y),
        }


    def _fixed_macro_overlap_area(self, positions: torch.Tensor, sizes: torch.Tensor, cell_xmin: torch.Tensor,
                                  cell_xmax: torch.Tensor, cell_ymin: torch.Tensor, cell_ymax: torch.Tensor) -> torch.Tensor:
        """
        Computes the exact continuous overlap between fixed macros and every grid routing cell. 
        Returns a tensor of per-cell accumulated overlap areas for density and congestion calculations.
        """
        if positions.numel() == 0:
            return torch.zeros_like(cell_xmin)

        # Compute exact continuous overlap area between each fixed macro and
        # each evaluator grid cell. This part is already piecewise linear, so
        # no further smoothing is needed.
        x_min = positions[:, 0:1] - sizes[:, 0:1] / 2
        x_max = positions[:, 0:1] + sizes[:, 0:1] / 2

        y_min = positions[:, 1:2] - sizes[:, 1:2] / 2
        y_max = positions[:, 1:2] + sizes[:, 1:2] / 2

        overlap_x = torch.relu(torch.minimum(x_max, cell_xmax.unsqueeze(0)) - torch.maximum(x_min, cell_xmin.unsqueeze(0)))
        overlap_y = torch.relu(torch.minimum(y_max, cell_ymax.unsqueeze(0)) - torch.maximum(y_min, cell_ymin.unsqueeze(0)))
        
        return (overlap_x * overlap_y).sum(dim=0)

    def _optimize_hard_macros(
        self, benchmark: Benchmark, hard_start: torch.Tensor,
        movable_mask: torch.Tensor, objective: dict) -> torch.Tensor:

        num_hard = benchmark.num_hard_macros
        sizes = benchmark.macro_sizes[:num_hard].detach().cpu().to(torch.float32)

        # Optimize the hard-macro centers directly. 
        # Fixed macros are clamped back after every step.
        param = hard_start.clone().to(torch.float32)
        param.requires_grad_(True)

        movable_idx = torch.where(movable_mask)[0]
        if movable_idx.numel() == 0:
            return hard_start

        # The evaluator uses top-k averages for density/congestion.
        density_tail = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        congestion_tail = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        optimizer = torch.optim.Adam([param, density_tail, congestion_tail], lr=self.learning_rate)

        x_lo = sizes[:, 0] / 2
        x_hi = float(benchmark.canvas_width) - sizes[:, 0] / 2
        y_lo = sizes[:, 1] / 2
        y_hi = float(benchmark.canvas_height) - sizes[:, 1] / 2

        best_param = param.detach().clone()
        best_score = float("inf")

        for step in range(self.iterations):
            optimizer.zero_grad()

            # Only movable hard macros follow the optimizer variables; fixed
            # ones remain at their legalized start locations.
            hard_pos = hard_start.clone()
            hard_pos[movable_idx] = param[movable_idx]

            wire = self._wirelength_cost(hard_pos, objective)
            density = self._density_cost(hard_pos, objective, density_tail)
            congestion = self._congestion_cost(hard_pos, objective, congestion_tail)
            overlap = self._overlap_penalty(hard_pos, sizes, movable_mask)

            # Fade the overlap penalty slightly over time: strong enough early
            # to avoid collisions, but less dominant later once the placement
            # has spread out.
            anneal = 1.0 - 0.5 * (step / max(self.iterations - 1, 1))
            loss = (
                PROXY_WEIGHTS["wirelength"] * wire
                + PROXY_WEIGHTS["density"] * density
                + PROXY_WEIGHTS["congestion"] * congestion
                + anneal * self.overlap_weight * overlap
            )

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                # Project every macro center back into the canvas after each
                # update and explicitly restore fixed macros.
                param[:, 0].clamp_(x_lo, x_hi)
                param[:, 1].clamp_(y_lo, y_hi)
                param[~movable_mask] = hard_start[~movable_mask]

                score = float(loss.detach())
                if score < best_score:
                    best_score = score
                    best_param = hard_pos.detach().clone()

        return best_param
    
    def _optimize_soft_macros(self, benchmark: Benchmark, soft_start: torch.Tensor,
                          movable_mask: torch.Tensor, objective: dict) -> torch.Tensor:
        """
        Refines only the soft macros while holding hard macros fixed.
        Reuses the same differentiable wirelength / density / congestion surrogates.
        """
        device = objective["hard_sizes"].device
        num_hard = benchmark.num_hard_macros
        sizes = benchmark.macro_sizes[num_hard:].detach().to(torch.float32).to(device)

        soft_start = soft_start.clone().to(torch.float32).to(device)
        movable_mask = movable_mask.to(device)

        param = soft_start.clone()
        param.requires_grad_(True)

        movable_idx = torch.where(movable_mask)[0]
        if movable_idx.numel() == 0:
            return soft_start.detach().cpu()

        density_tail = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float32, device=device))
        congestion_tail = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float32, device=device))
        optimizer = torch.optim.Adam([param, density_tail, congestion_tail], lr=self.soft_learning_rate)

        x_lo = sizes[:, 0] / 2
        x_hi = float(benchmark.canvas_width) - sizes[:, 0] / 2
        y_lo = sizes[:, 1] / 2
        y_hi = float(benchmark.canvas_height) - sizes[:, 1] / 2

        best_param = soft_start.detach().clone()
        best_score = float("inf")

        for _ in range(self.soft_iterations):
            optimizer.zero_grad()

            soft_pos = soft_start.clone()
            soft_pos[movable_idx] = param[movable_idx]

            wire = self._wirelength_cost(soft_pos, objective)
            density = self._density_cost(soft_pos, objective, density_tail)
            congestion = self._congestion_cost(soft_pos, objective, congestion_tail)

            loss = (
                PROXY_WEIGHTS["wirelength"] * wire
                + PROXY_WEIGHTS["density"] * density
                + PROXY_WEIGHTS["congestion"] * congestion
            )

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                param[:, 0].clamp_(x_lo, x_hi)
                param[:, 1].clamp_(y_lo, y_hi)
                param[~movable_mask] = soft_start[~movable_mask]

                score = float(loss.detach())
                if score < best_score:
                    best_score = score
                    best_param = soft_pos.detach().clone()

        return best_param.detach().cpu()


    def _resolve_pin_coordinates(self, hard_pos: torch.Tensor, owner: torch.Tensor, offset_x: torch.Tensor,
                                 offset_y: torch.Tensor, fixed_x: torch.Tensor, fixed_y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        """
        Resolves pin positions by checking ownership: if a pin owns a hard macro, it computes the dynamic
        coordinate by adding the pin's offset to the macro's centre. Else, it returns the fixed aboslute
        coordinate for that pin.
        """
        if owner.numel() == 0:
            empty = torch.zeros(0, dtype=torch.float32)
            return empty, empty

        # For entries with owner >= 0, fetch the owning macro center and add
        # the pin offset. Otherwise fall back to the fixed coordinate.
        owner_safe = owner.clamp(min=0)
        dyn_x = hard_pos[owner_safe, 0] + offset_x
        dyn_y = hard_pos[owner_safe, 1] + offset_y

        is_dyn = owner >= 0

        x = torch.where(is_dyn, dyn_x, fixed_x)
        y = torch.where(is_dyn, dyn_y, fixed_y)

        return x, y

    def _segment_logsumexp(self, values: torch.Tensor, seg_ids: torch.Tensor, num_seg: int, tau: float) -> torch.Tensor:
        """
        Computes a batched soft maximum over segments, using log-sum-exponent for stability.
        Returns a differentiable approximation of per-segment maxima.
        """

        if values.numel() == 0 or num_seg == 0:
            return torch.zeros(num_seg, dtype=torch.float32)

        # Batched soft-max over ragged nets. This is the core primitive behind
        # soft-HPWL: logsumexp approximates max, and -logsumexp(-x) approximates min.
        scaled = values / tau

        max_scaled = torch.full((num_seg,), -1e9, dtype=values.dtype, device=values.device)
        max_scaled.scatter_reduce_(0, seg_ids, scaled, reduce="amax", include_self=True)

        exp_sum = torch.zeros(num_seg, dtype=values.dtype, device=values.device)
        exp_sum.index_add_(0, seg_ids, torch.exp(scaled - max_scaled[seg_ids]))

        return tau * (max_scaled + torch.log(exp_sum.clamp_min(1e-12)))

    def _wirelength_cost(self, hard_pos: torch.Tensor, objective: dict) -> torch.Tensor:
        """
        Computes a smooth, differentiable value for the wirelegth proxy cost using log-sum-exponents
        instead of max/min to approximate the bounding box of each net's pins.
        """

        # Resolve every active net pin to its current continuous coordinate.
        pin_x, pin_y = self._resolve_pin_coordinates(
            hard_pos,
            objective["pin_owner"],
            objective["pin_offset_x"],
            objective["pin_offset_y"],
            objective["pin_fixed_x"],
            objective["pin_fixed_y"],
        )

        if pin_x.numel() == 0 or objective["num_active_nets"] == 0:
            return torch.tensor(0.0, dtype=torch.float32)

        # Soft-HPWL surrogate:
        #   softmax(x) - softmin(x) + softmax(y) - softmin(y)
        # per net, weighted exactly like the evaluator nets.
        tau = self.wire_tau

        x_max = self._segment_logsumexp(pin_x, objective["pin_net_ids"], objective["num_active_nets"], tau)
        x_min = -self._segment_logsumexp(-pin_x, objective["pin_net_ids"], objective["num_active_nets"], tau)

        y_max = self._segment_logsumexp(pin_y, objective["pin_net_ids"], objective["num_active_nets"], tau)
        y_min = -self._segment_logsumexp(-pin_y, objective["pin_net_ids"], objective["num_active_nets"], tau)

        hpwl = objective["net_weights"] * ((x_max - x_min) + (y_max - y_min))

        # Match the evaluator's normalization by canvas perimeter and net count.
        norm = (objective["grid_w"] * objective["grid_cols"] + objective["grid_h"] * objective["grid_rows"]) * objective["plc_net_count"]

        return hpwl.sum() / max(norm, 1e-6)

    def _tail_average(self, values: torch.Tensor, ratio: float, tail_var: torch.nn.Parameter) -> torch.Tensor:
        """
        Computes a smooth approximation of the average of top values using a learnable threshold.
        Result is differentiable and behaves like a top k-tail instead of a hard selection.
        """

        if values.numel() == 0:
            return torch.tensor(0.0, dtype=torch.float32)

        # Smooth approximation to "average of the top ratio fraction":
        #   t + sum(softplus(v - t)) / k
        # where t is a learnable threshold and k is the nominal tail size.
        k = max(1, int(math.floor(values.numel() * ratio)))

        return tail_var + F.softplus(values - tail_var, beta=self.tail_beta).sum() / float(k)

    def _density_cost(self, hard_pos: torch.Tensor, objective: dict, tail_var: torch.nn.Parameter) -> torch.Tensor:
        """
        Computes a scalar density surrogate used during optimization, mimics the evaluator's top-10% density cost.
        """

        sizes = objective["hard_sizes"]

        # Compute exact overlap area between every hard macro and every routing
        # cell, then convert that area to per-cell density.
        x_min = hard_pos[:, 0:1] - sizes[:, 0:1] / 2
        x_max = hard_pos[:, 0:1] + sizes[:, 0:1] / 2
        y_min = hard_pos[:, 1:2] - sizes[:, 1:2] / 2
        y_max = hard_pos[:, 1:2] + sizes[:, 1:2] / 2

        overlap_x = torch.relu(torch.minimum(x_max, objective["cell_xmax"].unsqueeze(0)) - torch.maximum(x_min, objective["cell_xmin"].unsqueeze(0)))
        overlap_y = torch.relu(torch.minimum(y_max, objective["cell_ymax"].unsqueeze(0)) - torch.maximum(y_min, objective["cell_ymin"].unsqueeze(0)))
        hard_area = (overlap_x * overlap_y).sum(dim=0)

        density = (hard_area + objective["fixed_density_area"]) / objective["grid_area"]

        # The evaluator returns 0.5 * average(top 10%), but we ignore here for tighter optimization.
        # Let's compare results with a lower (0.05) and higher (0.20) tail average
        return self._tail_average(density, 0.05, tail_var)

    def _smooth_route_map(self, route_map: torch.Tensor, smooth_range: int, axis: int) -> torch.Tensor:
        """
        Replaces each point with the local average over a neighbourhood of a given width along the chosen axis.
        """

        if smooth_range <= 0:
            return route_map

        # Reproduce the evaluator's one-dimensional box smoothing along the
        # routing direction, but as a differentiable convolution.
        kernel = torch.ones(1, 1, 2 * smooth_range + 1, dtype=route_map.dtype, device=route_map.device)
        if axis == 1:
            data = route_map
        else:
            data = route_map.transpose(0, 1)

        data_1d = data.reshape(-1, 1, data.shape[-1])
        numer = F.conv1d(data_1d, kernel, padding=smooth_range)
        denom = F.conv1d(torch.ones_like(data_1d), kernel, padding=smooth_range)
        smoothed = (numer / denom.clamp_min(1e-6)).reshape_as(data)
        return smoothed if axis == 1 else smoothed.transpose(0, 1)

    def _congestion_cost(self, hard_pos: torch.Tensor, objective: dict, tail_var: torch.nn.Parameter) -> torch.Tensor:
        """
        Computes a smooth differentiable congestion cost inspired by the evaluator's 
        discrete model using continuous pin coordinates and interval coverage instead.
        """

        # Build smooth source/sink coordinates for every active two-pin pair.
        src_x, src_y = self._resolve_pin_coordinates(
            hard_pos,
            objective["src_owner"],
            objective["src_offset_x"],
            objective["src_offset_y"],
            objective["src_fixed_x"],
            objective["src_fixed_y"],
        )
        sink_x, sink_y = self._resolve_pin_coordinates(
            hard_pos,
            objective["sink_owner"],
            objective["sink_offset_x"],
            objective["sink_offset_y"],
            objective["sink_fixed_x"],
            objective["sink_fixed_y"],
        )

        if src_x.numel() == 0:
            return torch.tensor(0.0, dtype=torch.float32)

        # sigma controls how softly endpoints snap to nearby grid rows/cols.
        # tau controls how softly a route covers the interval between endpoints.
        sigma_x = self.route_sigma_scale * objective["grid_w"]
        sigma_y = self.route_sigma_scale * objective["grid_h"]
        tau_x = self.interval_tau_scale * objective["grid_w"]
        tau_y = self.interval_tau_scale * objective["grid_h"]

        # Endpoint-to-grid association: each endpoint distributes demand to
        # nearby rows/cols with normalized Gaussian weights instead of a hard
        # floor() snap.
        src_row_prob = self._normalized_gaussian(src_y.unsqueeze(1), objective["row_centers"].unsqueeze(0), sigma_y)
        sink_col_prob = self._normalized_gaussian(sink_x.unsqueeze(1), objective["col_centers"].unsqueeze(0), sigma_x)

        x_lo = torch.minimum(src_x, sink_x).unsqueeze(1)
        x_hi = torch.maximum(src_x, sink_x).unsqueeze(1)
        y_lo = torch.minimum(src_y, sink_y).unsqueeze(1)
        y_hi = torch.maximum(src_y, sink_y).unsqueeze(1)

        # Smooth interval indicator for the L-route span in each dimension.
        # This replaces the evaluator's discrete step over covered grid edges.
        col_cover = torch.sigmoid((objective["col_centers"].unsqueeze(0) - x_lo) / tau_x) - torch.sigmoid((objective["col_centers"].unsqueeze(0) - x_hi) / tau_x)
        row_cover = torch.sigmoid((objective["row_centers"].unsqueeze(0) - y_lo) / tau_y) - torch.sigmoid((objective["row_centers"].unsqueeze(0) - y_hi) / tau_y)

        weighted_rows = src_row_prob * objective["conn_weights"].unsqueeze(1)
        weighted_cols = sink_col_prob * objective["conn_weights"].unsqueeze(1)

        # Horizontal demand is generated by rows chosen near the source and the
        # covered x-interval; vertical demand is analogous for columns/sinks.
        h_route = weighted_rows.transpose(0, 1) @ col_cover
        v_route = row_cover.transpose(0, 1) @ weighted_cols

        sizes = objective["hard_sizes"]
        # Macro blockage uses continuous overlap widths/heights with grid cells,
        # analogous to the evaluator's macro routing allocation term.
        x_min = hard_pos[:, 0:1] - sizes[:, 0:1] / 2
        x_max = hard_pos[:, 0:1] + sizes[:, 0:1] / 2
        y_min = hard_pos[:, 1:2] - sizes[:, 1:2] / 2
        y_max = hard_pos[:, 1:2] + sizes[:, 1:2] / 2

        overlap_x = torch.relu(torch.minimum(x_max, objective["cell_xmax"].unsqueeze(0)) - torch.maximum(x_min, objective["cell_xmin"].unsqueeze(0)))
        overlap_y = torch.relu(torch.minimum(y_max, objective["cell_ymax"].unsqueeze(0)) - torch.maximum(y_min, objective["cell_ymin"].unsqueeze(0)))

        v_macro = (overlap_x * objective["vrouting_alloc"]).sum(dim=0).reshape(objective["grid_rows"], objective["grid_cols"])
        h_macro = (overlap_y * objective["hrouting_alloc"]).sum(dim=0).reshape(objective["grid_rows"], objective["grid_cols"])

        # Normalize by per-cell routing capacity, just like the evaluator.
        h_norm = h_route / max(objective["cap_h"], 1e-6)
        v_norm = v_route / max(objective["cap_v"], 1e-6)
        h_macro = h_macro / max(objective["cap_h"], 1e-6)
        v_macro = v_macro / max(objective["cap_v"], 1e-6)

        # Apply the evaluator's smoothing and then take a smooth top-5% tail.
        v_smooth = self._smooth_route_map(v_norm, objective["smooth_range"], axis=1)
        h_smooth = self._smooth_route_map(h_norm, objective["smooth_range"], axis=0)

        congestion = torch.cat([(v_smooth + v_macro).flatten(), (h_smooth + h_macro).flatten()])

        # return self._tail_average(congestion, 0.05, tail_var)
        return self._tail_average(congestion, 0.2, tail_var)

    def _normalized_gaussian(self, values: torch.Tensor, centers: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        Computes normalized Gaussian weights for each value-centre pair, weight 
        decays with distance and is controlled by sigma. The output is normalized 
        so that the weights for each value sum to 1 across all centers.
        """

        weights = torch.exp(-0.5 * ((values - centers) / max(sigma, 1e-6)) ** 2)
        
        return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-9)

    def _overlap_penalty(self, hard_pos: torch.Tensor, sizes: torch.Tensor, movable_mask: torch.Tensor) -> torch.Tensor:
        """
        Calculates a differentiable penalty based on the total pairwise overlap area between hard macros, 
        weighted by the optimizer's current focus on resolving collisions. Only pairs involving at least
        one movable macro contribute to the penalty.
        """

        dx = hard_pos[:, 0].unsqueeze(1) - hard_pos[:, 0].unsqueeze(0)
        dy = hard_pos[:, 1].unsqueeze(1) - hard_pos[:, 1].unsqueeze(0)

        abs_dx = torch.sqrt(dx * dx + 1e-8)
        abs_dy = torch.sqrt(dy * dy + 1e-8)

        sep_x = (sizes[:, 0].unsqueeze(1) + sizes[:, 0].unsqueeze(0)) / 2 + 1e-3
        sep_y = (sizes[:, 1].unsqueeze(1) + sizes[:, 1].unsqueeze(0)) / 2 + 1e-3

        overlap_x = torch.relu(sep_x - abs_dx)
        overlap_y = torch.relu(sep_y - abs_dy)

        pair_overlap = overlap_x * overlap_y

        active = movable_mask.unsqueeze(1) | movable_mask.unsqueeze(0)
        tri = torch.triu(torch.ones_like(pair_overlap, dtype=torch.bool), diagonal=1)

        return pair_overlap[active & tri].sum() / max(int(hard_pos.shape[0]), 1)

    def _legalize(self, positions: np.ndarray, movable: np.ndarray,
                  sizes: np.ndarray, canvas_w: float, canvas_h: float) -> np.ndarray:
        """
        Greedy nearest-feasible legalizer to resolve any remaining overlaps after optimization. 
        Places larger macros first, keeping already-legal positions when possible, and otherwise 
        searching outward on an expanding square ring.
        """

        num_hard = positions.shape[0]
        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2
        sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
        sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2

        order = sorted(range(num_hard), key=lambda i: -sizes[i, 0] * sizes[i, 1])
        placed = np.zeros(num_hard, dtype=bool)
        legal = positions.copy()

        for idx in order:
            if not movable[idx]:
                # Fixed macros anchor the legalizer and are never moved.
                placed[idx] = True
                continue

            if placed.any():
                # If the current location is already legal with respect to the
                # macros we have committed so far, keep it.
                dx = np.abs(legal[idx, 0] - legal[:, 0])
                dy = np.abs(legal[idx, 1] - legal[:, 1])
                clash = (dx < sep_x[idx] + 1e-3) & (dy < sep_y[idx] + 1e-3) & placed
                clash[idx] = False
                if not clash.any():
                    placed[idx] = True
                    continue

            # Search granularity scales with macro size so large blocks move in
            # meaningful increments without exploring a huge dense lattice.
            step = max(sizes[idx, 0], sizes[idx, 1]) * 0.25
            best_point = legal[idx].copy()
            best_dist = float("inf")

            for radius in range(1, 180):
                found = False
                for dxm in range(-radius, radius + 1):
                    for dym in range(-radius, radius + 1):
                        # Only visit the perimeter of the current square ring.
                        if abs(dxm) != radius and abs(dym) != radius:
                            continue

                        # Clamp every candidate to the canvas so the legalizer
                        # enforces boundary legality as well as non-overlap.
                        cand_x = np.clip(positions[idx, 0] + dxm * step, half_w[idx], canvas_w - half_w[idx])
                        cand_y = np.clip(positions[idx, 1] + dym * step, half_h[idx], canvas_h - half_h[idx])

                        if placed.any():
                            dx = np.abs(cand_x - legal[:, 0])
                            dy = np.abs(cand_y - legal[:, 1])
                            clash = (dx < sep_x[idx] + 1e-3) & (dy < sep_y[idx] + 1e-3) & placed
                            clash[idx] = False
                            if clash.any():
                                continue

                        # Keep the legal candidate with minimum displacement 
                        # from the pre-legalization position.
                        dist = (cand_x - positions[idx, 0]) ** 2 + (cand_y - positions[idx, 1]) ** 2
                        if dist < best_dist:
                            best_dist = dist
                            best_point = np.array([cand_x, cand_y], dtype=np.float64)
                            found = True
                if found:
                    break

            legal[idx] = best_point
            placed[idx] = True

        return legal
