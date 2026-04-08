"""
Mass-on-Spring Analytical placer for Integrated Circuits (MOSAIC)

Team members: Arnav Patil
              Alexandre Singer

1. General Placement:
    a. Greedy initialization using spectal decomposition.
    b. Damped spring-mass equilibrium model with external forces.
2. Legalization:
    a. Lifted Will's legalizer for the time being, finds the 
       nearest legal solution from current placement.
3. Detailed Placement:
    a. Constrained SA for final tuning (maybe implemented in C++?).

Usage:
    uv run evaluate submissions/mosaiic/placer.py
    uv run evaluate submissions/mosaiic/placer.py --all
    uv run evaluate submissions/mosaiic/placer.py -b ibm03
"""

import torch
import numpy as np
from pathlib import Path
from macro_place.benchmark import Benchmark
from macro_place.loader import load_benchmark, load_benchmark_from_dir

OVERLAP_THESHOLD = 0.005
LEGALIZER_STEP_SIZE = 0.25

class MOSAICPlacer:
    
    def place(self, benchmark: Benchmark) -> torch.Tensor:
        # Setting up variables and parameters
        num_hard = benchmark.num_hard_macros
        sizes_np = benchmark.macro_sizes[:num_hard].numpy().astype(np.float64)
        canvas_w = float(benchmark.canvas_width)
        canvas_h = float(benchmark.canvas_height)
        half_w = sizes_np[:, 0] / 2
        half_h = sizes_np[:, 1] / 2
        
        # Extract movable hard macros
        movable = benchmark.get_movable_mask()[:num_hard].numpy()
        
        # Load placement to build adjacency matrix
        plc = self.load_plc(benchmark.name)
        if plc is not None:
            edges, edge_weights = self.extract_edges(benchmark, plc)
        else:
            edges = torch.zeros(0, 2, dtype=torch.long)
            edge_weights = torch.zeros(0)
        adjacency_mat = self.edges_to_adjacency(edges, edge_weights, num_hard)
        
        """ 1. General Placement """
        # positions = self.initialize_spectral(adjacency_mat, half_w, half_h, canvas_w, canvas_h)
        
        """ 2. Legalization """
        # positions = positions.astype(np.float64)
        positions = benchmark.macro_positions[:num_hard].numpy().copy().astype(np.float64)  # Use if want to test legalizer
        positions = self.legalize_simple(positions, movable, sizes_np, half_w, 
                                         half_h, canvas_w, canvas_h, num_hard)
        
        # Build full placement by keeping soft macros at initial positions
        full_positions = benchmark.macro_positions.clone()
        full_positions[:num_hard] = torch.tensor(positions, dtype=torch.float32)
        
        """ 3. Detailed Placement """
        # TODO
        
        return full_positions

    """ Placement Flow Functions """
    def legalize_simple(self, positions, movable, sizes, half_w, half_h, canvas_w, canvas_h, num_hard):
        """
        Implements a legalizer that finds the nearest legal solution from the provided initial placement.
        
        Greedy macro legalizer:
        1. Sort macros in order of descending area
        2. For each macro:
            a. If fixed, keep as is,
            b. If no overlap with already placed macros, keep its position
            c. Else, search outwards until it can fit.  
        3. Return the legalized positions.
        """
        
        # Precompute spacing constraints and parameters
        sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
        sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
        
        order = sorted(range(num_hard), key=lambda i: -sizes[i,0] * sizes[i,1])
        placed = np.zeros(num_hard, dtype=bool)
        legal = positions.copy()
        
        # In descending order by macro area
        for i in order:
            
            # Skip fixed macros
            if not movable[i]:
                placed[i] = True
                continue
            
            # Check already-placed macros for overlap
            if placed.any():
                dx = np.abs(legal[i, 0] - legal[:, 0])
                dy = np.abs(legal[i, 1] - legal[:, 1])
                c = (dx < sep_x[i] + OVERLAP_THESHOLD) & (dy < sep_y[i] + OVERLAP_THESHOLD) & placed
                c[i] = False
                
                if not c.any():
                    placed[i] = True
                    continue
            
            # Define step size, and track best candidate position that minimizes distance from current
            step = max(sizes[i, 0], sizes[i, 1]) * LEGALIZER_STEP_SIZE
            best_p = legal[i].copy()
            best_d = float('inf')
            
            for r in range(1, 150):
                found = False
                
                for dxm in range(-r, r+1):
                    for dym in range (-r, r+1):
                        # Skip already-explored inner points (?)
                        if abs(dxm) != r and abs(dym) != r: 
                            continue
                            
                        # Clamp to canvas bounds if necessary
                        cx = np.clip(positions[i, 0] + dxm * step, half_w[i], canvas_w - half_w[i])
                        cy = np.clip(positions[i, 1] + dym * step, half_h[i], canvas_h - half_h[i])
                        
                        # Same collision check for already-placed macros
                        if placed.any():
                            dx = np.abs(cx-legal[:, 0])
                            dy = np.abs(cy - legal[:, 1])
                            c = (dx < sep_x[i] + OVERLAP_THESHOLD) & (dy < sep_y[i] + OVERLAP_THESHOLD) & placed
                            c[i] = False
                            
                            if c.any(): 
                                continue
                        
                        # Compute displacement cost, this is the minimal value we want to track
                        d = (cx - positions[i, 0]) ** 2 + (cy - positions[i, 1]) ** 2
                        
                        if d < best_d:
                            best_d = d; best_p = np.array([cx, cy])
                            found = True
                
                if found: 
                    break
            
            # Finalize the best placement for that macro
            legal[i] = best_p
            placed[i] = True
    
        return legal
    
    def initialize_spectral(self, adjacency_mat, half_w, half_h, canvas_w, canvas_h):
        N = adjacency_mat.shape[0]
        
        # Fallback if edges somehow returns 0
        if adjacency_mat.sum() == 0:
            positions = np.zeros((N, 2), dtype=np.float64)
            positions[:, 0] = np.random.uniform(half_w, canvas_w - half_w)
            positions[:, 1] = np.random.uniform(half_h, canvas_h - half_h)
            return positions
        
        # Build Laplacian matrix
        W = adjacency_mat.astype(np.float64)
        D = np.diag(W.sum(axis=1))
        L = D - W
        
        # Stabilize the matrix
        L += 1e-6 * np.eye(N)
        
        # Get 2nd and 3rd eigenvectors
        eigvals, eigvecs = np.linalg.eigh(L)
        idx = np.argsort(eigvals)
        
        x_coords = eigvecs[:, idx[1]]
        y_coords = eigvecs[:, idx[2]]
        
        # Normalize
        def normalize(arr, lo, hi):
            mini, maxi = arr.min(), arr.max()
            if maxi - mini < 1e-8:
                return np.full_like(arr, (lo+hi) / 2)
            return lo + (arr - mini) * (hi - lo) / (maxi - mini)
        
        x_coords = normalize(x_coords, half_w, canvas_w - half_w)
        y_coords = normalize(y_coords, half_h, canvas_h - half_h)
        
        positions = np.stack([x_coords, y_coords], axis=1)
        
        # Add small noise then clamp
        positions += 1e-3 * np.random.randn(N, 2)
        positions[:, 0] = np.clip(positions[:, 0], half_w, canvas_w - half_w)
        positions[:, 1] = np.clip(positions[:, 1], half_h, canvas_h - half_h)
        
        return positions
    
    """ Helper Functions """
    def load_plc(self, name):
        root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
        
        if root.exists():
            _, plc = load_benchmark_from_dir(str(root))
            return plc
        
        ng45 = {"ariane133_ng45": "ariane133", "ariane136_ng45": "ariane136",
            "nvdla_ng45": "nvdla", "mempool_tile_ng45": "mempool_tile"}
        d = ng45.get(name)
        
        if d:
            base = Path("external/MacroPlacement/Flows/NanGate45") / d / "netlist" / "output_CT_Grouping"
            
            if (base / "netlist.pb.txt").exists():
                _, plc = load_benchmark(str(base / "netlist.pb.txt"), str(base / "initial.plc"))
                return plc
            
        return None
    
    def extract_edges(self, benchmark, plc):
        name_to_bidx = {}
        
        for bidx, idx in enumerate(plc.hard_macro_indices):
            name_to_bidx[plc.modules_w_pins[idx].get_name()] = bidx
        
        edge_dict = {}
        
        for driver, sinks in plc.nets.items():
            macros = set()
            
            for pin in [driver] + sinks:
                parent = pin.split("/")[0]
                
                if parent in name_to_bidx:
                    macros.add(name_to_bidx[parent])
            
            if len(macros) >= 2:
                ml = sorted(macros)
                w = 1.0 / (len(ml) - 1)
                
                for i in range(len(ml)):
                    for j in range(i+1, len(ml)):
                        pair = (ml[i], ml[j])
                        edge_dict[pair] = edge_dict.get(pair, 0) + w
        
        if not edge_dict:
            return torch.zeros(0, 2, dtype=torch.long), torch.zeros(0)
        
        return (torch.tensor(list(edge_dict.keys()), dtype=torch.long),
                torch.tensor([edge_dict[e] for e in edge_dict], dtype=torch.float32)) 
    
    def edges_to_adjacency(self, edges, edge_weights, num_hard):
        A = np.zeros((num_hard, num_hard), dtype=np.float32)
        
        if len(edges) == 0:
            return A

        e = edges.numpy()
        w = edge_weights.numpy()
        
        A[e[:, 0], e[:, 1]] = w
        A[e[:, 1], e[:, 0]] = w  # Adjacency matric should be symmetric
        
        return A
    
    
        
    
    
    
    