"""
Mass-on-Spring Analytical placer for Integrated Circuits (MOSAIC)

Team members: Arnav Patil
              Alexandre Singer
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
        # -------------------------
        # Setup
        # -------------------------
        num_hard = benchmark.num_hard_macros
        sizes_np = benchmark.macro_sizes[:num_hard].numpy().astype(np.float64)
        canvas_w = float(benchmark.canvas_width)
        canvas_h = float(benchmark.canvas_height)
        half_w = sizes_np[:, 0] / 2
        half_h = sizes_np[:, 1] / 2
        
        movable = benchmark.get_movable_mask()[:num_hard].numpy()
        
        # Build adjacency
        plc = self.load_plc(benchmark.name)
        if plc is not None:
            edges, edge_weights = self.extract_edges(benchmark, plc)
        else:
            edges = torch.zeros(0, 2, dtype=torch.long)
            edge_weights = torch.zeros(0)
        
        adjacency_mat = self.edges_to_adjacency(edges, edge_weights, num_hard)
        
        # -------------------------
        # 1. General Placement
        # -------------------------
        positions = self.initialize_spectral(
            adjacency_mat, half_w, half_h, canvas_w, canvas_h
        )

        positions = self.run_spring_system(
            positions,
            adjacency_mat,
            movable,
            half_w,
            half_h,
            canvas_w,
            canvas_h
        )

        # -------------------------
        # 2. Legalization
        # -------------------------
        positions = self.legalize_simple(
            positions, movable, sizes_np,
            half_w, half_h, canvas_w, canvas_h, num_hard
        )
        
        # -------------------------
        # Build full placement
        # -------------------------
        full_positions = benchmark.macro_positions.clone()
        full_positions[:num_hard] = torch.tensor(positions, dtype=torch.float32)
        
        return full_positions

    # =========================================================
    # SPRING SYSTEM
    # =========================================================
    def run_spring_system(self, positions, adjacency_mat, movable,
                          half_w, half_h, canvas_w, canvas_h,
                          steps=200, dt=0.05):

        N = positions.shape[0]
        pos = positions.copy()
        vel = np.zeros_like(pos)

        k_wire = 1.0
        k_rep = 0.01
        damping = 0.9

        for t in range(steps):
            F = np.zeros_like(pos)

            # -------------------------
            # Wire forces
            # -------------------------
            for i in range(N):
                for j in range(i+1, N):
                    w = adjacency_mat[i, j]
                    if w == 0:
                        continue

                    diff = pos[j] - pos[i]
                    F_ij = k_wire * w * diff

                    F[i] += F_ij
                    F[j] -= F_ij

            # -------------------------
            # Repulsion
            # -------------------------
            for i in range(N):
                for j in range(i+1, N):
                    diff = pos[j] - pos[i]
                    dist_sq = diff[0]**2 + diff[1]**2 + 1e-6

                    rep_force = k_rep * diff / dist_sq

                    F[i] -= rep_force
                    F[j] += rep_force

            # -------------------------
            # Boundary forces
            # -------------------------
            for i in range(N):
                if pos[i, 0] < half_w[i]:
                    F[i, 0] += (half_w[i] - pos[i, 0])
                if pos[i, 0] > canvas_w - half_w[i]:
                    F[i, 0] -= (pos[i, 0] - (canvas_w - half_w[i]))

                if pos[i, 1] < half_h[i]:
                    F[i, 1] += (half_h[i] - pos[i, 1])
                if pos[i, 1] > canvas_h - half_h[i]:
                    F[i, 1] -= (pos[i, 1] - (canvas_h - half_h[i]))

            # -------------------------
            # Update
            # -------------------------
            for i in range(N):
                if not movable[i]:
                    continue

                vel[i] = damping * vel[i] + dt * F[i]
                pos[i] += dt * vel[i]

            # Clamp
            pos[:, 0] = np.clip(pos[:, 0], half_w, canvas_w - half_w)
            pos[:, 1] = np.clip(pos[:, 1], half_h, canvas_h - half_h)

        return pos

    # =========================================================
    # SPECTRAL INIT
    # =========================================================
    def initialize_spectral(self, adjacency_mat, half_w, half_h, canvas_w, canvas_h):
        N = adjacency_mat.shape[0]

        if adjacency_mat.sum() == 0:
            positions = np.zeros((N, 2), dtype=np.float64)
            positions[:, 0] = np.random.uniform(half_w, canvas_w - half_w)
            positions[:, 1] = np.random.uniform(half_h, canvas_h - half_h)
            return positions

        W = adjacency_mat.astype(np.float64)
        D = np.diag(W.sum(axis=1))
        L = D - W

        L += 1e-6 * np.eye(N)

        eigvals, eigvecs = np.linalg.eigh(L)
        idx = np.argsort(eigvals)

        x_coords = eigvecs[:, idx[1]]
        y_coords = eigvecs[:, idx[2]]

        def normalize(arr, lo, hi):
            mn, mx = arr.min(), arr.max()
            if mx - mn < 1e-8:
                return np.full_like(arr, (lo + hi) / 2)
            return lo + (arr - mn) * (hi - lo) / (mx - mn)

        x_coords = normalize(x_coords, half_w, canvas_w - half_w)
        y_coords = normalize(y_coords, half_h, canvas_h - half_h)

        positions = np.stack([x_coords, y_coords], axis=1)

        positions += 1e-3 * np.random.randn(N, 2)

        positions[:, 0] = np.clip(positions[:, 0], half_w, canvas_w - half_w)
        positions[:, 1] = np.clip(positions[:, 1], half_h, canvas_h - half_h)

        return positions

    # =========================================================
    # LEGALIZER (UNCHANGED)
    # =========================================================
    def legalize_simple(self, positions, movable, sizes, half_w, half_h, canvas_w, canvas_h, num_hard):
        sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
        sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
        
        order = sorted(range(num_hard), key=lambda i: -sizes[i,0] * sizes[i,1])
        placed = np.zeros(num_hard, dtype=bool)
        legal = positions.copy()
        
        for i in order:
            if not movable[i]:
                placed[i] = True
                continue
            
            if placed.any():
                dx = np.abs(legal[i, 0] - legal[:, 0])
                dy = np.abs(legal[i, 1] - legal[:, 1])
                c = (dx < sep_x[i] + OVERLAP_THESHOLD) & (dy < sep_y[i] + OVERLAP_THESHOLD) & placed
                c[i] = False
                
                if not c.any():
                    placed[i] = True
                    continue
            
            step = max(sizes[i, 0], sizes[i, 1]) * LEGALIZER_STEP_SIZE
            best_p = legal[i].copy()
            best_d = float('inf')
            
            for r in range(1, 150):
                found = False
                
                for dxm in range(-r, r+1):
                    for dym in range (-r, r+1):
                        if abs(dxm) != r and abs(dym) != r: 
                            continue
                            
                        cx = np.clip(positions[i, 0] + dxm * step, half_w[i], canvas_w - half_w[i])
                        cy = np.clip(positions[i, 1] + dym * step, half_h[i], canvas_h - half_h[i])
                        
                        if placed.any():
                            dx = np.abs(cx-legal[:, 0])
                            dy = np.abs(cy - legal[:, 1])
                            c = (dx < sep_x[i] + OVERLAP_THESHOLD) & (dy < sep_y[i] + OVERLAP_THESHOLD) & placed
                            c[i] = False
                            
                            if c.any(): 
                                continue
                        
                        d = (cx - positions[i, 0]) ** 2 + (cy - positions[i, 1]) ** 2
                        
                        if d < best_d:
                            best_d = d; best_p = np.array([cx, cy])
                            found = True
                
                if found: 
                    break
            
            legal[i] = best_p
            placed[i] = True
    
        return legal

    # =========================================================
    # HELPERS
    # =========================================================
    def load_plc(self, name):
        root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
        
        if root.exists():
            _, plc = load_benchmark_from_dir(str(root))
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
            return torch.zeros(0, 2), torch.zeros(0)
        
        return (
            torch.tensor(list(edge_dict.keys())),
            torch.tensor(list(edge_dict.values()), dtype=torch.float32)
        )

    def edges_to_adjacency(self, edges, edge_weights, num_hard):
        A = np.zeros((num_hard, num_hard), dtype=np.float32)
        
        if len(edges) == 0:
            return A

        e = edges.numpy()
        w = edge_weights.numpy()
        
        A[e[:, 0], e[:, 1]] = w
        A[e[:, 1], e[:, 0]] = w
        
        return A