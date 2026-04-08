"""
Mass-on-Spring Analytical placer for Integrated Circuits (MOSAIC)

Team members: Arnav Patil
              [to be added...]

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
        movable = benchmark.get_movable_mask()[:num_hard].numpy()
        
        """ 1. General Placement """
        # TODO
        
        """ 2. Legalization """
        positions = benchmark.macro_positions[:num_hard].numpy().copy().astype(np.float64)
        positions = self.legalize_simple(positions, movable, sizes_np, half_w, 
                                         half_h, canvas_w, canvas_h, num_hard)
        
        # Build full placement by keeping soft macros at initial positions
        full_positions = benchmark.macro_positions.clone()
        full_positions[:num_hard] = torch.tensor(positions, dtype=torch.float32)
        
        """ 3. Detailed Placement """
        # TODO
        
        return full_positions

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
    
    
    
    
    
    