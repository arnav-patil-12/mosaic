"""
Mass-on-Spring Analytical placer for Integrated Circuits (MOSAIC)

Usage:
    uv run evaluate submissions/mosaiic/placer.py
    uv run evaluate submissions/mosaiic/placer.py --all
    uv run evaluate submissions/mosaiic/placer.py -b ibm03
"""

import torch

from macro_place.benchmark import Benchmark

class MOSAIICPlacer:
    """
    Something needs to go here
    """
    
    def place(self, benchmark: Benchmark) -> torch.Tensor:
        placement = benchmark.macro_positions.clone()
        
        
        
        return self.legalizer(benchmark=benchmark, placement=placement)
    
    def legalizer(self, benchmark: Benchmark, placement: torch.Tensor) -> torch.Tensor:
        # only move hard macros, soft stay at initial position
        movable = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        movable_indices = torch.where(movable)[0].tolist()
        
        sizes = benchmark.macro_sizes
        width = benchmark.canvas_width
        height = benchmark.canvas_height
        
        movable_indices.sort(key=lambda i: -sizes[i, 1].item())
        
        gap = 0.001
        cursor_x = 0.0
        cursor_y = 0.0
        row_height = 0.0
        
        for i in movable_indices:
            w = sizes[i, 0].item()
            h = sizes[i, 1].item()
            
            # if the macro doesn't fit, start a new row
            if cursor_x + w > width:
                cursor_x = 0.0
                cursor_y += row_height + gap
                row_height = 0.0
            
            # check if out of vertical space
            if cursor_y + h > height:
                placement[i, 0] = w / 2
                placement[i, 1] = h / 2
                continue
                
            placement[i, 0] = cursor_x + w / 2
            placement[i, 1] = cursor_y + h / 2
            
            cursor_x += w + gap
            row_height = max(row_height, h)
        
        return placement
    
