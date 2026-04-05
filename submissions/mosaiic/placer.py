"""
Macro-Optimized Spectral and AI-Informed placement for Integrated Circuits (MOSAIIC)

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
        
        # lets start real simple and just print off some interesting stats about the 
        name = benchmark.name
        height = benchmark.canvas_height
        width = benchmark.canvas_width
        
        num_macros = benchmark.num_macros
        num_hard_macros = benchmark.num_hard_macros
        num_soft_macros = benchmark.num_soft_macros
        
        num_nets = benchmark.num_nets
        
        # print(name, height, weight, num_macros, num_hard_macros, num_soft_macros, num_nets)
        print(f"Name: {name}")
        print(f"Dimensions: {height} um x {width} um")
        print(f"Total Macros: {num_macros}")
        print(f"\tHard Macros: {num_hard_macros}")
        print(f"\tSoft Macros: {num_soft_macros}")
        print(f"Total Nets: {num_nets}")
        
        return placement
    
