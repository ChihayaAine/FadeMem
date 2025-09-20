"""
Evaluation and Analysis Module for Enhanced Dual-Layer Memory Architecture

This module provides comprehensive evaluation tools for analyzing the performance
and behavior of the dual-layer memory system according to the methodology paper.

Key Components:
- Memory decay analysis and validation
- Half-life calculation verification  
- Importance scoring evaluation
- Conflict resolution assessment
- Fusion quality analysis
- System performance benchmarking
"""

from .memory_analyzer import MemorySystemAnalyzer
from .methodology_validator import MethodologyValidator
from .performance_benchmarks import PerformanceBenchmarks
from .visualization import MemoryVisualization

__all__ = [
    'MemorySystemAnalyzer',
    'MethodologyValidator', 
    'PerformanceBenchmarks',
    'MemoryVisualization'
]
