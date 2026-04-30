"""Pre-built problem templates for common optimization problems.

Each function returns a tuple (QUBO matrix, variable map) ready for solving.
"""

from qubify.presets.tsp import tsp
from qubify.presets.maxcut import maxcut
from qubify.presets.knapsack import knapsack

__all__ = ["tsp", "maxcut", "knapsack"]
