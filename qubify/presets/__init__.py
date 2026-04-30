"""Pre-built problem templates for common optimization problems.

Each function returns a tuple (QUBO matrix, variable map) ready for solving.
Each preset also provides a decode function for interpreting solver output.
"""

from qubify.presets.tsp import tsp, tsp_decode
from qubify.presets.maxcut import maxcut, maxcut_decode
from qubify.presets.knapsack import knapsack, knapsack_decode, knapsack_decode_full

__all__ = [
    "tsp", "tsp_decode",
    "maxcut", "maxcut_decode",
    "knapsack", "knapsack_decode", "knapsack_decode_full",
]
