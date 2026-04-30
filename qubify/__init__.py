"""
qubify = make it QUBO

A constraint-to-QUBO compiler for quantum optimization.
Convert optimization problems into QUBO matrices for quantum solvers
(CIM, D-Wave, simulated annealing, etc.) — no hardware lock-in.

Quick start:
    >>> from qubify import qubify
    >>> problem = {
    ...     "variables": {"x": ("binary", (4,))},
    ...     "objective": [{"coeff": 1.0, "vars": [0, 1]}],
    ...     "constraints": [{"type": "one_hot", "vars": [0, 1, 2, 3]}],
    ... }
    >>> matrix, varmap = qubify(problem)

Or use presets:
    >>> from qubify.presets import tsp, maxcut
    >>> matrix, _ = tsp([[0,10,15],[10,0,35],[15,35,0]])
"""

from qubify.compiler import qubify
from qubify.expressions import QuboExpr, var, prod
from qubify.utils import qubo_to_ising
from qubify import constraints
from qubify import presets

__version__ = "0.2.0"
__all__ = [
    "qubify",
    "QuboExpr",
    "var",
    "prod",
    "qubo_to_ising",
    "constraints",
    "presets",
]
