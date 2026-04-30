"""
Tests for the qubify compiler.
"""

import numpy as np
from qubify import qubify


def test_simple_one_hot():
    """Compile a simple one-hot constrained problem."""
    problem = {
        "variables": {"x": ("binary", (4,))},
        "objective": [{"coeff": -1.0, "vars": [0]}],
        "constraints": [{"type": "one_hot", "vars": [0, 1, 2, 3]}],
    }
    matrix, varmap = qubify(problem)
    assert matrix.shape == (4, 4)
    assert varmap["x"]["size"] == 4
    assert varmap["x"]["shape"] == (4,)
    assert varmap["x"]["start"] == 0


def test_no_constraints():
    """Compiler should work with objective only (no constraints)."""
    problem = {
        "variables": {"x": ("binary", (3,))},
        "objective": [
            {"coeff": 1.0, "vars": [0, 1]},
            {"coeff": 2.0, "vars": [1, 2]},
        ],
        "constraints": [],
    }
    matrix, varmap = qubify(problem)
    # Q[0,1] = 1.0, Q[1,2] = 2.0
    assert matrix.shape == (3, 3)
    assert matrix[0, 1] == 1.0
    assert matrix[1, 2] == 2.0


def test_multiple_variable_blocks():
    """Compiler should handle multiple named variable blocks."""
    problem = {
        "variables": {
            "a": ("binary", (2,)),
            "b": ("binary", (3,)),
        },
        "objective": [
            {"coeff": 1.0, "vars": [0]},
            {"coeff": 2.0, "vars": [("b", 0)]},
        ],
        "constraints": [],
    }
    matrix, varmap = qubify(problem)
    # a[0] = index 0, a[1] = index 1
    # b[0] = index 2, b[1] = index 3, b[2] = index 4
    assert matrix.shape == (5, 5)
    assert varmap["a"]["start"] == 0
    assert varmap["a"]["size"] == 2
    assert varmap["b"]["start"] == 2
    assert varmap["b"]["size"] == 3
    # coeff 1.0 on var 0 (a[0]) → Q[0,0] = 1.0
    assert matrix[0, 0] == 1.0
    # coeff 2.0 on var ("b", 0) → index 2 → Q[2,2] = 2.0
    assert matrix[2, 2] == 2.0


def test_penalty_dominates_objective():
    """Constraint penalty should be larger than objective coefficients."""
    problem = {
        "variables": {"x": ("binary", (3,))},
        "objective": [
            {"coeff": -100.0, "vars": [0]},
            {"coeff": 50.0, "vars": [1]},
        ],
        "constraints": [{"type": "one_hot", "vars": [0, 1, 2]}],
    }
    matrix, _ = qubify(problem)
    # The penalty should scale with max objective coeff
    # max |coeff| = 100, max constraint size = 3 → penalty ≈ 2*100*3 = 600
    # Check that constraint-related off-diagonal terms are large
    assert abs(matrix[0, 1]) > 100  # should be dominated by penalty, not objective
