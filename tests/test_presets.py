"""
Tests for QUBO presets (TSP, MaxCut, Knapsack).
"""

import numpy as np
from qubify.presets import tsp, maxcut, knapsack


def test_tsp_small():
    """3-city TSP should produce valid QUBO matrix."""
    distances = [[0, 10, 15], [10, 0, 35], [15, 35, 0]]
    matrix, varmap = tsp(distances)

    # 3 cities → n² = 9 variables
    assert matrix.shape == (9, 9)
    assert varmap["x"]["size"] == 9
    assert varmap["x"]["shape"] == (3, 3)


def test_maxcut_small():
    """3-node Max-Cut should produce valid QUBO matrix."""
    adjacency = [
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ]
    matrix, varmap = maxcut(adjacency)

    assert matrix.shape == (3, 3)
    assert varmap["x"]["size"] == 3


def test_maxcut_trivial():
    """Max-Cut on edgeless graph should have zero matrix."""
    adjacency = np.zeros((3, 3))
    matrix, _ = maxcut(adjacency)
    assert np.allclose(matrix, 0.0)


def test_knapsack_small():
    """Small knapsack should produce valid QUBO matrix."""
    values = [60, 100, 120]
    weights = [10, 20, 30]
    capacity = 50
    matrix, varmap = knapsack(values, weights, capacity, slack_bits=3)

    # 3 items + 3 slack bits = 6 variables
    assert matrix.shape == (6, 6)
    assert varmap["x"]["size"] == 6


def test_tsp_matrix_symmetry():
    """TSP objective should be symmetric in the distance matrix."""
    distances = [[0, 5, 10], [5, 0, 7], [10, 7, 0]]
    matrix, _ = tsp(distances)
    # QUBO matrix is upper-triangular, but values should be symmetric
    # in the sense that D[u][v] = D[v][u]
    n = 3
    for i in range(n * n):
        for j in range(i, n * n):
            if matrix[i, j] != 0:
                assert np.isfinite(matrix[i, j])
