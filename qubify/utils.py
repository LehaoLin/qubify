"""
Utility functions for qubify.

Includes QUBO ↔ Ising conversion and other matrix helpers.
Zero Kaiwu SDK dependencies — pure numpy math.
"""

import numpy as np


def qubo_to_ising(qubo_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Convert a QUBO matrix to Ising form.

    Transform: min x^T Q x (x ∈ {0,1}) → min s^T J s + h^T s (s ∈ {-1,1})

    Uses the standard mapping: x_i = (1 - s_i) / 2

    Args:
        qubo_matrix: n×n upper-triangular QUBO matrix.

    Returns:
        (J, h, offset) where:
            J: n×n Ising coupling matrix (upper triangular, J_{ii} = 0)
            h: n-vector of linear biases
            offset: constant energy shift (add to compare energies)

    Example:
        >>> Q = np.array([[1, -2], [0, 3]])
        >>> J, h, offset = qubo_to_ising(Q)
        >>> # J → coupling terms, h → biases, offset → constant shift
    """
    n = qubo_matrix.shape[0]

    # Symmetrize: S = (Q + Q^T) / 2
    S = (qubo_matrix + qubo_matrix.T) / 2.0

    # Build J (upper triangular, no diagonal):
    #   J_{ij} = S_{ij} / 2   for i < j
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            J[i, j] = S[i, j] / 2.0

    # Build h:
    #   h_i = -½ Σ_j S_{ij}
    h = -0.5 * np.sum(S, axis=1)

    # Offset = (Σ_{i,j} S_{ij} + Σ_i S_{ii}) / 4   (constant energy shift)
    offset = (np.sum(S) + np.trace(S)) / 4.0

    return J, h, offset
