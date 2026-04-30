"""
TSP (Traveling Salesman Problem) preset.

Given an n × n distance matrix, returns a QUBO matrix for the TSP.

Variable encoding: x_{u,j} = 1 if city u is visited at position j.
Total variables: n².
"""

import numpy as np


def tsp(distances):
    """Convert a TSP instance to QUBO matrix.

    Args:
        distances: n × n distance matrix (numpy array or list of lists).
                   distances[i][j] = distance from city i to city j.

    Returns:
        (QUBO_matrix, var_map) where var_map maps "x" → flat index range.
    """
    from qubify.compiler import qubify

    D = np.array(distances, dtype=float)
    n = D.shape[0]

    # Build objective: min Σ D[u][v] * x[u][j] * x[v][(j+1)%n]
    objective = []
    for u in range(n):
        for v in range(n):
            if u != v and D[u, v] > 0:
                for j in range(n):
                    next_j = (j + 1) % n
                    i_flat = u * n + j
                    j_flat = v * n + next_j
                    objective.append({
                        "coeff": D[u, v],
                        "vars": [i_flat, j_flat],
                    })

    # Constraints:
    # 1. Each city appears exactly once: for each u, Σ_j x[u][j] = 1
    # 2. Each position holds exactly one city: for each j, Σ_u x[u][j] = 1
    constraints = []
    for u in range(n):
        constraints.append({
            "type": "one_hot",
            "vars": [u * n + j for j in range(n)],
        })
    for j in range(n):
        constraints.append({
            "type": "one_hot",
            "vars": [u * n + j for u in range(n)],
        })

    problem = {
        "variables": {"x": ("binary", (n, n))},
        "objective": objective,
        "constraints": constraints,
    }
    return qubify(problem)


def tsp_qubo(distances):
    """Alias for tsp()."""
    return tsp(distances)
