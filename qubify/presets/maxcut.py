"""
Max-Cut problem preset.

Given an n × n adjacency matrix of an undirected graph,
returns a QUBO matrix for the Max-Cut problem.

Variable encoding: x_i = 1 if node i is in partition A, 0 if in partition B.
Total variables: n.
"""

import numpy as np


def maxcut(adjacency):
    """Convert a Max-Cut instance to QUBO matrix.

    Args:
        adjacency: n × n adjacency or weight matrix (undirected).
                   adjacency[i][j] = weight of edge (i, j), 0 if no edge.

    Returns:
        (QUBO_matrix, var_map)
    """
    from qubify.compiler import qubify

    W = np.array(adjacency, dtype=float)
    n = W.shape[0]

    # Max-Cut QUBO formulation:
    # For each edge (i, j) with weight w, the contribution to the cut is
    # w * (x_i + x_j - 2*x_i*x_j) when x_i, x_j are in different sets.
    #
    # Since we minimize: minimize Σ_{i<j} -w_{ij}*(x_i + x_j - 2*x_i*x_j)
    #                 = minimize Σ_{i<j} w_{ij}*(2*x_i*x_j - x_i - x_j)
    #
    # Linear term for x_i: -Σ_{j≠i} w_{ij}
    # Quadratic term for (i, j): 2*w_{ij} * x_i * x_j

    objective = []
    for i in range(n):
        total_weight = 0.0
        for j in range(n):
            if i != j:
                total_weight += W[i, j]
        objective.append({"coeff": -total_weight, "vars": [i]})

    for i in range(n):
        for j in range(i + 1, n):
            if W[i, j] != 0:
                objective.append({"coeff": 2.0 * W[i, j], "vars": [i, j]})

    # Max-Cut has no constraints — it's a pure quadratic optimization
    problem = {
        "variables": {"x": ("binary", (n,))},
        "objective": objective,
        "constraints": [],
    }
    return qubify(problem)


def maxcut_decode(solution, adjacency=None):
    """Decode a binary solution vector from Max-Cut into two partitions.

    Args:
        solution: 1D array/list of n binary values from solver output.
                  solution[i] = 1 → node i in partition A.
                  solution[i] = 0 → node i in partition B.
        adjacency: Optional adjacency matrix. If provided, computes cut value.

    Returns:
        If adjacency is None:
            (partition_A, partition_B) — two lists of node indices.
        If adjacency is provided:
            (partition_A, partition_B, cut_value) — includes the cut weight.

    Example:
        >>> adj = [[0,1,0],[1,0,1],[0,1,0]]
        >>> matrix, _ = maxcut(adj)
        >>> sol = [1, 0, 1]
        >>> A, B = maxcut_decode(sol)  # A=[0,2], B=[1]
        >>> A, B, val = maxcut_decode(sol, adj)  # val=2
    """
    sol = np.array(solution, dtype=float).flatten()
    n = len(sol)

    partition_A = [int(i) for i in range(n) if sol[i] >= 0.5]
    partition_B = [int(i) for i in range(n) if sol[i] < 0.5]

    if adjacency is not None:
        adj = np.array(adjacency, dtype=float)
        cut_value = 0.0
        set_a = set(partition_A)
        for i in range(n):
            for j in range(i + 1, n):
                if (i in set_a) != (j in set_a):  # i and j are in different sets
                    cut_value += adj[i, j]
        return partition_A, partition_B, cut_value

    return partition_A, partition_B
