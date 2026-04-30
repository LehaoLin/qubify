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


def tsp_decode(solution, var_map=None, n_cities=None):
    """Decode a binary solution vector from TSP QUBO into a tour order.

    Args:
        solution: 1D array/list of n² binary values from solver output.
                  solution[u*n + j] = 1 means city u is visited at position j.
        var_map: Optional var_map from tsp(). If provided, n_cities is inferred.
        n_cities: Number of cities. Required if var_map not provided.

    Returns:
        List of city indices in visiting order: [city_at_pos_0, city_at_pos_1, ...].

    Raises:
        ValueError: if the solution is not a valid tour.

    Example:
        >>> distances = [[0,10,15],[10,0,35],[15,35,0]]
        >>> matrix, vmap = tsp(distances)
        >>> # After solving...
        >>> sol = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        >>> tour = tsp_decode(sol, vmap)  # → [0, 1, 2]
    """
    if var_map is not None:
        n_cities = var_map.get("x", {}).get("shape", (0,))[0]
        if n_cities is None or n_cities == 0:
            if "x" in var_map:
                n_cities = int(np.sqrt(var_map["x"]["size"]))

    if n_cities is None:
        n_cities = int(np.sqrt(len(solution)))

    n = n_cities
    sol = np.array(solution, dtype=float).flatten()

    if len(sol) != n * n:
        raise ValueError(
            f"Solution length {len(sol)} doesn't match n²={n*n} for n={n}"
        )

    tour = [-1] * n
    for j in range(n):
        cities_at_pos = []
        for u in range(n):
            if sol[u * n + j] >= 0.5:
                cities_at_pos.append(u)
        if len(cities_at_pos) != 1:
            raise ValueError(
                f"Position {j}: expected exactly 1 city, got {len(cities_at_pos)}. "
                f"Solution may be infeasible (constraints violated)."
            )
        tour[j] = cities_at_pos[0]

    return tour


def tsp_qubo(distances):
    """Alias for tsp()."""
    return tsp(distances)
