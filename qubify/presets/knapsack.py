"""
0/1 Knapsack problem preset.

Given n items with values and weights, and a capacity C,
returns a QUBO matrix for the knapsack problem.

Variable encoding: x_i = 1 if item i is selected, 0 otherwise.
Slack variables: s_k for k = 0..slack_bits-1 encode capacity remainder.
Total variables: n + slack_bits.
"""

import numpy as np


def knapsack(values, weights, capacity, slack_bits=4):
    """Convert a 0/1 knapsack instance to QUBO matrix.

    Args:
        values: list of n item values
        weights: list of n item weights
        capacity: maximum total weight
        slack_bits: number of slack binary variables for capacity encoding
                    (default 4, supports capacity up to 2^slack_bits - 1)

    Returns:
        (QUBO_matrix, var_map)
    """
    from qubify.compiler import qubify

    vals = np.array(values, dtype=float)
    wts = np.array(weights, dtype=float)
    n = len(vals)
    C = float(capacity)

    # Objective: maximize Σ v_i * x_i  →  minimize -Σ v_i * x_i
    # Capacity baked into objective via penalty expansion
    slack_start = n
    total_vars = n + slack_bits

    objective = []
    for i in range(n):
        objective.append({"coeff": -vals[i], "vars": [i]})

    # Capacity constraint as quadratic penalty: (Σ w_i*x_i + Σ 2^k*s_k - C)²
    for i in range(n):
        # Diagonal: w_i² * x_i
        objective.append({"coeff": wts[i] ** 2, "vars": [i]})
        # Cross terms between items
        for j in range(i + 1, n):
            objective.append({
                "coeff": 2.0 * wts[i] * wts[j],
                "vars": [i, j],
            })
        # Cross terms between items and slack
        for k in range(slack_bits):
            objective.append({
                "coeff": 2.0 * (2 ** k) * wts[i],
                "vars": [i, slack_start + k],
            })

    for k in range(slack_bits):
        # Diagonal: 4^k * s_k
        objective.append({
            "coeff": 4.0 ** k,
            "vars": [slack_start + k],
        })
        # Cross terms between slacks
        for l in range(k + 1, slack_bits):
            objective.append({
                "coeff": 2.0 ** (k + l + 1),
                "vars": [slack_start + k, slack_start + l],
            })

    # Linear term from -2C * Σ w_i*x_i
    for i in range(n):
        idx = next(
            j for j, t in enumerate(objective) if t["vars"] == [i]
        )
        objective[idx]["coeff"] -= 2.0 * C * wts[i]

    # Linear term from -2C * Σ 2^k*s_k
    for k in range(slack_bits):
        idx = next(
            j for j, t in enumerate(objective)
            if t["vars"] == [slack_start + k]
        )
        objective[idx]["coeff"] -= 2.0 * C * (2 ** k)

    # No explicit constraint list — the capacity is baked into the objective
    problem = {
        "variables": {"x": ("binary", (total_vars,))},
        "objective": objective,
        "constraints": [],
    }
    return qubify(problem)


def knapsack_decode(solution, slack_bits=4):
    """Decode a binary solution vector from Knapsack QUBO.

    Args:
        solution: 1D array/list of n + slack_bits binary values.
                  First n values are item selections (x_i).
                  Remaining values are slack variables (ignored in decode).
        slack_bits: Number of slack bits used in the encoding (default 4).

    Returns:
        dict with:
            - "selected": list of indices of selected items
            - "total_value": sum of values of selected items
            - "total_weight": sum of weights of selected items

    Example:
        >>> sol = [1, 0, 1, 0, 0, 0, 0]  # n=3, slack_bits=4
        >>> result = knapsack_decode(sol, slack_bits=4)
        >>> result["selected"]  # → [0, 2]
    """
    sol = np.array(solution, dtype=float).flatten()
    n = len(sol) - slack_bits

    selected = [int(i) for i in range(n) if sol[i] >= 0.5]

    return {
        "selected": selected,
        "total_value": None,  # caller must supply values to compute
        "total_weight": None,  # caller must supply weights to compute
    }


def knapsack_decode_full(solution, values, weights, slack_bits=4):
    """Decode a binary solution with value/weight computation.

    Args:
        solution: 1D array of n + slack_bits binary values.
        values: list of item values.
        weights: list of item weights.
        slack_bits: Number of slack bits.

    Returns:
        dict with "selected", "total_value", "total_weight", "capacity_used".
    """
    result = knapsack_decode(solution, slack_bits)
    vals = np.array(values, dtype=float)
    wts = np.array(weights, dtype=float)

    idxs = result["selected"]
    result["total_value"] = float(np.sum(vals[idxs]))
    result["total_weight"] = float(np.sum(wts[idxs]))
    return result
