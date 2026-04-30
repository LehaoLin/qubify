"""
0/1 Knapsack problem preset.

Given n items with values and weights, and a capacity C,
returns a QUBO matrix for the knapsack problem.

Variable encoding: x_i = 1 if item i is selected, 0 otherwise.
Total variables: n.
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
    objective = []
    for i in range(n):
        objective.append({"coeff": -vals[i], "vars": [i]})

    # Capacity constraint: Σ w_i * x_i ≤ C
    # Use slack variable encoding: Σ w_i * x_i + Σ 2^k * s_k = C
    # Total variables: n items + slack_bits slack variables
    slack_start = n
    total_vars = n + slack_bits

    # Objective: add nothing for slack vars (they don't affect value)
    # They appear only in constraints.

    # Constraint: Σ w_i * x_i + Σ 2^k * s_k = C
    # Rewrite as cardinality over weighted terms isn't directly supported,
    # so we use equality: Σ_i w_i * x_i + Σ_k 2^k * s_k - C = 0
    #
    # Since our constraints work on binary variables directly, we need to
    # encode the weighted sum as a quadratic penalty:
    #   (Σ w_i*x_i + Σ 2^k*s_k - C)²
    #
    # Expand by hand:
    # Objective terms already in `objective`.
    # Cross terms between items and slack vars.
    # Diagonal terms from (w_i*x_i)² = w_i²*x_i  (since x_i² = x_i)
    # Diagonal terms from (2^k*s_k)² = 4^k*s_k
    # Cross terms: 2*w_i*2^k * x_i*s_k = 2^(k+1)*w_i * x_i*s_k
    # Cross terms between items: 2*w_i*w_j * x_i*x_j
    # Cross terms between slacks: 2*2^k*2^l * s_k*s_l = 2^(k+l+1) * s_k*s_l

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
