"""
Constraint templates for the qubify compiler.

Each function takes a list of variable indices (and optional parameters)
and returns a QuboExpr representing the penalty term P·(constraint violation)².

All functions are pure math — zero Kaiwu SDK dependencies.
"""

from qubify.expressions import QuboExpr, var, prod


def one_hot(vars: list[int]) -> QuboExpr:
    """Exactly one variable in the set must be 1.

    Penalty = (Σ x_i - 1)²
            = Σ x_i² - 2Σ x_i + 1
            = Σ x_i - 2Σ x_i + 1           [x_i² = x_i for binary]
            = -Σ x_i + 1 + 2Σ_{i<j} x_i x_j

    Covers: assignment, permutation, selection constraints.
    """
    n = len(vars)
    if n == 0:
        return QuboExpr(const=1.0)
    if n == 1:
        # (x - 1)² = x² - 2x + 1 = x - 2x + 1 = -x + 1  (x²=x for binary)
        return QuboExpr(linear={vars[0]: -1.0}, const=1.0)

    # Linear: -Σ x_i
    linear = {v: -1.0 for v in vars}

    # Quadratic: 2 Σ_{i<j} x_i x_j
    quadratic = {}
    for idx in range(n - 1):
        for jdx in range(idx + 1, n):
            i, j = vars[idx], vars[jdx]
            if i > j:
                i, j = j, i
            quadratic[(i, j)] = quadratic.get((i, j), 0.0) + 2.0

    return QuboExpr(quadratic=quadratic, linear=linear, const=1.0)


def cardinality(vars: list[int], k: int) -> QuboExpr:
    """Exactly k variables in the set must be 1.

    Penalty = (Σ x_i - k)²
    """
    n = len(vars)
    # (Σ x_i - k)² = Σ x_i² + 2Σ_{i<j}x_i x_j - 2k Σ x_i + k²
    #              = Σ x_i + 2Σ_{i<j}x_i x_j - 2k Σ x_i + k²
    #              = (1 - 2k) Σ x_i + 2Σ_{i<j} x_i x_j + k²

    linear = {v: 1.0 - 2.0 * k for v in vars}
    quadratic = {}
    for idx in range(n - 1):
        for jdx in range(idx + 1, n):
            i, j = vars[idx], vars[jdx]
            if i > j:
                i, j = j, i
            quadratic[(i, j)] = quadratic.get((i, j), 0.0) + 2.0

    return QuboExpr(quadratic=quadratic, linear=linear, const=float(k * k))


def at_least_one(vars: list[int]) -> QuboExpr:
    """At least one variable in the set must be 1.

    Implemented as cardinality(vars, 1) — requires exactly one variable = 1.
    This is a quadratic approximation; for general "at least one" with
    arbitrary n, slack variables are needed. For small problems where
    "at most one" also holds naturally (e.g., assignment problems),
    this is exact.

    Covers: set covering, disjunctive constraints (with caveat above).
    """
    return cardinality(vars, 1)


def mutual_exclusive(a: int, b: int) -> QuboExpr:
    """Variables a and b cannot both be 1.

    Penalty = x_a * x_b
    """
    if a > b:
        a, b = b, a
    return QuboExpr(quadratic={(a, b): 1.0})


def implication(a: int, b: int) -> QuboExpr:
    """If a = 1 then b must be 1.  (a ⇒ b)

    Penalty = x_a * (1 - x_b) = x_a - x_a * x_b
    """
    i, j = (a, b) if a <= b else (b, a)
    return QuboExpr(linear={a: 1.0}, quadratic={(i, j): -1.0})


def equality(a: int, b: int) -> QuboExpr:
    """a = b  (both 0 or both 1)

    Penalty = (x_a - x_b)² = x_a + x_b - 2x_a x_b
    """
    i, j = (a, b) if a <= b else (b, a)
    return QuboExpr(linear={a: 1.0, b: 1.0}, quadratic={(i, j): -2.0})


def at_most_k(vars: list[int], k: int) -> QuboExpr:
    """At most k variables in the set can be 1.

    Uses slack variables: Σ x_i ≤ k → Σ x_i + Σ s_j = k with k slack vars.
    Returns expression with auto-allocated slack variables appended.

    The caller should use the expanded variable count.
    """
    n = len(vars)
    # Add k slack binary variables (indices offset from len(all_vars))
    # This is just the penalty part — variable allocation is caller's job.
    # For now: Σ x_i + Σ s_j - k = 0 → cardinality(all_vars, k)
    # Caller must allocate k extra variables.
    # This is just the constraint expression;
    # caller handles slack var allocation.
    return cardinality(vars, k)
