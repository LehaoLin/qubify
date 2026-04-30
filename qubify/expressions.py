"""
Lightweight QUBO expression algebra.
Zero external dependencies beyond numpy.

A QuboExpr represents:  Σ Q_{ij} x_i x_j + Σ c_i x_i + const
where x_i ∈ {0, 1} and Q is upper-triangular (i ≤ j).
"""

import numpy as np


class QuboExpr:
    """A quadratic expression over binary variables."""

    def __init__(self, quadratic=None, linear=None, const=0.0):
        """
        Args:
            quadratic: dict {(i,j): coeff} for i ≤ j (upper triangular)
            linear:    dict {i: coeff} for single-variable terms
            const:     float, constant offset
        """
        self.quadratic = dict(quadratic) if quadratic else {}
        self.linear = dict(linear) if linear else {}
        self.const = float(const)

    # ── arithmetic ────────────────────────────────────────────────

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return QuboExpr(self.quadratic, self.linear, self.const + other)
        q = _merge_dicts(self.quadratic, other.quadratic)
        l = _merge_dicts(self.linear, other.linear)
        return QuboExpr(q, l, self.const + other.const)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return QuboExpr(self.quadratic, self.linear, self.const - other)
        return self + (-1.0 * other)

    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            q = {k: v * scalar for k, v in self.quadratic.items()}
            l = {k: v * scalar for k, v in self.linear.items()}
            return QuboExpr(q, l, self.const * scalar)
        return NotImplemented

    def __rmul__(self, scalar):
        return self * scalar

    def __neg__(self):
        return self * -1.0

    # ── matrix conversion ─────────────────────────────────────────

    def to_matrix(self, n_vars: int) -> np.ndarray:
        """Convert to upper-triangular QUBO matrix of size n_vars × n_vars.

        QUBO form: x^T Q x  where Q is upper-triangular.
        The constant term is dropped (doesn't affect minimizer).
        """
        Q = np.zeros((n_vars, n_vars))
        for (i, j), coeff in self.quadratic.items():
            if i < n_vars and j < n_vars:
                Q[i, j] += coeff
        for i, coeff in self.linear.items():
            if i < n_vars:
                Q[i, i] += coeff
        return Q

    def __repr__(self):
        return f"QuboExpr(q={len(self.quadratic)} terms, l={len(self.linear)} terms, c={self.const})"


# ── helpers ───────────────────────────────────────────────────────

def _merge_dicts(a: dict, b: dict) -> dict:
    """Merge two coefficient dicts, summing overlapping keys."""
    result = dict(a)
    for k, v in b.items():
        result[k] = result.get(k, 0.0) + v
    return result


def var(i: int) -> QuboExpr:
    """Create a single binary variable expression x_i."""
    return QuboExpr(linear={i: 1.0})


def prod(i: int, j: int) -> QuboExpr:
    """Create a quadratic term x_i * x_j with i ≤ j."""
    if i > j:
        i, j = j, i
    return QuboExpr(quadratic={(i, j): 1.0})
