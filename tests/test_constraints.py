"""Tests for qubify.constraints — all 7 constraint types."""

import numpy as np
import pytest
from qubify.constraints import (
    one_hot, cardinality, at_least_one,
    mutual_exclusive, implication, equality, at_most_k,
)
from qubify.expressions import QuboExpr


def _eval_expr(expr: QuboExpr, assignment: list[int]) -> float:
    """Evaluate a QuboExpr on a binary assignment, including constant term."""
    n = max(max(expr.linear.keys(), default=0),
            max((k[1] for k in expr.quadratic.keys()), default=0)) + 1
    matrix = expr.to_matrix(n)
    x = np.array(assignment, dtype=float)
    return float(x.T @ matrix @ x + expr.const)


class TestOneHot:
    """Exactly one variable = 1."""

    def test_valid_states(self):
        penalty = one_hot([0, 1, 2])
        # Valid: exactly one 1
        assert _eval_expr(penalty, [1, 0, 0]) == pytest.approx(0.0)
        assert _eval_expr(penalty, [0, 1, 0]) == pytest.approx(0.0)
        assert _eval_expr(penalty, [0, 0, 1]) == pytest.approx(0.0)

    def test_invalid_all_zero(self):
        penalty = one_hot([0, 1, 2])
        assert _eval_expr(penalty, [0, 0, 0]) > 0.0

    def test_invalid_multiple_ones(self):
        penalty = one_hot([0, 1, 2])
        assert _eval_expr(penalty, [1, 1, 0]) > 0.0
        assert _eval_expr(penalty, [1, 1, 1]) > 0.0

    def test_single_element(self):
        penalty = one_hot([5])
        assert _eval_expr(penalty, [0, 0, 0, 0, 0, 1]) == pytest.approx(0.0)

    def test_empty(self):
        penalty = one_hot([])
        assert penalty.const == 1.0


class TestCardinality:
    """Exactly k variables = 1."""

    def test_k_equals_one(self):
        # cardinality(vars, 1) should equal one_hot(vars)
        penalty_c = cardinality([0, 1, 2], 1)
        penalty_h = one_hot([0, 1, 2])
        for assignment in [[1,0,0], [0,1,0], [0,0,1]]:
            assert _eval_expr(penalty_c, assignment) == pytest.approx(0.0)
            assert _eval_expr(penalty_c, assignment) == pytest.approx(
                _eval_expr(penalty_h, assignment)
            )

    def test_k_equals_two(self):
        penalty = cardinality([0, 1, 2], 2)
        assert _eval_expr(penalty, [1, 1, 0]) == pytest.approx(0.0)
        assert _eval_expr(penalty, [1, 0, 1]) == pytest.approx(0.0)
        assert _eval_expr(penalty, [0, 1, 1]) == pytest.approx(0.0)
        assert _eval_expr(penalty, [1, 1, 1]) > 0.0
        assert _eval_expr(penalty, [1, 0, 0]) > 0.0

    def test_k_equals_zero(self):
        penalty = cardinality([0, 1], 0)
        assert _eval_expr(penalty, [0, 0]) == pytest.approx(0.0)
        assert _eval_expr(penalty, [1, 0]) > 0.0


class TestAtLeastOne:
    """At least one variable = 1.

    Note: currently implemented as cardinality(vars, 1) — an approximation
    that requires exactly 1, not ≥1. Valid for assignment problems where
    "at most one" also naturally holds.
    """

    def test_small_set(self):
        penalty = at_least_one([0, 1, 2])
        # cardinality([0,1,2], 1): exactly 1 is valid
        assert _eval_expr(penalty, [1, 0, 0]) == pytest.approx(0.0)
        assert _eval_expr(penalty, [0, 1, 0]) == pytest.approx(0.0)
        assert _eval_expr(penalty, [0, 0, 1]) == pytest.approx(0.0)
        # 0 or 2+: penalized
        assert _eval_expr(penalty, [0, 0, 0]) > 0.0
        assert _eval_expr(penalty, [0, 1, 1]) > 0.0

    def test_single_var(self):
        penalty = at_least_one([0])
        assert _eval_expr(penalty, [1]) == pytest.approx(0.0)
        assert _eval_expr(penalty, [0]) > 0.0


class TestMutualExclusive:
    """Two variables cannot both be 1."""

    def test_both_zero(self):
        penalty = mutual_exclusive(0, 3)
        assert _eval_expr(penalty, [0, 0, 0, 0]) == pytest.approx(0.0)

    def test_one_zero(self):
        penalty = mutual_exclusive(0, 1)
        assert _eval_expr(penalty, [1, 0]) == pytest.approx(0.0)
        assert _eval_expr(penalty, [0, 1]) == pytest.approx(0.0)

    def test_both_one(self):
        penalty = mutual_exclusive(0, 1)
        assert _eval_expr(penalty, [1, 1]) == pytest.approx(1.0)  # x0 * x1 = 1


class TestImplication:
    """a ⇒ b: if a=1 then b=1."""

    def test_valid(self):
        penalty = implication(0, 1)
        assert _eval_expr(penalty, [0, 0]) == pytest.approx(0.0)
        assert _eval_expr(penalty, [0, 1]) == pytest.approx(0.0)
        assert _eval_expr(penalty, [1, 1]) == pytest.approx(0.0)

    def test_violated(self):
        penalty = implication(0, 1)
        assert _eval_expr(penalty, [1, 0]) == pytest.approx(1.0)


class TestEquality:
    """a = b."""

    def test_valid(self):
        penalty = equality(0, 1)
        assert _eval_expr(penalty, [0, 0]) == pytest.approx(0.0)
        assert _eval_expr(penalty, [1, 1]) == pytest.approx(0.0)

    def test_violated(self):
        penalty = equality(0, 1)
        assert _eval_expr(penalty, [1, 0]) == pytest.approx(1.0)
        assert _eval_expr(penalty, [0, 1]) == pytest.approx(1.0)


class TestAtMostK:
    """At most k variables = 1.

    Note: currently implemented as cardinality(vars, k), requiring exactly k.
    This is an approximation — for true "at most", slack variables
    are needed. The implementation correctly penalizes states with
    more than k variables, but also penalizes states with fewer than k.
    """

    def test_at_most_one(self):
        penalty = at_most_k([0, 1, 2], 1)
        # cardinality([0,1,2], 1): exactly 1 is valid
        assert _eval_expr(penalty, [1, 0, 0]) == pytest.approx(0.0)
        assert _eval_expr(penalty, [0, 1, 0]) == pytest.approx(0.0)
        # 0 or 2+: penalized (cardinality limitation)
        assert _eval_expr(penalty, [0, 0, 0]) > 0.0
        assert _eval_expr(penalty, [1, 1, 0]) > 0.0

    def test_at_most_two(self):
        penalty = at_most_k([0, 1, 2], 2)
        # cardinality([0,1,2], 2): exactly 2 is valid
        assert _eval_expr(penalty, [1, 1, 0]) == pytest.approx(0.0)
        assert _eval_expr(penalty, [1, 0, 1]) == pytest.approx(0.0)
        assert _eval_expr(penalty, [0, 1, 1]) == pytest.approx(0.0)
        # 0, 1, or 3: penalized
        assert _eval_expr(penalty, [0, 0, 0]) > 0.0
        assert _eval_expr(penalty, [1, 0, 0]) > 0.0
        assert _eval_expr(penalty, [1, 1, 1]) > 0.0
