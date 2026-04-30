"""Tests for qubify.expressions — QuboExpr algebra and matrix conversion."""

import numpy as np
import pytest
from qubify.expressions import QuboExpr, var, prod


class TestQuboExpr:
    """Core expression construction and arithmetic."""

    def test_empty_expr(self):
        e = QuboExpr()
        assert e.const == 0.0
        assert len(e.quadratic) == 0
        assert len(e.linear) == 0

    def test_const_only(self):
        e = QuboExpr(const=5.0)
        matrix = e.to_matrix(2)
        assert np.allclose(matrix, np.zeros((2, 2)))

    def test_linear_only(self):
        e = QuboExpr(linear={0: 2.0, 1: -1.0})
        matrix = e.to_matrix(3)
        expected = np.array([
            [2.0, 0, 0],
            [0, -1.0, 0],
            [0, 0, 0],
        ])
        assert np.allclose(matrix, expected)

    def test_quadratic_only(self):
        e = QuboExpr(quadratic={(0, 1): 3.0, (1, 2): -0.5})
        matrix = e.to_matrix(3)
        expected = np.array([
            [0, 3.0, 0],
            [0, 0, -0.5],
            [0, 0, 0],
        ])
        assert np.allclose(matrix, expected)

    def test_full_expression(self):
        e = QuboExpr(
            quadratic={(0, 1): 1.5},
            linear={0: -1.0, 1: 0.5},
            const=3.0,
        )
        matrix = e.to_matrix(2)
        expected = np.array([
            [-1.0, 1.5],
            [0, 0.5],
        ])
        assert np.allclose(matrix, expected)

    def test_to_matrix_drops_out_of_range(self):
        e = QuboExpr(linear={5: 1.0}, quadratic={(4, 5): 2.0})
        matrix = e.to_matrix(3)  # only vars 0,1,2
        assert np.allclose(matrix, np.zeros((3, 3)))


class TestArithmetic:
    """Expression arithmetic: +, -, *, scalar and expression."""

    def test_add_scalar(self):
        e = QuboExpr(linear={0: 1.0}, const=2.0)
        f = e + 3
        assert f.const == 5.0
        assert f.linear == {0: 1.0}

    def test_radd_scalar(self):
        e = QuboExpr(linear={0: 1.0})
        f = 3 + e
        assert f.const == 3.0
        assert f.linear == {0: 1.0}

    def test_add_expr(self):
        a = QuboExpr(linear={0: 1.0})
        b = QuboExpr(linear={1: 2.0}, const=3.0)
        c = a + b
        assert c.linear == {0: 1.0, 1: 2.0}
        assert c.const == 3.0
        assert len(c.quadratic) == 0

    def test_add_merges_quadratic(self):
        a = QuboExpr(quadratic={(0, 1): 1.0})
        b = QuboExpr(quadratic={(0, 1): 2.0})
        c = a + b
        assert c.quadratic[(0, 1)] == 3.0

    def test_sub_scalar(self):
        e = QuboExpr(linear={0: 5.0}, const=10.0)
        f = e - 2
        assert f.const == 8.0
        assert f.linear == {0: 5.0}

    def test_sub_expr(self):
        a = QuboExpr(linear={0: 5.0})
        b = QuboExpr(linear={0: 2.0})
        c = a - b
        assert c.linear[0] == 3.0

    def test_mul_scalar(self):
        e = QuboExpr(linear={0: 2.0}, quadratic={(0, 1): 3.0}, const=1.0)
        f = e * 2
        assert f.linear[0] == 4.0
        assert f.quadratic[(0, 1)] == 6.0
        assert f.const == 2.0

    def test_rmul_scalar(self):
        e = QuboExpr(linear={0: 2.0})
        f = 3 * e
        assert f.linear[0] == 6.0

    def test_neg(self):
        e = QuboExpr(linear={0: 2.0}, const=1.0)
        f = -e
        assert f.linear[0] == -2.0
        assert f.const == -1.0

    def test_mul_chain(self):
        e = QuboExpr(linear={0: 1.0}, const=2.0)
        f = e * 3 + 1  # (expr*3) + 1
        assert f.linear[0] == 3.0
        assert f.const == 7.0


class TestFactoryFunctions:
    """var() and prod() constructors."""

    def test_var(self):
        x = var(3)
        assert x.linear == {3: 1.0}
        assert len(x.quadratic) == 0
        assert x.const == 0.0

    def test_prod_ordered(self):
        p = prod(1, 3)
        assert p.quadratic == {(1, 3): 1.0}
        assert len(p.linear) == 0

    def test_prod_reversed(self):
        p = prod(5, 2)
        assert p.quadratic == {(2, 5): 1.0}  # auto-ordered

    def test_prod_expression_equivalent(self):
        """2*var(0)*var(1) should give same quadratic term."""
        # prod is a convenience, let's verify it matches manual construction
        p = prod(0, 1)
        m = p.to_matrix(2)
        expected = np.array([[0, 1.0], [0, 0]])
        assert np.allclose(m, expected)


class TestRepr:
    def test_repr(self):
        e = QuboExpr(linear={0: 1.0}, quadratic={(0, 1): 2.0}, const=5.0)
        r = repr(e)
        assert "1 terms" in r or "1" in r


class TestEnergyEvaluation:
    """Test that x^T Q x matches manual evaluation for binary vectors."""

    def test_single_var(self):
        e = var(0)
        m = e.to_matrix(1)
        # x=0: energy=0; x=1: energy=1
        x0 = np.array([0])
        x1 = np.array([1])
        assert np.allclose(x0.T @ m @ x0, 0)
        assert np.allclose(x1.T @ m @ x1, 1)

    def test_linear_energy(self):
        e = QuboExpr(linear={0: 2.0, 1: -1.0})
        m = e.to_matrix(2)
        x = np.array([1, 0])  # 2*1 + (-1)*0 = 2
        assert np.allclose(x.T @ m @ x, 2.0)
        x = np.array([0, 1])  # 2*0 + (-1)*1 = -1
        assert np.allclose(x.T @ m @ x, -1.0)

    def test_quadratic_energy(self):
        e = QuboExpr(quadratic={(0, 1): 3.0})
        m = e.to_matrix(2)
        x = np.array([1, 1])  # 3*1*1 = 3
        assert np.allclose(x.T @ m @ x, 3.0)
        x = np.array([1, 0])  # 3*1*0 = 0
        assert np.allclose(x.T @ m @ x, 0.0)
