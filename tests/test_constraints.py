"""
Tests for qubify constraint templates.
"""

import numpy as np
from qubify.expressions import QuboExpr, var, prod
from qubify.constraints import (
    one_hot,
    cardinality,
    mutual_exclusive,
    implication,
    equality,
)


def test_one_hot_two_vars():
    """one_hot([0, 1]) should penalize (x0 + x1 - 1)²."""
    penalty = one_hot([0, 1])
    mat = penalty.to_matrix(2)

    # (x0 + x1 - 1)² = -x0 - x1 + 2x0x1 + 1
    # QUBO upper-triangular: Q[0,0]=-1, Q[1,1]=-1, Q[0,1]=2, Q[1,0]=0
    expected = np.array([[-1., 2.], [0., -1.]])
    assert np.allclose(mat, expected), f"Expected:\n{expected}\nGot:\n{mat}"


def test_one_hot_valid_state():
    """One-hot penalty should be 0 for valid state (exactly one 1)."""
    penalty = one_hot([0, 1, 2])
    mat = penalty.to_matrix(3)

    # Valid state: x=[1,0,0] → cost contribution = -1 + 1 = 0
    x = np.array([1, 0, 0])
    cost = x @ mat @ x
    assert abs(cost + penalty.const) < 1e-10  # const=1 offsets


def test_cardinality():
    """cardinality([0,1,2], 2) = (x0+x1+x2 - 2)²."""
    penalty = cardinality([0, 1, 2], 2)
    mat = penalty.to_matrix(3)

    # x=[1,1,0]: Σx=2 → cost = (2-2)² = 0
    x_valid = np.array([1, 1, 0])
    cost_valid = x_valid @ mat @ x_valid + penalty.const
    assert abs(cost_valid) < 1e-10

    # x=[1,0,0]: Σx=1 → cost = (1-2)² = 1
    x_invalid = np.array([1, 0, 0])
    cost_invalid = x_invalid @ mat @ x_invalid + penalty.const
    assert abs(cost_invalid - 1.0) < 1e-10


def test_mutual_exclusive():
    """x_a * x_b should be 1 only when both are 1."""
    penalty = mutual_exclusive(0, 1)
    mat = penalty.to_matrix(2)

    expected = np.array([[0., 1.], [0., 0.]])
    assert np.allclose(mat, expected)


def test_implication():
    """a ⇒ b: penalty = x_a - x_a*x_b."""
    penalty = implication(0, 1)
    mat = penalty.to_matrix(2)

    # a=1,b=0 → Q[0,0]=1, Q[0,1]=-1, Q[1,1]=0 → cost = 1
    x = np.array([1, 0])
    cost = x @ mat @ x
    assert abs(cost - 1.0) < 1e-10

    # a=1,b=1 → cost = 1 + (-1) = 0
    x = np.array([1, 1])
    cost = x @ mat @ x
    assert abs(cost) < 1e-10


def test_equality():
    """a=b: penalty = xa + xb - 2*xa*xb."""
    penalty = equality(0, 1)
    mat = penalty.to_matrix(2)

    # a=1,b=1 → cost = 1+1-2 = 0
    x = np.array([1, 1])
    cost = x @ mat @ x
    assert abs(cost) < 1e-10

    # a=0,b=1 → cost = 1
    x = np.array([0, 1])
    cost = x @ mat @ x
    assert abs(cost - 1.0) < 1e-10


def test_expression_arithmetic():
    """Test QuboExpr arithmetic operations."""
    e1 = 2 * var(0) + 3 * var(1)
    e2 = prod(0, 1) * 5
    combined = e1 + e2 + 10

    mat = combined.to_matrix(3)
    # linear: 2*x0 + 3*x1
    # quadratic: 5*x0*x1
    # const: 10
    expected = np.array([
        [2., 5., 0.],
        [0., 3., 0.],
        [0., 0., 0.],
    ])
    assert np.allclose(mat, expected)
    assert combined.const == 10.0


def test_to_matrix_no_effect_on_invalid_states():
    """to_matrix should handle variable indices beyond n_vars gracefully."""
    penalty = one_hot([0, 1, 2])
    mat = penalty.to_matrix(2)  # only 2 vars, but constraint references 3
    assert mat.shape == (2, 2)
