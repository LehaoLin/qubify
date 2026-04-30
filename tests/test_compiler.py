"""Tests for qubify.compiler — validation, penalty, and the qubify() pipeline."""

import numpy as np
import pytest
from qubify.compiler import qubify, _validate_problem


# ── helpers ────────────────────────────────────────────────────────

def _make_simple_problem(**overrides):
    """Create a minimal valid problem dict, with optional overrides."""
    base = {
        "variables": {"x": ("binary", (3,))},
        "objective": [
            {"coeff": 1.0, "vars": [0, 1]},
        ],
        "constraints": [],
    }
    base.update(overrides)
    return base


# ── validation tests ───────────────────────────────────────────────

class TestValidationTopLevel:
    """Top-level schema validation."""

    def test_not_a_dict(self):
        with pytest.raises(ValueError, match="must be a dict"):
            _validate_problem([1, 2, 3])

    def test_missing_variables(self):
        with pytest.raises(ValueError, match="Missing required keys"):
            _validate_problem({"objective": [], "constraints": []})

    def test_missing_objective(self):
        with pytest.raises(ValueError, match="Missing required keys"):
            _validate_problem({"variables": {}, "constraints": []})

    def test_empty_problem(self):
        # Should pass: all keys present, even if empty
        _validate_problem({"variables": {}, "objective": [], "constraints": []})


class TestValidationVariables:
    """Variable declaration validation."""

    def test_string_binary(self):
        _validate_problem(_make_simple_problem(variables={"x": "binary"}))

    def test_tuple_binary(self):
        _validate_problem(_make_simple_problem(variables={"x": ("binary", (4,))}))

    def test_tuple_int_shape(self):
        _validate_problem(_make_simple_problem(variables={"x": ("binary", 5)}))

    def test_bad_type_str(self):
        with pytest.raises(ValueError, match="must be 'binary'"):
            _validate_problem(_make_simple_problem(variables={"x": "float"}))

    def test_bad_type_tuple(self):
        with pytest.raises(ValueError, match="must be \\('binary', shape\\)"):
            _validate_problem(_make_simple_problem(variables={"x": ("float", (3,))}))

    def test_negative_dim(self):
        with pytest.raises(ValueError, match="must be positive"):
            _validate_problem(_make_simple_problem(variables={"x": ("binary", (-3,))}))


class TestValidationObjective:
    """Objective term validation."""

    def test_not_a_list(self):
        with pytest.raises(ValueError, match="must be a list"):
            _validate_problem(_make_simple_problem(objective="not_list"))

    def test_missing_coeff(self):
        with pytest.raises(ValueError, match="missing required key 'coeff'"):
            _validate_problem(_make_simple_problem(objective=[{"vars": [0]}]))

    def test_missing_vars(self):
        with pytest.raises(ValueError, match="missing required key 'vars'"):
            _validate_problem(_make_simple_problem(objective=[{"coeff": 1.0}]))

    def test_bad_var_length(self):
        with pytest.raises(ValueError, match="list of 1 or 2"):
            _validate_problem(_make_simple_problem(
                objective=[{"coeff": 1.0, "vars": [0, 1, 2]}]
            ))


class TestValidationConstraints:
    """Constraint validation."""

    def test_not_a_list(self):
        with pytest.raises(ValueError, match="must be a list"):
            _validate_problem(_make_simple_problem(constraints="bad"))

    def test_unknown_type(self):
        with pytest.raises(ValueError, match="unknown type"):
            _validate_problem(_make_simple_problem(
                constraints=[{"type": "not_real", "vars": [0]}]
            ))

    def test_missing_type(self):
        with pytest.raises(ValueError, match="missing required key 'type'"):
            _validate_problem(_make_simple_problem(
                constraints=[{"vars": [0]}]
            ))

    def test_missing_vars(self):
        with pytest.raises(ValueError, match="missing required key 'vars'"):
            _validate_problem(_make_simple_problem(
                constraints=[{"type": "one_hot"}]
            ))

    def test_cardinality_missing_rhs(self):
        with pytest.raises(ValueError, match="requires 'rhs'"):
            _validate_problem(_make_simple_problem(
                constraints=[{"type": "cardinality", "vars": [0, 1]}]
            ))


class TestValidationPenalty:
    """Optional penalty validation."""

    def test_valid_penalty(self):
        _validate_problem(_make_simple_problem(penalty=10.0))

    def test_zero_penalty(self):
        with pytest.raises(ValueError, match="must be a positive number"):
            _validate_problem(_make_simple_problem(penalty=0))

    def test_negative_penalty(self):
        with pytest.raises(ValueError, match="must be a positive number"):
            _validate_problem(_make_simple_problem(penalty=-5.0))

    def test_string_penalty(self):
        with pytest.raises(ValueError, match="must be a positive number"):
            _validate_problem(_make_simple_problem(penalty="high"))


# ── pipeline tests ─────────────────────────────────────────────────

class TestQubifyPipeline:
    """Test the full qubify() compilation pipeline."""

    def test_simple_objective(self):
        problem = {
            "variables": {"x": ("binary", (2,))},
            "objective": [
                {"coeff": 1.0, "vars": [0]},
                {"coeff": 2.0, "vars": [0, 1]},
            ],
            "constraints": [],
        }
        matrix, varmap = qubify(problem)
        assert matrix.shape == (2, 2)
        assert matrix[0, 0] == 1.0
        assert matrix[0, 1] == 2.0
        assert varmap["x"]["size"] == 2

    def test_one_hot_constraint(self):
        problem = {
            "variables": {"x": ("binary", (3,))},
            "objective": [],
            "constraints": [
                {"type": "one_hot", "vars": [0, 1, 2]},
            ],
        }
        matrix, _ = qubify(problem)
        assert matrix.shape == (3, 3)
        # Should be non-zero (penalty added)
        assert not np.allclose(matrix, np.zeros((3, 3)))

    def test_user_penalty(self):
        """User-specified penalty should be used directly."""
        problem = {
            "variables": {"x": ("binary", (2,))},
            "objective": [{"coeff": 1.0, "vars": [0]}],
            "constraints": [
                {"type": "one_hot", "vars": [0, 1]},
            ],
            "penalty": 100.0,
        }
        matrix, _ = qubify(problem)
        # The penalty should be 100x the constraint expression
        # Constraint one_hot([0,1]) with penalty=100:
        # linear: 100*(-1)*x0 + 100*(-1)*x1 = -100x0 - 100x1
        # quadratic: 100*2*x0*x1 = 200*x0*x1
        # const: 100
        # Plus objective: 1*x0
        # So linear[0] = -100 + 1 = -99
        assert matrix[0, 0] == pytest.approx(-99.0)
        assert matrix[1, 1] == pytest.approx(-100.0)
        assert matrix[0, 1] == pytest.approx(200.0)

    def test_variable_ref_by_name(self):
        problem = {
            "variables": {"a": ("binary", (2,)), "b": ("binary", (2,))},
            "objective": [
                {"coeff": 1.0, "vars": ["a", "b"]},
            ],
            "constraints": [],
        }
        matrix, varmap = qubify(problem)
        assert matrix.shape == (4, 4)
        # "a" maps to index 0, "b" maps to index 2
        assert matrix[0, 2] == 1.0

    def test_variable_ref_by_tuple(self):
        problem = {
            "variables": {"x": ("binary", (3,))},
            "objective": [
                {"coeff": 5.0, "vars": [("x", 1)]},
            ],
            "constraints": [],
        }
        matrix, varmap = qubify(problem)
        assert matrix[1, 1] == 5.0

    def test_multiple_constraints(self):
        problem = {
            "variables": {"x": ("binary", (4,))},
            "objective": [],
            "constraints": [
                {"type": "one_hot", "vars": [0, 1]},
                {"type": "one_hot", "vars": [2, 3]},
            ],
        }
        matrix, _ = qubify(problem)
        assert matrix.shape == (4, 4)

    def test_constraint_type_cardinality(self):
        problem = {
            "variables": {"x": ("binary", (4,))},
            "objective": [],
            "constraints": [
                {"type": "cardinality", "vars": [0, 1, 2, 3], "rhs": 2},
            ],
        }
        matrix, _ = qubify(problem)
        assert matrix.shape == (4, 4)

    def test_constraint_type_mutual_exclusive_pairwise(self):
        """mutual_exclusive with >2 vars should apply pairwise."""
        problem = {
            "variables": {"x": ("binary", (3,))},
            "objective": [],
            "constraints": [
                {"type": "mutual_exclusive", "vars": [0, 1, 2]},
            ],
        }
        matrix, _ = qubify(problem)
        assert matrix.shape == (3, 3)

    def test_multi_dim_variable_index(self):
        """Multi-dimensional variable with named tuple indexing."""
        problem = {
            "variables": {"x": ("binary", (2, 3))},
            "objective": [
                {"coeff": 1.0, "vars": [("x", (0, 1))]},
            ],
            "constraints": [],
        }
        matrix, varmap = qubify(problem)
        assert matrix.shape == (6, 6)
        # x_(0,1) = flat index 0*3+1 = 1
        assert matrix[1, 1] == 1.0

    def test_all_constraint_types(self):
        """Smoke test: compile a problem using every constraint type."""
        problem = {
            "variables": {"x": ("binary", (6,))},
            "objective": [{"coeff": 1.0, "vars": [0]}],
            "constraints": [
                {"type": "one_hot", "vars": [0, 1]},
                {"type": "cardinality", "vars": [0, 1, 2], "rhs": 1},
                {"type": "at_least_one", "vars": [3, 4]},
                {"type": "mutual_exclusive", "vars": [3, 5]},
                {"type": "implication", "vars": [2, 3]},
                {"type": "equality", "vars": [1, 4]},
                {"type": "at_most_k", "vars": [3, 4, 5], "rhs": 1},
            ],
        }
        matrix, _ = qubify(problem)
        assert matrix.shape == (6, 6)
        # Should not be zero
        assert not np.allclose(matrix, np.zeros((6, 6)))

    def test_variable_map_structure(self):
        problem = {
            "variables": {"items": ("binary", (3,)), "aux": ("binary", (2,))},
            "objective": [],
            "constraints": [],
        }
        _, varmap = qubify(problem)
        assert varmap["items"]["start"] == 0
        assert varmap["items"]["size"] == 3
        assert varmap["aux"]["start"] == 3
        assert varmap["aux"]["size"] == 2
