"""
The qubify compiler: turn a problem description into a QUBO matrix.

qubify = make it QUBO
"""

import numpy as np
from typing import Any

from qubify.expressions import QuboExpr
from qubify.constraints import (
    one_hot,
    cardinality,
    at_least_one,
    mutual_exclusive,
    implication,
    equality,
    at_most_k,
)


# ── public API ──────────────────────────────────────────────────────

def qubify(problem: dict[str, Any]) -> tuple[np.ndarray, dict]:
    """Compile a problem description into a QUBO matrix.

    Args:
        problem: dict with keys:
            - variables: dict {name: "binary"} or {name: ("binary", shape)}
            - objective: list of {"coeff": c, "vars": [i, j]}  (quadratic terms)
                         or list of {"coeff": c, "vars": [i]}     (linear terms)
            - constraints: list of {"type": constraint_name, "vars": [...], "rhs": k}
            - penalty: optional float, global penalty coefficient (default: auto)

    Returns:
        (QUBO_matrix, var_map) where var_map maps variable_name → (index, shape)

    Raises:
        ValueError: if the problem description is invalid.

    Example:
        problem = {
            "variables": {"x": ("binary", (4,))},
            "objective": [
                {"coeff": 1.0, "vars": [0, 1]},
                {"coeff": 2.0, "vars": [2, 3]},
            ],
            "constraints": [
                {"type": "one_hot", "vars": [0, 1, 2, 3]},
            ],
        }
        matrix, vmap = qubify(problem)
    """
    # ── 0. Validate ────────────────────────────────────────────────
    _validate_problem(problem)

    # ── 1. Parse variables ─────────────────────────────────────────
    var_map = _parse_variables(problem.get("variables", {}))
    n_vars = sum(info["size"] for info in var_map.values())

    # ── 2. Build objective expression ──────────────────────────────
    obj_expr = _build_objective(problem.get("objective", []), var_map)

    # ── 3. Build constraint expressions ────────────────────────────
    penalty_coeff = _estimate_penalty(obj_expr, problem.get("constraints", []), problem)

    total = obj_expr
    for c in problem.get("constraints", []):
        ctype = c["type"]
        c_vars_raw = c.get("vars", [])
        c_vars = [_flatten_var(v, var_map) for v in c_vars_raw]
        rhs = c.get("rhs")

        constraint_expr = _dispatch_constraint(ctype, c_vars, rhs, n_vars)
        total = total + penalty_coeff * constraint_expr

    # ── 4. Convert to matrix ───────────────────────────────────────
    matrix = total.to_matrix(n_vars)
    return matrix, var_map


# ── validation ─────────────────────────────────────────────────────

VALID_CONSTRAINT_TYPES = {
    "one_hot", "cardinality", "at_least_one",
    "mutual_exclusive", "implication", "equality", "at_most_k",
}

REQUIRED_KEYS = {"variables", "objective", "constraints"}

def _validate_problem(problem: dict[str, Any]) -> None:
    """Validate the DSL problem dict and raise clear ValueError on issues."""

    if not isinstance(problem, dict):
        raise ValueError(
            f"Problem must be a dict, got {type(problem).__name__}. "
            f"See https://github.com/LehaoLin/qubify#readme for the DSL schema."
        )

    # Required top-level keys
    missing = REQUIRED_KEYS - set(problem.keys())
    if missing:
        raise ValueError(
            f"Missing required keys: {sorted(missing)}. "
            f"A qubify problem must include: variables, objective, constraints. "
            f"Each can be empty (e.g. `\"constraints\": []`)."
        )

    # variables
    variables = problem.get("variables", {})
    if not isinstance(variables, dict):
        raise ValueError(
            f"'variables' must be a dict, got {type(variables).__name__}. "
            f"Example: {{\"x\": (\"binary\", (4,))}}"
        )
    for name, spec in variables.items():
        if not isinstance(name, str):
            raise ValueError(
                f"Variable names must be strings, got {type(name).__name__}: {name!r}"
            )
        if isinstance(spec, str):
            if spec != "binary":
                raise ValueError(
                    f"Variable type for '{name}' must be 'binary', got {spec!r}"
                )
        elif isinstance(spec, tuple):
            if len(spec) != 2 or spec[0] != "binary":
                raise ValueError(
                    f"Variable spec for '{name}' must be ('binary', shape), got {spec!r}"
                )
            shape = spec[1]
            if isinstance(shape, int):
                if shape <= 0:
                    raise ValueError(
                        f"Variable shape for '{name}' must be positive, got {shape}"
                    )
            elif isinstance(shape, tuple):
                for d in shape:
                    if not isinstance(d, int) or d <= 0:
                        raise ValueError(
                            f"Variable shape dimensions for '{name}' must be positive integers, "
                            f"got {shape}"
                        )
            else:
                raise ValueError(
                    f"Variable shape for '{name}' must be int or tuple of ints, "
                    f"got {type(shape).__name__}"
                )
        else:
            raise ValueError(
                f"Variable spec for '{name}' must be 'binary' or ('binary', shape), "
                f"got {type(spec).__name__}: {spec!r}"
            )

    # objective
    objective = problem.get("objective", [])
    if not isinstance(objective, list):
        raise ValueError(
            f"'objective' must be a list, got {type(objective).__name__}"
        )
    for idx, term in enumerate(objective):
        if not isinstance(term, dict):
            raise ValueError(
                f"Objective term [{idx}] must be a dict, got {type(term).__name__}"
            )
        if "coeff" not in term:
            raise ValueError(
                f"Objective term [{idx}] missing required key 'coeff'. "
                f"Format: {{\"coeff\": <float>, \"vars\": [<indices>]}}"
            )
        if "vars" not in term:
            raise ValueError(
                f"Objective term [{idx}] missing required key 'vars'. "
                f"Format: {{\"coeff\": <float>, \"vars\": [<indices>]}}"
            )
        vlist = term["vars"]
        if not isinstance(vlist, list) or len(vlist) not in (1, 2):
            raise ValueError(
                f"Objective term [{idx}] 'vars' must be a list of 1 or 2 indices, "
                f"got {vlist!r}"
            )

    # constraints
    constraints = problem.get("constraints", [])
    if not isinstance(constraints, list):
        raise ValueError(
            f"'constraints' must be a list, got {type(constraints).__name__}"
        )
    for idx, c in enumerate(constraints):
        if not isinstance(c, dict):
            raise ValueError(
                f"Constraint [{idx}] must be a dict, got {type(c).__name__}"
            )
        if "type" not in c:
            raise ValueError(
                f"Constraint [{idx}] missing required key 'type'. "
                f"Available types: {sorted(VALID_CONSTRAINT_TYPES)}"
            )
        ctype = c["type"]
        if ctype not in VALID_CONSTRAINT_TYPES:
            raise ValueError(
                f"Constraint [{idx}] unknown type {ctype!r}. "
                f"Available types: {sorted(VALID_CONSTRAINT_TYPES)}"
            )
        if "vars" not in c:
            raise ValueError(
                f"Constraint [{idx}] of type {ctype!r} missing required key 'vars'"
            )
        cvars = c["vars"]
        if not isinstance(cvars, list):
            raise ValueError(
                f"Constraint [{idx}] 'vars' must be a list, got {type(cvars).__name__}"
            )
        if ctype in ("cardinality", "at_most_k"):
            if "rhs" not in c:
                raise ValueError(
                    f"Constraint [{idx}] of type {ctype!r} requires 'rhs' (right-hand side integer). "
                    f"Example: {{\"type\": \"cardinality\", \"vars\": [0,1,2], \"rhs\": 2}}"
                )

    # penalty (optional)
    if "penalty" in problem:
        p = problem["penalty"]
        if not isinstance(p, (int, float)) or p <= 0:
            raise ValueError(
                f"'penalty' must be a positive number, got {p!r}"
            )


# ── internal helpers ───────────────────────────────────────────────

def _parse_variables(variables: dict) -> dict:
    """Parse variable declarations into a flat index map.

    Returns:
        {name: {"start": int, "size": int, "shape": tuple}}
    """
    var_map = {}
    offset = 0
    for name, spec in variables.items():
        if isinstance(spec, str):
            size = 1
            shape = (1,)
        elif isinstance(spec, tuple) and spec[0] == "binary":
            shape = spec[1] if isinstance(spec[1], tuple) else (spec[1],)
            size = int(np.prod(shape))
        else:
            raise ValueError(f"Invalid variable spec for '{name}': {spec}")

        var_map[name] = {"start": offset, "size": size, "shape": shape}
        offset += size
    return var_map


def _flatten_var(vspec, var_map: dict) -> int:
    """Convert a variable reference to a flat index.

    Args:
        vspec: int (direct index), str "name", or tuple ("name", idx)
        var_map: parsed variable map

    Returns:
        Flat integer index.

    Raises:
        ValueError, KeyError, IndexError on bad references.
    """
    if isinstance(vspec, int):
        total_vars = sum(info["size"] for info in var_map.values())
        if vspec < 0 or vspec >= total_vars:
            raise ValueError(
                f"Variable index {vspec} is out of range [0, {total_vars - 1}]"
            )
        return vspec

    if isinstance(vspec, str):
        if vspec not in var_map:
            raise KeyError(
                f"Variable '{vspec}' not found. Defined variables: {list(var_map.keys())}"
            )
        return var_map[vspec]["start"]

    if isinstance(vspec, (list, tuple)):
        name = vspec[0]
        if name not in var_map:
            raise KeyError(
                f"Variable '{name}' not found. Defined variables: {list(var_map.keys())}"
            )
        idx = vspec[1] if len(vspec) > 1 else 0
        if isinstance(idx, int):
            size = var_map[name]["size"]
            if idx < 0 or idx >= size:
                raise IndexError(
                    f"Index {idx} out of range for variable '{name}' (size {size})"
                )
            return var_map[name]["start"] + idx
        # multi-dimensional index: compute flat offset
        shape = var_map[name]["shape"]
        return var_map[name]["start"] + int(np.ravel_multi_index(idx, shape))

    raise ValueError(
        f"Invalid variable reference: {vspec!r}. "
        f"Use an int (flat index), str (variable name), or tuple (name, index)."
    )


def _resolve_var(vspec, var_map: dict) -> QuboExpr:
    """Get a single-var QuboExpr from a variable reference."""
    from qubify.expressions import var as _var

    idx = _flatten_var(vspec, var_map)
    return _var(idx)


def _build_objective(terms: list[dict], var_map: dict) -> QuboExpr:
    """Build objective expression from term list."""
    from qubify.expressions import var as _var, prod as _prod

    expr = QuboExpr()
    for term in terms:
        coeff = float(term["coeff"])
        vlist = term["vars"]
        if len(vlist) == 1:
            idx = _flatten_var(vlist[0], var_map)
            expr = expr + coeff * _var(idx)
        elif len(vlist) == 2:
            i = _flatten_var(vlist[0], var_map)
            j = _flatten_var(vlist[1], var_map)
            expr = expr + coeff * _prod(i, j)
    return expr


def _dispatch_constraint(ctype: str, vars: list[int], rhs, n_vars: int) -> QuboExpr:
    """Dispatch a constraint type to the appropriate penalty function."""
    if ctype == "one_hot":
        return one_hot(vars)
    elif ctype == "cardinality":
        k = rhs if rhs is not None else 1
        return cardinality(vars, int(k))
    elif ctype == "at_least_one":
        return at_least_one(vars)
    elif ctype == "mutual_exclusive":
        if len(vars) == 2:
            return mutual_exclusive(vars[0], vars[1])
        # pairwise for >2 vars
        expr = QuboExpr()
        for i in range(len(vars)):
            for j in range(i + 1, len(vars)):
                expr = expr + mutual_exclusive(vars[i], vars[j])
        return expr
    elif ctype == "implication":
        return implication(vars[0], vars[1])
    elif ctype == "equality":
        return equality(vars[0], vars[1])
    elif ctype == "at_most_k":
        k = rhs if rhs is not None else 1
        return at_most_k(vars, int(k))
    else:
        raise ValueError(f"Unknown constraint type: {ctype}")


def _estimate_penalty(objective: QuboExpr, constraints: list[dict], problem: dict) -> float:
    """Determine the penalty coefficient.

    If user provides 'penalty' in the problem dict, use it directly.
    Otherwise, auto-estimate: penalty ≈ 2 × max(|obj coeff|) × max(constraint size).

    Ensures constraint violation costs more than objective gain.

    Auto-estimation is a heuristic. For best results on problems with
    widely-varying objective coefficients, manually set 'penalty' in the
    problem dict.
    """
    # User-specified penalty takes priority
    user_penalty = problem.get("penalty")
    if user_penalty is not None:
        return float(user_penalty)

    # Auto-estimation heuristic
    max_obj = 0.0
    for v in objective.quadratic.values():
        max_obj = max(max_obj, abs(v))
    for v in objective.linear.values():
        max_obj = max(max_obj, abs(v))

    max_obj = max(max_obj, 1.0)  # avoid zero penalty for zero-objective problems

    max_constraint_size = 1
    for c in constraints:
        size = len(c.get("vars", []))
        max_constraint_size = max(max_constraint_size, size)

    return 2.0 * max_obj * max_constraint_size
