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
    # ── 1. Parse variables ─────────────────────────────────────────
    var_map = _parse_variables(problem.get("variables", {}))
    n_vars = sum(info["size"] for info in var_map.values())

    # ── 2. Build objective expression ──────────────────────────────
    objective = QuboExpr()
    for term in problem.get("objective", []):
        coeff = float(term["coeff"])
        vlist = term["vars"]
        if len(vlist) == 1:
            objective = objective + coeff * _resolve_var(vlist[0], var_map)
        elif len(vlist) == 2:
            i = _resolve_var(vlist[0], var_map)
            j = _resolve_var(vlist[1], var_map)
            # i and j are QuboExpr(var{i}) and QuboExpr(var{j})
            # We need to build the product manually
            # For now, handle simple case: vlist[i] is {idx: ...}
            pass

    # Rebuild objective using flattened indices
    obj_expr = _build_objective(problem.get("objective", []), var_map)

    # ── 3. Build constraint expressions ────────────────────────────
    penalty_coeff = _estimate_penalty(obj_expr, problem.get("constraints", []))

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
            shape = spec[1]
            size = int(np.prod(shape))
        else:
            raise ValueError(f"Invalid variable spec for '{name}': {spec}")

        var_map[name] = {"start": offset, "size": size, "shape": shape}
        offset += size
    return var_map


def _flatten_var(vspec, var_map: dict) -> int:
    """Convert a variable reference to a flat index.

    Args:
        vspec: int (direct index) or str "name" or tuple ("name", idx)
        var_map: parsed variable map
    """
    if isinstance(vspec, int):
        return vspec
    if isinstance(vspec, str):
        return var_map[vspec]["start"]
    if isinstance(vspec, (list, tuple)):
        name = vspec[0]
        idx = vspec[1] if len(vspec) > 1 else 0
        if isinstance(idx, int):
            return var_map[name]["start"] + idx
        # multi-dimensional index: compute flat offset
        shape = var_map[name]["shape"]
        return var_map[name]["start"] + int(np.ravel_multi_index(idx, shape))
    raise ValueError(f"Invalid variable reference: {vspec}")


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
        # pairwise
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


def _estimate_penalty(objective: QuboExpr, constraints: list[dict]) -> float:
    """Automatically estimate penalty coefficient.

    Heuristic: penalty ≈ 2 × max(|obj coeff|) × max(constraint size).
    Ensures constraint violation costs more than objective gain.

    For precise control, users can override by setting 'penalty' in problem dict.
    """
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
