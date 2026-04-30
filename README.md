# qubify

> **qubify = make it QUBO**

A constraint-to-QUBO compiler for quantum optimization. Convert optimization problems into QUBO matrices — no hardware lock-in, no solver dependency. Feed the output to CIM, D-Wave, Fujitsu DA, simulated annealing, or any QUBO solver.

```bash
pip install qubify
```

---

## Why qubify?

| You have | qubify gives you |
|----------|-----------------|
| A TSP with N cities | An N² × N² QUBO matrix |
| A Max-Cut graph | An N × N QUBO matrix |
| A knapsack with items + capacity | An (N + slack) × (N + slack) QUBO matrix |
| Your own constraint problem | A QUBO matrix from a declarative DSL |

The compiler handles **variable allocation**, **constraint-to-penalty conversion**, and **automatic penalty coefficient estimation** so you don't have to hand-derive QUBO matrices.

---

## Quick Start

### Using Presets (easiest)

```python
from qubify.presets import tsp, maxcut, knapsack

# Traveling Salesman Problem
distances = [[0, 10, 15], [10, 0, 35], [15, 35, 0]]
qubo_matrix, varmap = tsp(distances)

# Max-Cut
adjacency = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
qubo_matrix, varmap = maxcut(adjacency)

# Knapsack
qubo_matrix, varmap = knapsack(
    values=[60, 100, 120],
    weights=[10, 20, 30],
    capacity=50,
)
```

### Using the Compiler (flexible)

```python
from qubify import qubify

problem = {
    "variables": {"x": ("binary", (5,))},
    "objective": [
        {"coeff": -1.0, "vars": [0]},        # favor x[0] = 1
        {"coeff": 2.0, "vars": [0, 1]},      # penalize x[0]x[1]
    ],
    "constraints": [
        {"type": "one_hot", "vars": [0, 1, 2]},
        {"type": "cardinality", "vars": [3, 4], "rhs": 1},
    ],
}

matrix, varmap = qubify(problem)
# matrix is ready for any QUBO/Ising solver
```

### Feed to a Quantum Solver

```python
import kaiwu as kw

# Convert to Ising (if the solver requires it)
ising_matrix = kw.qubo.qubo_to_ising(matrix)[0]

# Solve with simulated annealing
opt = kw.classical.SimulatedAnnealingOptimizer()
solution = opt.solve(ising_matrix)

# Or with CIM quantum hardware
opt = kw.cim.CIMOptimizer(task_name="my-tsp", wait=True)
solution = opt.solve(ising_matrix)
```

---

## Available Constraint Types

| Constraint | Signature | Description |
|-----------|-----------|-------------|
| `one_hot` | `vars: [i, j, ...]` | Exactly one variable = 1 |
| `cardinality` | `vars: [...]`, `rhs: k` | Exactly k variables = 1 |
| `at_least_one` | `vars: [...]` | At least one variable = 1 |
| `mutual_exclusive` | `vars: [i, j]` | Cannot both be 1 |
| `implication` | `vars: [a, b]` | a ⇒ b |
| `equality` | `vars: [a, b]` | a = b |
| `at_most_k` | `vars: [...]`, `rhs: k` | At most k variables = 1 |

All constraints are compiled into quadratic penalties using exact binary arithmetic — no approximations, no ancilla qubits needed for basic constraints.

---

## Variable System

Declare named variable blocks with shapes:

```python
"variables": {
    "x": ("binary", (2, 3)),    # 2×3 grid → 6 vars, indices 0–5
    "y": ("binary", (4,)),      # 1D array → 4 vars, indices 6–9
}
```

Reference variables in objectives and constraints by:
- **Flat index**: `0`, `5`
- **Named index**: `("x", (0, 1))` → flat index 1
- **Named block head**: `"y"` → flat index 6 (first element of block)

---

## Presets Included

| Preset | Function | Variables | Constraints |
|--------|----------|-----------|-------------|
| **TSP** | `tsp(distances)` | n² | 2n one_hot |
| **Max-Cut** | `maxcut(adjacency)` | n | none |
| **Knapsack** | `knapsack(v, w, C)` | n + slack | capacity (via objective) |

More presets coming: graph coloring, scheduling, portfolio optimization, set covering.

---

## Architecture

```
Problem (dict) → qubify() → QUBO matrix (numpy)
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   variables    objective   constraints
   (flat idx)  (QuboExpr)  (one_hot, etc.)
        │           │           │
        └───────────┴───────────┘
                    │
                    ▼
            penalty estimation
                    │
                    ▼
              Σ → QUBO matrix
```

**qubify is solver-agnostic.** It produces a standard numpy QUBO matrix. What you do with it is up to you.

---

## Install

```bash
pip install qubify
```

Dev install with tests:

```bash
pip install qubify[dev]
pytest tests/
```

---

## Related Projects

- **kaiwu-cli-mcp** — Docker + CLI + MCP wrapper for Kaiwu SDK (玻色量子 CIM). Use qubify to generate matrices, then feed them to kaiwu-cli-mcp for execution on quantum hardware.
- Built as a standalone library — works with any QUBO/Ising solver.

---

## Links

- [GitHub](https://github.com/LehaoLin/qubify)
- [Kaiwu SDK Docs](https://kaiwu-sdk-docs.qboson.com/zh/latest/index.html)

## License

MIT
