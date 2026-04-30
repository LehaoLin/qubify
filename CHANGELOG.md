# Changelog

All notable changes to qubify will be documented in this file.

## [0.1.0] - 2026-05-01

### Added
- **Core engine**: `qubify()` compiler pipeline — DSL → QUBO matrix in one call
- **7 constraint types**: `one_hot`, `cardinality`, `at_least_one`, `mutual_exclusive`, `implication`, `equality`, `at_most_k`
- **3 presets**: TSP, Max-Cut, 0/1 Knapsack with automatic QUBO generation
- **qubo_to_ising()**: solver-agnostic QUBO → Ising conversion (no hardware lock-in)
- **Solution decoders**: `tsp_decode()`, `maxcut_decode()`, `knapsack_decode()` / `knapsack_decode_full()`
- **DSL validation**: clear error messages for malformed problem descriptions
- **User-specified penalty**: override auto-estimated penalty coefficient via `penalty` key
- **101 unit tests** covering expressions, constraints, compiler pipeline, presets, and utilities
- **GitHub Actions CI**: automated testing on Python 3.10–3.12, PyPI publish on tag push
- **pip installable**: `pip install qubify` from PyPI

### Changed
- `at_least_one` and `at_most_k` documented as `cardinality(vars, k)` approximations; true inequality support with slack variables planned for v0.2

### Fixed
- Dead code removed: unused `combinations` import, dead `sign` variable in `implication`
- `qubo_to_ising()` offset now includes diagonal trace term for energy equivalence
- `maxcut_decode()` cut value no longer double-counts symmetric edges
