"""Tests for qubify.presets — tsp, maxcut, knapsack and their decoders."""

import numpy as np
import pytest
from qubify.presets import (
    tsp, tsp_decode,
    maxcut, maxcut_decode,
    knapsack, knapsack_decode, knapsack_decode_full,
)


class TestTSP:
    """Traveling Salesman Problem preset."""

    def test_small_tsp_shape(self):
        distances = [[0, 10, 15], [10, 0, 35], [15, 35, 0]]
        matrix, varmap = tsp(distances)
        n = 3
        assert matrix.shape == (n * n, n * n)  # 9×9
        assert varmap["x"]["size"] == 9
        assert varmap["x"]["shape"] == (3, 3)

    def test_tsp_known_solution(self):
        """For a triangle TSP, [0,1,2] is feasible — verify it decodes correctly."""
        distances = [[0, 1, 10], [1, 0, 10], [10, 10, 0]]
        matrix, varmap = tsp(distances)

        # Tour [0,1,2]: x[0*3+0]=1, x[1*3+1]=1, x[2*3+2]=1
        x = np.zeros(9)
        x[0] = 1  # city 0 at pos 0
        x[4] = 1  # city 1 at pos 1
        x[8] = 1  # city 2 at pos 2

        energy = x @ matrix @ x
        # Energy includes constraint penalty constants; just verify it's finite
        assert np.isfinite(energy)

        # The tour should decode correctly
        tour = tsp_decode(x, var_map=varmap)
        assert tour == [0, 1, 2]

    def test_tsp_decode(self):
        sol = [1, 0, 0, 0, 1, 0, 0, 0, 1]  # city0→city1→city2
        tour = tsp_decode(sol, n_cities=3)
        assert tour == [0, 1, 2]

    def test_tsp_decode_with_varmap(self):
        distances = [[0, 10, 15], [10, 0, 35], [15, 35, 0]]
        matrix, varmap = tsp(distances)
        sol = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        tour = tsp_decode(sol, var_map=varmap)
        assert tour == [0, 1, 2]

    def test_tsp_decode_different_order(self):
        sol = [0, 1, 0, 0, 0, 1, 1, 0, 0]  # city2→city0→city1
        tour = tsp_decode(sol, n_cities=3)
        assert tour == [2, 0, 1]

    def test_tsp_decode_infeasible(self):
        sol = [1, 1, 0, 0, 0, 0, 0, 0, 0]  # two cities at pos 0
        with pytest.raises(ValueError, match="expected exactly 1"):
            tsp_decode(sol, n_cities=3)


class TestMaxCut:
    """Max-Cut preset."""

    def test_small_graph_shape(self):
        adj = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        matrix, varmap = maxcut(adj)
        assert matrix.shape == (3, 3)
        assert varmap["x"]["size"] == 3

    def test_maxcut_known_energy(self):
        """Line graph of 3 nodes: cut {0,2} vs {1} gives value 2."""
        adj = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        matrix, _ = maxcut(adj)

        # Cut [1, 0, 1]: nodes 0,2 in one set, node 1 in other
        x = np.array([1, 0, 1])
        energy = x @ matrix @ x

        # The cut value is -energy (since we minimize, cut = -energy)
        # Edges (0,1) and (1,2) are cut = 2
        assert -energy == pytest.approx(2.0)

    def test_maxcut_decode(self):
        sol = [1, 0, 1]
        A, B = maxcut_decode(sol)
        assert A == [0, 2]
        assert B == [1]

    def test_maxcut_decode_with_cut_value(self):
        adj = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        sol = [1, 0, 1]
        A, B, val = maxcut_decode(sol, adjacency=adj)
        assert val == pytest.approx(2.0)

    def test_maxcut_decode_all_one(self):
        sol = [1, 1, 1]
        A, B = maxcut_decode(sol)
        assert len(A) == 3
        assert len(B) == 0


class TestKnapsack:
    """Knapsack preset."""

    def test_small_knapsack_shape(self):
        values = [60, 100, 120]
        weights = [10, 20, 30]
        capacity = 50
        slack_bits = 4
        matrix, varmap = knapsack(values, weights, capacity, slack_bits)
        assert matrix.shape == (3 + slack_bits, 3 + slack_bits)
        assert varmap["x"]["size"] == 7

    def test_knapsack_nonzero(self):
        values = [60, 100, 120]
        weights = [10, 20, 30]
        capacity = 50
        matrix, _ = knapsack(values, weights, capacity)
        assert not np.allclose(matrix, np.zeros(matrix.shape))

    def test_knapsack_decode(self):
        # Solution: items 0 and 2 selected, slack=0
        sol = [1, 0, 1, 0, 0, 0, 0]  # n=3, slack_bits=4
        result = knapsack_decode(sol, slack_bits=4)
        assert result["selected"] == [0, 2]

    def test_knapsack_decode_full(self):
        values = [60, 100, 120]
        weights = [10, 20, 30]
        sol = [1, 0, 1, 0, 0, 0, 0]
        result = knapsack_decode_full(sol, values, weights, slack_bits=4)
        assert result["selected"] == [0, 2]
        assert result["total_value"] == 180.0
        assert result["total_weight"] == 40.0

    def test_knapsack_decode_none_selected(self):
        sol = [0, 0, 0, 0, 0, 0, 0]
        result = knapsack_decode(sol, slack_bits=4)
        assert result["selected"] == []

    def test_knapsack_decode_all_selected(self):
        sol = [1, 1, 1, 0, 0, 0, 0]
        result = knapsack_decode(sol, slack_bits=4)
        assert result["selected"] == [0, 1, 2]
