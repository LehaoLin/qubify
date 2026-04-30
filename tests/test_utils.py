"""Tests for qubify.utils — QUBO ↔ Ising conversion."""

import numpy as np
import pytest
from qubify.utils import qubo_to_ising


class TestQuboToIsing:
    """QUBO → Ising conversion correctness."""

    def test_shape(self):
        Q = np.array([[1, -2], [0, 3]])
        J, h, offset = qubo_to_ising(Q)
        assert J.shape == (2, 2)
        assert h.shape == (2,)

    def test_diagonal_zero(self):
        """J should have zero diagonal (only coupling terms)."""
        Q = np.array([[1, 2], [0, 3]])
        J, _, _ = qubo_to_ising(Q)
        assert J[0, 0] == 0.0
        assert J[1, 1] == 0.0

    def test_upper_triangular(self):
        """J should be upper triangular."""
        Q = np.array([[1, 2], [0, 3]])
        J, _, _ = qubo_to_ising(Q)
        assert J[1, 0] == 0.0
        assert J[0, 1] != 0.0

    def test_energy_equivalence(self):
        """QUBO and Ising energies should match (up to offset)."""
        Q = np.array([[1, -2], [0, 3]])
        J, h, offset = qubo_to_ising(Q)

        # Binary vectors and their Ising equivalents
        test_vectors = [
            np.array([0, 0]),
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([1, 1]),
        ]

        for x in test_vectors:
            s = 1 - 2 * x  # x ∈ {0,1} → s ∈ {+1,-1}, using x = (1-s)/2
            energy_qubo = x @ Q @ x
            energy_ising = s @ J @ s + h @ s + offset
            assert energy_qubo == pytest.approx(energy_ising, rel=1e-10)

    def test_simple_diagonal(self):
        """Single-variable: QUBO x_0 → Ising -s_0/2 + 1/2."""
        Q = np.array([[1.0]])
        J, h, offset = qubo_to_ising(Q)

        # Manual: S = Q (1×1), S = [[1]]
        # h = -0.5 * S[0,0] = -0.5
        # offset = (sum(S) + trace(S)) / 4 = (1+1)/4 = 0.5
        # J = 0 (no off-diagonal)
        assert J[0, 0] == 0.0
        assert h[0] == pytest.approx(-0.5)
        assert offset == pytest.approx(0.5)

        # Verify: x=0 → s=1, energy = 1*J*1 + h*1 + offset = 0 - 0.5 + 0.25 = -0.25? 
        # Wait, x=0 → qubo energy = 0*1*0 = 0
        # s=(1-2*0)=1, energy_ising = 1*0*1 + (-0.5)*1 + 0.25 = -0.25
        # That doesn't match. Let me check the formula more carefully.
        # x = (1-s)/2, s = 1-2x
        
        # For x=0: s=1
        s = np.array([1.0])
        x = np.array([0.0])
        equbo = x @ Q @ x
        eising = s @ J @ s + h @ s + offset
        assert equbo == pytest.approx(eising, rel=1e-10)

    def test_known_case(self):
        """Test a 2×2 case with known manual derivation."""
        # Q = [[a, b], [0, d]] → QUBO: a*x0 + d*x1 + b*x0*x1
        # Let's use a=2, b=1, d=3
        Q = np.array([[2.0, 1.0], [0.0, 3.0]])
        J, h, offset = qubo_to_ising(Q)

        # All 4 binary states
        for x0 in [0, 1]:
            for x1 in [0, 1]:
                x = np.array([x0, x1], dtype=float)
                s = 1 - 2 * x
                equbo = x @ Q @ x
                eising = s @ J @ s + h @ s + offset
                assert equbo == pytest.approx(eising, rel=1e-10)

    def test_large_matrix(self):
        """Smoke test with a 5×5 random matrix."""
        rng = np.random.RandomState(42)
        Q = rng.randn(5, 5)
        Q = np.triu(Q)  # make upper triangular
        J, h, offset = qubo_to_ising(Q)

        assert J.shape == (5, 5)
        assert h.shape == (5,)
        assert np.allclose(np.diag(J), np.zeros(5))

        # Test a few random binary vectors
        for _ in range(10):
            x = rng.randint(0, 2, size=5).astype(float)
            s = 1 - 2 * x
            assert x @ Q @ x == pytest.approx(s @ J @ s + h @ s + offset, rel=1e-10)
