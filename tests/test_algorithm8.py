"""Tests for Algorithm 8 — Winner Reconstruction.

Reference: Algorithm 8 in "Fairness Without Exposure: Privacy-Preserving
Phragmén Voting", Section 5.8.

Standard test parameters: p=11, n=3 parties, t=2 threshold.
"""

import random

import pytest

from mpc_secret_shares import share
from protocol.algorithm8_reconstruct_winner import algorithm_8_reconstruct_winner
from protocol.types import Shares

P, N, T = 11, 3, 2


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _share(v: int) -> Shares:
    return share(v, N, T, P)


def _chi(winner: int, m: int) -> list[Shares]:
    """Build a valid one-hot chi for M candidates with the given winner."""
    return [_share(1 if i == winner else 0) for i in range(m)]


# ---------------------------------------------------------------------------
# Normal operation
# ---------------------------------------------------------------------------

class TestNormalOperation:
    def setup_method(self):
        random.seed(10)

    def test_m1_sole_candidate_wins(self):
        """M=1: only possible winner is index 0."""
        assert algorithm_8_reconstruct_winner([_share(1)], T, P) == 0

    def test_winner_is_first(self):
        """Winner at index 0 among three candidates."""
        assert algorithm_8_reconstruct_winner(_chi(0, 3), T, P) == 0

    def test_winner_is_middle(self):
        """Winner at index 1 among three candidates."""
        assert algorithm_8_reconstruct_winner(_chi(1, 3), T, P) == 1

    def test_winner_is_last(self):
        """Winner at index 2 (last) among three candidates."""
        assert algorithm_8_reconstruct_winner(_chi(2, 3), T, P) == 2

    def test_winner_last_of_five(self):
        """Winner at index 4 among five candidates."""
        assert algorithm_8_reconstruct_winner(_chi(4, 5), T, P) == 4

    def test_return_type_is_int(self):
        """Return value must be a plain Python int."""
        result = algorithm_8_reconstruct_winner(_chi(1, 3), T, P)
        assert isinstance(result, int)

    def test_winner_index_is_zero_indexed(self):
        """Returned index is 0-based (paper uses 1-based; implementation is 0-based)."""
        # chi with winner at position 0 → returns 0, not 1
        assert algorithm_8_reconstruct_winner(_chi(0, 4), T, P) == 0


# ---------------------------------------------------------------------------
# Abort conditions (Lines 3-6 of the paper)
# ---------------------------------------------------------------------------

class TestAbortConditions:
    def setup_method(self):
        random.seed(20)

    def test_all_zeros_raises(self):
        """All χ_m = 0 → Σ χ_m = 0 ≠ 1 → ValueError."""
        chi = [_share(0), _share(0), _share(0)]
        with pytest.raises(ValueError, match="≠ 1"):
            algorithm_8_reconstruct_winner(chi, T, P)

    def test_two_ones_raises(self):
        """Two candidates with χ=1 → Σ χ_m = 2 ≠ 1 → ValueError."""
        chi = [_share(1), _share(1), _share(0)]
        with pytest.raises(ValueError, match="≠ 1"):
            algorithm_8_reconstruct_winner(chi, T, P)

    def test_all_ones_raises(self):
        """All χ_m = 1 → Σ χ_m = M > 1 → ValueError."""
        chi = [_share(1), _share(1), _share(1)]
        with pytest.raises(ValueError, match="≠ 1"):
            algorithm_8_reconstruct_winner(chi, T, P)

    def test_malformed_value_raises(self):
        """χ_m = 2 ∉ {0,1} → ValueError on the validity check (Lines 3-4)."""
        chi = [_share(0), _share(2), _share(0)]
        with pytest.raises(ValueError, match="not in"):
            algorithm_8_reconstruct_winner(chi, T, P)

    def test_malformed_value_raises_before_sum_check(self):
        """A malformed value aborts at the per-entry check, not the sum check."""
        # value=5 is not in {0,1}; error must mention the invalid value
        chi = [_share(5), _share(0)]
        with pytest.raises(ValueError, match="not in"):
            algorithm_8_reconstruct_winner(chi, T, P)

    def test_empty_chi_raises(self):
        """Empty input raises ValueError before any reconstruction."""
        with pytest.raises(ValueError):
            algorithm_8_reconstruct_winner([], T, P)


# ---------------------------------------------------------------------------
# Consistency with Algorithm 6 output
# ---------------------------------------------------------------------------

class TestConsistencyWithAlgorithm6:
    """Verify that Algorithm 8 correctly identifies winners produced by
    Algorithm 6, using hand-crafted but realistic chi vectors.
    """

    def setup_method(self):
        random.seed(30)

    def test_unique_winner_round_trips(self):
        """_chi(m, M) → algorithm_8 returns m for all positions."""
        for m_count in (2, 3, 4):
            for winner in range(m_count):
                chi = _chi(winner, m_count)
                assert algorithm_8_reconstruct_winner(chi, T, P) == winner

    def test_winner_identified_regardless_of_position(self):
        """Winner in the middle of a 5-candidate field is correctly identified."""
        assert algorithm_8_reconstruct_winner(_chi(2, 5), T, P) == 2
