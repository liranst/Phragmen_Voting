"""Tests for Algorithm 2 — Input Validation, Sanitization & Compaction.

Reference: Algorithm 2 in "Fairness Without Exposure: Privacy-Preserving
Phragmén Voting", Section 5.2.

Standard test parameters: p=11 (prime), n=3 parties, t=2 threshold.
"""

import random

import pytest

from mpc_primitives.mpc_project.mpc_secret_shares import (
    protocol_1_share,
    protocol_2_reconstruct,
)
from protocol.algorithm2_validation import algorithm_2_input_validation
from protocol.types import BallotMatrix, Shares

# ---------------------------------------------------------------------------
# Standard MPC parameters
# ---------------------------------------------------------------------------

P, N, T = 11, 3, 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _share(secret: int) -> Shares:
    """Create a (T,N)-sharing of secret over Z_P."""
    return protocol_1_share(secret, N, T, P)


def _ballot_matrix(values: list[list[int]]) -> BallotMatrix:
    """Build a BallotMatrix from a 2-D list of plaintext ballot values."""
    return [[_share(v) for v in row] for row in values]


def _reconstruct_ballot(B_hat: BallotMatrix) -> list[list[int]]:
    """Reconstruct every entry of B_hat for assertion comparison."""
    return [
        [protocol_2_reconstruct(B_hat[i][m][:T], P) for m in range(len(B_hat[i]))]
        for i in range(len(B_hat))
    ]


# ---------------------------------------------------------------------------
# Happy-path: no cheaters
# ---------------------------------------------------------------------------

class TestNoCheaters:
    def setup_method(self):
        random.seed(42)

    def test_all_zeros_valid(self):
        """All-zero ballot is valid (B=0 is in {0,1})."""
        ballots = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        B = _ballot_matrix(ballots)
        B_hat, n_valid = algorithm_2_input_validation(B, N, T, P)
        assert n_valid == 3
        assert _reconstruct_ballot(B_hat) == ballots

    def test_all_ones_valid(self):
        """All-one ballot is valid."""
        ballots = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        B = _ballot_matrix(ballots)
        B_hat, n_valid = algorithm_2_input_validation(B, N, T, P)
        assert n_valid == 3
        assert _reconstruct_ballot(B_hat) == ballots

    def test_mixed_zeros_and_ones(self):
        """Mixed 0/1 entries are all accepted."""
        ballots = [[1, 0, 1], [0, 1, 0], [1, 1, 0]]
        B = _ballot_matrix(ballots)
        B_hat, n_valid = algorithm_2_input_validation(B, N, T, P)
        assert n_valid == 3
        assert _reconstruct_ballot(B_hat) == ballots

    def test_n1_m1_valid(self):
        """Minimal case: N=1, M=1, honest voter."""
        B = _ballot_matrix([[1]])
        B_hat, n_valid = algorithm_2_input_validation(B, N, T, P)
        assert n_valid == 1
        assert len(B_hat) == 1
        assert protocol_2_reconstruct(B_hat[0][0][:T], P) == 1

    def test_row_order_preserved(self):
        """Honest voters retain their original relative order."""
        ballots = [[1, 0], [0, 1], [1, 1]]
        B = _ballot_matrix(ballots)
        B_hat, n_valid = algorithm_2_input_validation(B, N, T, P)
        assert n_valid == 3
        assert _reconstruct_ballot(B_hat) == ballots


# ---------------------------------------------------------------------------
# Cheater detection
# ---------------------------------------------------------------------------

class TestCheaterDetection:
    def setup_method(self):
        random.seed(99)

    def test_b_equals_2_detected(self):
        """B=2 is not in {0,1}: voter is flagged as cheater."""
        ballots = [[1, 0], [2, 0], [0, 1]]   # voter 1 cheats
        B = _ballot_matrix(ballots)
        B_hat, n_valid = algorithm_2_input_validation(B, N, T, P)
        assert n_valid == 2
        # Remaining rows are voters 0 and 2
        assert _reconstruct_ballot(B_hat) == [[1, 0], [0, 1]]

    def test_b_equals_3_detected(self):
        """B=3: cheater detected."""
        B = _ballot_matrix([[3, 0], [1, 0]])
        B_hat, n_valid = algorithm_2_input_validation(B, N, T, P)
        assert n_valid == 1
        assert _reconstruct_ballot(B_hat) == [[1, 0]]

    def test_b_equals_p_minus_1_detected(self):
        """B = p-1 ≡ -1 (mod p): B(B-1) = (p-1)(p-2) ≡ 2 ≠ 0 → cheater."""
        B = _ballot_matrix([[P - 1, 0], [1, 1]])
        B_hat, n_valid = algorithm_2_input_validation(B, N, T, P)
        assert n_valid == 1
        assert _reconstruct_ballot(B_hat) == [[1, 1]]

    def test_cheater_in_last_row(self):
        """Cheater is the last voter; only that row is removed."""
        ballots = [[1, 0], [0, 1], [2, 1]]   # voter 2 cheats
        B = _ballot_matrix(ballots)
        B_hat, n_valid = algorithm_2_input_validation(B, N, T, P)
        assert n_valid == 2
        assert _reconstruct_ballot(B_hat) == [[1, 0], [0, 1]]

    def test_cheater_in_first_row(self):
        """Cheater is the first voter; remaining two rows compacted."""
        ballots = [[2, 1], [1, 0], [0, 1]]   # voter 0 cheats
        B = _ballot_matrix(ballots)
        B_hat, n_valid = algorithm_2_input_validation(B, N, T, P)
        assert n_valid == 2
        assert _reconstruct_ballot(B_hat) == [[1, 0], [0, 1]]

    def test_multiple_cheaters(self):
        """Two cheaters: both rows removed, one honest row remains."""
        ballots = [[2, 0], [1, 1], [3, 0]]   # voters 0 and 2 cheat
        B = _ballot_matrix(ballots)
        B_hat, n_valid = algorithm_2_input_validation(B, N, T, P)
        assert n_valid == 1
        assert _reconstruct_ballot(B_hat) == [[1, 1]]

    def test_all_cheaters_n_valid_zero(self):
        """Every voter cheats → n_valid = 0, B_hat is empty."""
        B = _ballot_matrix([[2, 0], [3, 1], [5, 0]])
        B_hat, n_valid = algorithm_2_input_validation(B, N, T, P)
        assert n_valid == 0
        assert B_hat == []

    def test_break_on_first_bad_entry(self):
        """Detection stops at the first bad entry (second candidate column
        of a cheating voter is never evaluated)."""
        # voter 0: B[0][0]=2 → cheater; B[0][1] should never be checked
        # We verify this indirectly: if cheating occurs in col 0, the voter
        # is removed regardless of col 1.
        ballots = [[2, 1], [1, 0]]
        B = _ballot_matrix(ballots)
        B_hat, n_valid = algorithm_2_input_validation(B, N, T, P)
        assert n_valid == 1
        assert _reconstruct_ballot(B_hat) == [[1, 0]]


# ---------------------------------------------------------------------------
# Edge cases — empty / single-element inputs
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def setup_method(self):
        random.seed(7)

    def test_no_voters(self):
        """Empty voter list → n_valid=0, B_hat=[]."""
        B_hat, n_valid = algorithm_2_input_validation([], N, T, P)
        assert n_valid == 0
        assert B_hat == []

    def test_n1_m1_cheater(self):
        """Single voter cheats on single candidate → n_valid=0."""
        B = _ballot_matrix([[2]])
        B_hat, n_valid = algorithm_2_input_validation(B, N, T, P)
        assert n_valid == 0
        assert B_hat == []

    def test_single_candidate(self):
        """M=1; all voters honest."""
        ballots = [[0], [1], [0]]
        B = _ballot_matrix(ballots)
        B_hat, n_valid = algorithm_2_input_validation(B, N, T, P)
        assert n_valid == 3
        assert _reconstruct_ballot(B_hat) == ballots

    def test_b_hat_shares_are_identical_objects_to_original(self):
        """B_hat rows are the original share objects (no copy needed for MPC)."""
        ballots = [[1, 0], [0, 1]]
        B = _ballot_matrix(ballots)
        B_hat, _ = algorithm_2_input_validation(B, N, T, P)
        # The same share list should be reused (not deep-copied)
        assert B_hat[0] is B[0]
        assert B_hat[1] is B[1]
