"""Tests for Algorithm 4 — Oblivious Score Computation.

Reference: Algorithm 4 in "Fairness Without Exposure: Privacy-Preserving
Phragmén Voting", Section 5.4.

Standard test parameters: p=11, n=3 parties, t=2 threshold.

Score representation (paper Eq. 4):
    Score_m^num = Δ − T̃_m
    Score_m^den = |A(m)| · Δ
"""

import random

import pytest

from mpc_secret_shares import (
    share,
    reconstruct,
)
from protocol.algorithm4_score import algorithm_4_score_computation
from protocol.types import BallotMatrix, Shares

P, N, T = 11, 3, 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _share(secret: int) -> Shares:
    return share(secret, N, T, P)


def _rec(shares: Shares) -> int:
    return reconstruct(shares[:T], P)


def _ballot_matrix(values: list[list[int]]) -> BallotMatrix:
    return [[_share(v) for v in row] for row in values]


def _wallets(values: list[int]) -> list[Shares]:
    return [_share(v) for v in values]


# ---------------------------------------------------------------------------
# First-period baseline (all wallets zero, Δ=1)
# ---------------------------------------------------------------------------

class TestFirstPeriod:
    """In the first period Δ=1 and all W̃_i=0, so T̃_m=0 for every candidate.
    Score_m^num = 1, Score_m^den = |A(m)|  →  Score_m = 1/|A(m)|.
    """

    def setup_method(self):
        random.seed(10)

    def test_T_tilde_zero_when_wallets_zero(self):
        """T̃_m = 0 when all wallets are zero."""
        B = _ballot_matrix([[1, 0], [0, 1], [1, 1]])
        wallets = _wallets([0, 0, 0])
        delta = _share(1)
        scores, A_shares, T_tilde = algorithm_4_score_computation(
            B, wallets, delta, 3, 2, N, T, P
        )
        for T_m in T_tilde:
            assert _rec(T_m) == 0

    def test_score_num_equals_delta_when_T_tilde_zero(self):
        """Score_m^num = Δ − 0 = 1 for all candidates in the first period."""
        B = _ballot_matrix([[1, 0], [1, 1], [0, 1]])
        wallets = _wallets([0, 0, 0])
        delta = _share(1)
        scores, _, _ = algorithm_4_score_computation(
            B, wallets, delta, 3, 2, N, T, P
        )
        for score_num, _ in scores:
            assert _rec(score_num) == 1

    def test_score_den_equals_support_count(self):
        """Score_m^den = |A(m)| · Δ = |A(m)| when Δ=1."""
        # Ballots: voter 0 → {0}, voter 1 → {0,1}, voter 2 → {1}
        # |A(0)| = 2, |A(1)| = 2
        B = _ballot_matrix([[1, 0], [1, 1], [0, 1]])
        wallets = _wallets([0, 0, 0])
        delta = _share(1)
        scores, _, _ = algorithm_4_score_computation(
            B, wallets, delta, 3, 2, N, T, P
        )
        assert _rec(scores[0][1]) == 2   # |A(0)| · 1
        assert _rec(scores[1][1]) == 2   # |A(1)| · 1

    def test_more_supporters_lower_score_den(self):
        """Candidate with fewer supporters has larger denominator score → worse."""
        # |A(0)| = 3, |A(1)| = 1
        B = _ballot_matrix([[1, 1], [1, 0], [1, 0]])
        wallets = _wallets([0, 0, 0])
        delta = _share(1)
        scores, _, _ = algorithm_4_score_computation(
            B, wallets, delta, 3, 2, N, T, P
        )
        # Score_0 = 1/3, Score_1 = 1/1 → Score_0 < Score_1 → candidate 0 wins
        assert _rec(scores[0][1]) == 3   # |A(0)|
        assert _rec(scores[1][1]) == 1   # |A(1)|


# ---------------------------------------------------------------------------
# Support-count accumulation
# ---------------------------------------------------------------------------

class TestSupportCount:
    def setup_method(self):
        random.seed(20)

    def test_A_zero_for_unsupported_candidate(self):
        """Candidate with no supporters has |A(m)| = 0."""
        B = _ballot_matrix([[1, 0], [1, 0]])
        wallets = _wallets([0, 0])
        delta = _share(1)
        _, A_shares, _ = algorithm_4_score_computation(
            B, wallets, delta, 2, 2, N, T, P
        )
        assert _rec(A_shares[0]) == 2
        assert _rec(A_shares[1]) == 0

    def test_A_counts_all_supporters(self):
        """|A(m)| equals the number of voters with B_{i,m}=1."""
        B = _ballot_matrix([[1, 1, 0], [0, 1, 1], [1, 1, 1]])
        wallets = _wallets([0, 0, 0])
        delta = _share(1)
        _, A_shares, _ = algorithm_4_score_computation(
            B, wallets, delta, 3, 3, N, T, P
        )
        assert _rec(A_shares[0]) == 2   # voters 0 and 2
        assert _rec(A_shares[1]) == 3   # all voters
        assert _rec(A_shares[2]) == 2   # voters 1 and 2


# ---------------------------------------------------------------------------
# Wallet contribution (T̃_m)
# ---------------------------------------------------------------------------

class TestWalletContribution:
    def setup_method(self):
        random.seed(30)

    def test_T_tilde_sum_of_supporter_wallets(self):
        """T̃_m = sum of W̃_i for supporters of m.

        Setup: Δ=4, W̃=[1,2,0], B=[[1,0],[0,1],[1,0]]
        T̃_0 = W̃_0 + W̃_2 = 1 + 0 = 1
        T̃_1 = W̃_1         = 2
        """
        B = _ballot_matrix([[1, 0], [0, 1], [1, 0]])
        wallets = _wallets([1, 2, 0])
        delta = _share(4)
        _, _, T_tilde = algorithm_4_score_computation(
            B, wallets, delta, 3, 2, N, T, P
        )
        assert _rec(T_tilde[0]) == 1
        assert _rec(T_tilde[1]) == 2

    def test_score_num_equals_delta_minus_T_tilde(self):
        """Score_m^num = Δ − T̃_m verified against direct arithmetic.

        Δ=5, W̃=[2,1], B=[[1,0],[1,1]]
        T̃_0 = 2+1 = 3, Score_0^num = 5-3 = 2
        T̃_1 = 0+1 = 1, Score_1^num = 5-1 = 4
        """
        B = _ballot_matrix([[1, 0], [1, 1]])
        wallets = _wallets([2, 1])
        delta = _share(5)
        scores, _, T_tilde = algorithm_4_score_computation(
            B, wallets, delta, 2, 2, N, T, P
        )
        assert _rec(T_tilde[0]) == 3
        assert _rec(scores[0][0]) == 2    # 5 − 3
        assert _rec(T_tilde[1]) == 1
        assert _rec(scores[1][0]) == 4    # 5 − 1

    def test_score_den_equals_delta_times_A(self):
        """Score_m^den = Δ · |A(m)| verified against direct arithmetic.

        Δ=3, B=[[1,0],[1,1],[0,1]], |A(0)|=2, |A(1)|=2
        Score_0^den = 3·2 = 6, Score_1^den = 3·2 = 6
        """
        B = _ballot_matrix([[1, 0], [1, 1], [0, 1]])
        wallets = _wallets([0, 0, 0])
        delta = _share(3)
        scores, A_shares, _ = algorithm_4_score_computation(
            B, wallets, delta, 3, 2, N, T, P
        )
        assert _rec(A_shares[0]) == 2
        assert _rec(scores[0][1]) == (3 * 2) % P   # Δ · |A(0)|
        assert _rec(A_shares[1]) == 2
        assert _rec(scores[1][1]) == (3 * 2) % P   # Δ · |A(1)|

    def test_non_supporter_wallet_excluded(self):
        """A voter's wallet does not contribute to T̃_m when B_{i,m}=0."""
        # voter 0: W̃=5, supports only candidate 1 (not 0)
        B = _ballot_matrix([[0, 1]])
        wallets = _wallets([5])
        delta = _share(1)
        _, _, T_tilde = algorithm_4_score_computation(
            B, wallets, delta, 1, 2, N, T, P
        )
        assert _rec(T_tilde[0]) == 0   # voter 0 doesn't support candidate 0
        assert _rec(T_tilde[1]) == 5   # voter 0 supports candidate 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def setup_method(self):
        random.seed(40)

    def test_void_election_outside_paper_assumption(self):
        """OUTSIDE PAPER ASSUMPTIONS (Section 5, page 8).

        The paper requires at least one candidate to have |A(m)| > 0.
        A void election (n_valid=0) is explicitly excluded.

        This test only documents that the function does not crash and that
        every Score^den collapses to 0, making Algorithm 5's comparison
        undefined.  Do not use this case in a real election run.
        """
        delta = _share(1)
        scores, A_shares, T_tilde = algorithm_4_score_computation(
            [], [], delta, 0, 3, N, T, P
        )
        # Structural invariant: M output entries regardless of voter count.
        assert len(scores) == 3
        assert len(A_shares) == 3
        assert len(T_tilde) == 3
        # All counts and sums are zero — Score^den = 0 for every candidate.
        for m in range(3):
            assert _rec(A_shares[m]) == 0
            assert _rec(T_tilde[m]) == 0
            assert _rec(scores[m][1]) == 0   # Δ · 0 = 0 — comparison undefined

    def test_zero_candidates_returns_empty_lists(self):
        """num_candidates=0 → all output lists are empty."""
        wallets = _wallets([0, 0])
        scores, A_shares, T_tilde = algorithm_4_score_computation(
            [[], []], wallets, _share(1), 2, 0, N, T, P
        )
        assert scores == []
        assert A_shares == []
        assert T_tilde == []

    def test_single_voter_single_candidate(self):
        """Minimal N=1, M=1 case: supporter wallet feeds directly into T̃."""
        B = _ballot_matrix([[1]])
        wallets = _wallets([3])
        delta = _share(7)
        scores, A_shares, T_tilde = algorithm_4_score_computation(
            B, wallets, delta, 1, 1, N, T, P
        )
        assert _rec(A_shares[0]) == 1
        assert _rec(T_tilde[0]) == 3
        assert _rec(scores[0][0]) == (7 - 3) % P   # Δ − T̃
        assert _rec(scores[0][1]) == (7 * 1) % P   # Δ · |A|

    def test_output_lengths_match_num_candidates(self):
        """All three output lists have length M."""
        B = _ballot_matrix([[1, 0, 1, 0], [0, 1, 0, 1]])
        wallets = _wallets([0, 0])
        delta = _share(1)
        scores, A_shares, T_tilde = algorithm_4_score_computation(
            B, wallets, delta, 2, 4, N, T, P
        )
        assert len(scores) == 4
        assert len(A_shares) == 4
        assert len(T_tilde) == 4

    def test_zero_support_candidate_score_den_zero(self):
        """Candidate with no supporters: Score^den = Δ · 0 = 0.

        Remark 5.1: the MPC handles this obliviously — no branch on zero.
        """
        B = _ballot_matrix([[1, 0], [1, 0]])
        wallets = _wallets([0, 0])
        delta = _share(2)
        scores, A_shares, _ = algorithm_4_score_computation(
            B, wallets, delta, 2, 2, N, T, P
        )
        assert _rec(A_shares[1]) == 0
        assert _rec(scores[1][1]) == 0   # Δ · 0 = 0
