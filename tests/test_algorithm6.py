"""Tests for Algorithm 6 — One-Hot Winner with Permuted Tie-Break.

Reference: Algorithm 6 in "Fairness Without Exposure: Privacy-Preserving
Phragmén Voting", Section 5.6.

Standard test parameters: p=11, n=3 parties, t=2 threshold.

Field-size constraint (Section 7): cross-products Score_m^num · Score*^den
must not wrap mod p=11.  All test scores use num·den ≤ 9 < 11.
"""

import random

import pytest

from mpc_secret_shares import (
    share,
    reconstruct,
)
from protocol.algorithm6_one_hot import algorithm_6_one_hot_winner
from protocol.types import ScorePair, Shares

P, N, T = 11, 3, 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _share(v: int) -> Shares:
    return share(v, N, T, P)


def _rec(s: Shares) -> int:
    return reconstruct(s[:T], P)


def _score(num: int, den: int) -> ScorePair:
    return _share(num), _share(den)


def _reconstruct_chi(chi: list[Shares]) -> list[int]:
    return [_rec(c) for c in chi]


# ---------------------------------------------------------------------------
# Single candidate
# ---------------------------------------------------------------------------

class TestSingleCandidate:
    def setup_method(self):
        random.seed(10)

    def test_sole_candidate_wins(self):
        """M=1: the only candidate must receive χ=1."""
        scores = [_score(1, 1)]
        win = _score(1, 1)
        chi = algorithm_6_one_hot_winner(scores, win, [0], N, T, P)
        assert len(chi) == 1
        assert _rec(chi[0]) == 1

    def test_m1_second_loop_not_entered(self):
        """M=1: prefix scan loop body never executes; chi has exactly one entry."""
        scores = [_score(2, 3)]
        win = _score(2, 3)
        chi = algorithm_6_one_hot_winner(scores, win, [0], N, T, P)
        assert _reconstruct_chi(chi) == [1]


# ---------------------------------------------------------------------------
# Unique winner (no ties)
# ---------------------------------------------------------------------------

class TestUniqueWinner:
    def setup_method(self):
        random.seed(20)

    def test_first_candidate_wins(self):
        """Winner is candidate 0 (first in index order)."""
        scores = [_score(1, 3), _score(2, 3), _score(3, 3)]
        win = _score(1, 3)
        chi = algorithm_6_one_hot_winner(scores, win, [0, 1, 2], N, T, P)
        assert _reconstruct_chi(chi) == [1, 0, 0]

    def test_middle_candidate_wins(self):
        """Winner is candidate 1 (middle by index)."""
        scores = [_score(3, 3), _score(1, 3), _score(2, 3)]
        win = _score(1, 3)
        chi = algorithm_6_one_hot_winner(scores, win, [0, 1, 2], N, T, P)
        assert _reconstruct_chi(chi) == [0, 1, 0]

    def test_last_candidate_wins(self):
        """Winner is candidate 2 (last by index)."""
        scores = [_score(3, 3), _score(2, 3), _score(1, 3)]
        win = _score(1, 3)
        chi = algorithm_6_one_hot_winner(scores, win, [0, 1, 2], N, T, P)
        assert _reconstruct_chi(chi) == [0, 0, 1]

    def test_one_hot_sum_is_one(self):
        """The sum of all χ values must be exactly 1."""
        scores = [_score(2, 3), _score(1, 3), _score(3, 3)]
        win = _score(1, 3)
        chi = algorithm_6_one_hot_winner(scores, win, [0, 1, 2], N, T, P)
        assert sum(_reconstruct_chi(chi)) == 1


# ---------------------------------------------------------------------------
# Tie-breaking via permutation π
# ---------------------------------------------------------------------------

class TestTieBreaking:
    def setup_method(self):
        random.seed(30)

    def test_tie_first_in_pi_wins(self):
        """All candidates tied: the one that appears first in π wins."""
        scores = [_score(1, 3), _score(1, 3), _score(1, 3)]
        win = _score(1, 3)
        # π puts candidate 2 first → candidate 2 wins
        pi = [2, 0, 1]
        chi = algorithm_6_one_hot_winner(scores, win, pi, N, T, P)
        assert _rec(chi[2]) == 1
        assert _rec(chi[0]) == 0
        assert _rec(chi[1]) == 0

    def test_tie_identity_permutation(self):
        """Two tied candidates, identity permutation: candidate 0 (first) wins."""
        scores = [_score(1, 3), _score(1, 3), _score(2, 3)]
        win = _score(1, 3)
        chi = algorithm_6_one_hot_winner(scores, win, [0, 1, 2], N, T, P)
        assert _rec(chi[0]) == 1
        assert _rec(chi[1]) == 0
        assert _rec(chi[2]) == 0

    def test_tie_reversed_permutation(self):
        """Two tied candidates, reversed permutation: last-by-index wins."""
        scores = [_score(1, 3), _score(1, 3), _score(2, 3)]
        win = _score(1, 3)
        # π = [1, 0, 2] → candidate 1 is first in π among tied candidates
        pi = [1, 0, 2]
        chi = algorithm_6_one_hot_winner(scores, win, pi, N, T, P)
        assert _rec(chi[1]) == 1
        assert _rec(chi[0]) == 0

    def test_prefix_suppression_only_one_winner(self):
        """Even when three candidates tie, exactly one χ=1."""
        scores = [_score(1, 3)] * 3
        win = _score(1, 3)
        chi = algorithm_6_one_hot_winner(scores, win, [0, 1, 2], N, T, P)
        assert sum(_reconstruct_chi(chi)) == 1

    def test_non_tied_candidate_never_wins_via_pi(self):
        """A candidate with Score > Score* gets e_m=0 and cannot win regardless
        of its position in π."""
        scores = [_score(2, 3), _score(1, 3), _score(3, 3)]
        win = _score(1, 3)
        # π puts candidate 0 (score=2/3 ≠ winner) first → still loses
        pi = [0, 1, 2]
        chi = algorithm_6_one_hot_winner(scores, win, pi, N, T, P)
        assert _rec(chi[0]) == 0
        assert _rec(chi[1]) == 1

    def test_winner_last_in_pi_still_wins_if_unique(self):
        """Unique winner placed last in π still wins (no tie, s stays 0 until reached)."""
        scores = [_score(2, 3), _score(3, 3), _score(1, 3)]
        win = _score(1, 3)
        # π = [0, 1, 2]: candidate 2 (the winner) is last
        chi = algorithm_6_one_hot_winner(scores, win, [0, 1, 2], N, T, P)
        assert _rec(chi[2]) == 1
        assert _rec(chi[0]) == 0
        assert _rec(chi[1]) == 0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_empty_scores_raises(self):
        with pytest.raises(ValueError):
            algorithm_6_one_hot_winner([], _score(1, 1), [], N, T, P)

    def test_pi_wrong_length_raises(self):
        scores = [_score(1, 3), _score(2, 3)]
        with pytest.raises(ValueError):
            algorithm_6_one_hot_winner(scores, _score(1, 3), [0], N, T, P)
