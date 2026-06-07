"""Tests for Algorithm 5 — Find Minimum Score.

Reference: Algorithm 5 in "Fairness Without Exposure: Privacy-Preserving
Phragmén Voting", Section 5.5.

Scores are compared as fractions num/den via cross-multiplication:
    Score_a < Score_b  iff  a_num * b_den < b_num * a_den

Standard test parameters: p=11, n=3 parties, t=2 threshold.
"""

import random

import pytest

from mpc_secret_shares import (
    protocol_1_share as share,
    protocol_2_reconstruct as reconstruct,
)
from protocol.algorithm5_find_min import algorithm_5_find_minimum_score
from protocol.types import ScorePair, Shares

P, N, T = 11, 3, 2

# Field-size constraint (paper Section 7): values fed to SecureLT must not
# wrap modulo p.  The cross-products are Score_m^num · best^den and
# best^num · Score_m^den.  With p=11 the maximum safe product is 10, so test
# scores must satisfy num · den ≤ 10.  We keep denominators ≤ 3 (max product
# 3·3 = 9 < 11) to satisfy this constraint without enlarging p.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _share(secret: int) -> Shares:
    return share(secret, N, T, P)


def _rec(shares: Shares) -> int:
    return reconstruct(shares[:T], P)


def _score(num: int, den: int) -> ScorePair:
    """Build a ScorePair from plaintext numerator and denominator."""
    return _share(num), _share(den)


def _reconstruct_score(sp: ScorePair) -> tuple[int, int]:
    return _rec(sp[0]), _rec(sp[1])


# ---------------------------------------------------------------------------
# Single candidate
# ---------------------------------------------------------------------------

class TestSingleCandidate:
    def setup_method(self):
        random.seed(100)

    def test_m1_returns_only_score(self):
        """M=1: no loop executes; the sole score is the winning score."""
        scores = [_score(3, 5)]
        win = algorithm_5_find_minimum_score(scores, N, T, P)
        assert _reconstruct_score(win) == (3, 5)

    def test_empty_raises(self):
        """Empty score list must raise ValueError."""
        with pytest.raises(ValueError):
            algorithm_5_find_minimum_score([], N, T, P)


# ---------------------------------------------------------------------------
# Two candidates — positional variants
# ---------------------------------------------------------------------------

class TestTwoCandidates:
    def setup_method(self):
        random.seed(200)

    def test_first_wins(self):
        """Score_0 = 1/4 < Score_1 = 2/4 → first candidate wins."""
        # 1/4 < 2/4  →  cross: 1·4=4 < 4·2=8  ✓
        scores = [_score(1, 4), _score(2, 4)]
        win = algorithm_5_find_minimum_score(scores, N, T, P)
        assert _reconstruct_score(win) == (1, 4)

    def test_second_wins(self):
        """Score_0 = 2/3 > Score_1 = 1/3 → second candidate wins.

        Cross-products: 1·3=3 < 2·3=6, both ≤ 9 < p=11.
        """
        scores = [_score(2, 3), _score(1, 3)]
        win = algorithm_5_find_minimum_score(scores, N, T, P)
        assert _reconstruct_score(win) == (1, 3)

    def test_equal_scores_first_retained(self):
        """Tied scores: running best is unchanged (β=0), first is retained.

        Cross-products: 2·3=6 == 2·3=6  →  SecureLT(6,6)=0  →  best unchanged.
        """
        scores = [_score(2, 3), _score(2, 3)]
        win = algorithm_5_find_minimum_score(scores, N, T, P)
        assert _reconstruct_score(win) == (2, 3)


# ---------------------------------------------------------------------------
# Three or more candidates
# ---------------------------------------------------------------------------

class TestManyCandidates:
    def setup_method(self):
        random.seed(300)

    def test_minimum_in_middle(self):
        """Minimum score is the middle candidate: 1/3 < 2/3 < 3/3.

        Cross-products stay ≤ 3·3=9 < p=11.
        """
        scores = [_score(3, 3), _score(1, 3), _score(2, 3)]
        win = algorithm_5_find_minimum_score(scores, N, T, P)
        assert _reconstruct_score(win) == (1, 3)

    def test_minimum_at_end(self):
        """Minimum score is the last candidate: 1/3 < 2/3 < 3/3.

        Cross-products stay ≤ 3·3=9 < p=11.
        """
        scores = [_score(3, 3), _score(2, 3), _score(1, 3)]
        win = algorithm_5_find_minimum_score(scores, N, T, P)
        assert _reconstruct_score(win) == (1, 3)

    def test_minimum_at_start(self):
        """Minimum score is the first candidate; all subsequent updates skip.

        Cross-products stay ≤ 3·3=9 < p=11.
        """
        scores = [_score(1, 3), _score(2, 3), _score(3, 3)]
        win = algorithm_5_find_minimum_score(scores, N, T, P)
        assert _reconstruct_score(win) == (1, 3)

    def test_all_tied(self):
        """All scores identical: running best never changes (all β=0).

        Cross-products: 2·3=6 == 2·3=6 everywhere → all β=0.
        """
        scores = [_score(2, 3), _score(2, 3), _score(2, 3)]
        win = algorithm_5_find_minimum_score(scores, N, T, P)
        assert _reconstruct_score(win) == (2, 3)

    def test_unequal_denominators(self):
        """Comparison works correctly when denominators differ.

        Scores: 1/2 = 0.5,  1/3 ≈ 0.333,  1/4 = 0.25
        cross-mult: 1/4 < 1/3 < 1/2  →  winner is score (1, 4).
        """
        scores = [_score(1, 2), _score(1, 3), _score(1, 4)]
        win = algorithm_5_find_minimum_score(scores, N, T, P)
        # 1/4 is smallest: 1·3=3 < 4·1=4 (1/4 < 1/3) ✓  and 1·2=2 < 4·1=4 (1/4 < 1/2) ✓
        assert _reconstruct_score(win) == (1, 4)

    def test_first_period_equal_wallets_max_support_wins(self):
        """In the first period all Score^num = Δ = 1 and Score^den = |A(m)|.

        Candidate with the largest |A(m)| has the smallest Score = 1/|A(m)|
        and should be returned as the minimum.

        Scores: 1/1 (|A|=1), 1/3 (|A|=3), 1/2 (|A|=2)
        Minimum is 1/3 (most supporters).
        """
        scores = [_score(1, 1), _score(1, 3), _score(1, 2)]
        win = algorithm_5_find_minimum_score(scores, N, T, P)
        assert _reconstruct_score(win) == (1, 3)
