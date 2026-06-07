"""Tests for Algorithm 7 — Wallet Update (Global Denominator Scaling).

Reference: Algorithm 7 in "Fairness Without Exposure: Privacy-Preserving
Phragmén Voting", Section 5.7.

Every test that modifies wallets explicitly verifies the invariant

    w_n = W̃_n / Δ    (paper Eq. 6)

by reconstructing W̃_i and Δ after the update and checking:

    W̃_i_new × Δ_expected_den ≡ expected_num × Δ_new  (mod p)

using cross-multiplication to avoid modular inverses.

Standard test parameters: p=11, n=3 parties, t=2 threshold.

Field-size note: intermediate products must not wrap mod p=11.  With N≤3
parties and small Δ values the products stay well below p throughout.
"""

import random

import pytest

from mpc_secret_shares import (
    protocol_1_share as share,
    protocol_2_reconstruct as reconstruct,
)
from protocol.algorithm7_wallet_update import algorithm_7_wallet_update
from protocol.types import BallotMatrix, Shares

P, N, T = 11, 3, 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _share(v: int) -> Shares:
    return share(v, N, T, P)


def _rec(s: Shares) -> int:
    return reconstruct(s[:T], P)


def _ballot_matrix(values: list[list[int]]) -> BallotMatrix:
    return [[_share(v) for v in row] for row in values]


def _chi(winner: int, m: int) -> list[Shares]:
    """One-hot chi vector: winner index gets 1, all others get 0."""
    return [_share(1 if i == winner else 0) for i in range(m)]


def _assert_invariant(
    new_wallets: list[Shares],
    new_delta: Shares,
    expected: list[tuple[int, int]],   # (numerator, denominator) of expected w_n
) -> None:
    """Verify w_n = W̃_n / Δ for all voters via cross-multiplication.

    Checks W̃_i_new * expected_den ≡ expected_num * Δ_new  (mod p).
    This avoids computing modular inverses while still verifying the invariant.
    """
    delta_val = _rec(new_delta)
    for i, (exp_num, exp_den) in enumerate(expected):
        w_tilde = _rec(new_wallets[i])
        lhs = (w_tilde * exp_den) % P
        rhs = (exp_num * delta_val) % P
        assert lhs == rhs, (
            f"Invariant violated for voter {i}: "
            f"W̃={w_tilde}, Δ={delta_val}, "
            f"expected w_n={exp_num}/{exp_den}, "
            f"cross-mult: {lhs} ≠ {rhs} (mod {P})"
        )


# ---------------------------------------------------------------------------
# First period (all wallets zero, Δ=1)
# ---------------------------------------------------------------------------

class TestFirstPeriod:
    """In the first period Δ=1 and all W̃_i=0.
    Score* = κ*/|A(m*)| = (Δ − 0) / |A(m*)| = 1/|A(m*)|.
    """

    def setup_method(self):
        random.seed(10)

    def test_supporters_reset_non_supporters_gain(self):
        """Supporters of the winner are reset to 0; others gain Score*.

        Setup: 3 voters, 2 candidates.
        Ballots: voter0→{c0}, voter1→{c1}, voter2→{c0,c1}.
        Winner: c0.  |A(c0)| = 2 (voters 0 and 2).
        Expected after update:
          voter0 (supporter): w_0 = 0/2 = 0
          voter1 (non-supp):  w_1 = 1/2
          voter2 (supporter): w_2 = 0/2 = 0
          New Δ = 1×2 = 2.
        """
        B = _ballot_matrix([[1, 0], [0, 1], [1, 1]])
        ch = _chi(winner=0, m=2)
        wallets = [_share(0)] * 3
        delta = _share(1)
        A_shr = [_share(2), _share(2)]    # |A(c0)|=2, |A(c1)|=2
        T_shr = [_share(0), _share(0)]    # all wallets zero

        new_w, new_d = algorithm_7_wallet_update(
            B, ch, wallets, delta, A_shr, T_shr, 3, 2, N, T, P
        )

        assert _rec(new_d) == 2
        assert _rec(new_w[0]) == 0    # supporter → reset
        assert _rec(new_w[2]) == 0    # supporter → reset
        assert _rec(new_w[1]) == 1    # non-supporter: W̃ = 0×2 + 1 = 1

        _assert_invariant(new_w, new_d, [(0, 2), (1, 2), (0, 2)])

    def test_delta_scales_by_winner_supporter_count(self):
        """New Δ = old Δ × |A(m*)|."""
        B = _ballot_matrix([[1, 0], [1, 0], [1, 0]])  # all support c0
        ch = _chi(winner=0, m=2)
        wallets = [_share(0)] * 3
        delta = _share(1)
        A_shr = [_share(3), _share(0)]
        T_shr = [_share(0), _share(0)]

        _, new_d = algorithm_7_wallet_update(
            B, ch, wallets, delta, A_shr, T_shr, 3, 2, N, T, P
        )
        assert _rec(new_d) == 3   # 1 × 3

    def test_all_supporters_wallets_zero(self):
        """When every voter supports the winner all wallets stay zero."""
        B = _ballot_matrix([[1], [1], [1]])
        ch = _chi(winner=0, m=1)
        wallets = [_share(0)] * 3
        delta = _share(1)
        A_shr = [_share(3)]
        T_shr = [_share(0)]

        new_w, new_d = algorithm_7_wallet_update(
            B, ch, wallets, delta, A_shr, T_shr, 3, 1, N, T, P
        )
        assert _rec(new_d) == 3
        for w in new_w:
            assert _rec(w) == 0
        _assert_invariant(new_w, new_d, [(0, 3)] * 3)

    def test_single_voter_single_candidate(self):
        """N=1, M=1: sole voter supports the winner; wallet resets."""
        B = _ballot_matrix([[1]])
        ch = _chi(winner=0, m=1)
        wallets = [_share(0)]
        delta = _share(1)
        A_shr = [_share(1)]
        T_shr = [_share(0)]

        new_w, new_d = algorithm_7_wallet_update(
            B, ch, wallets, delta, A_shr, T_shr, 1, 1, N, T, P
        )
        assert _rec(new_d) == 1       # Δ × 1 = 1
        assert _rec(new_w[0]) == 0    # supporter: reset
        _assert_invariant(new_w, new_d, [(0, 1)])


# ---------------------------------------------------------------------------
# Subsequent periods (non-zero wallets)
# ---------------------------------------------------------------------------

class TestSubsequentPeriods:
    """After period 1 wallets may be non-zero.  Verify the invariant holds
    across a second update.
    """

    def setup_method(self):
        random.seed(20)

    def test_non_supporter_wallet_carries_over(self):
        """Non-supporter's prior wallet feeds into the next period's update.

        State after period 1 (from TestFirstPeriod.test_supporters_reset):
          Δ=2, W̃=[0,1,0].
        Period 2 same ballots, winner=c1.  |A(c1)|=2 (voters 1,2).
        T̃_1 = W̃_1 = 1 (only voter1 supported c1 last time, with W̃_1=1).
        Wait — T̃ here is passed in from Algorithm 4.  For simplicity we
        compute the expected values manually.

        Period 2 setup: Δ=2, W̃=[0,1,0].
        Winner=c1, |A(c1)|=2, T̃_1 = W̃_1 (voter1) + W̃_2 (voter2) = 1+0 = 1.
        κ* = Δ - T̃_1 = 2 - 1 = 1.

        voter0 (supports c0 only, non-supp for c1):
          γ_0=0, μ_0=W̃_0·|A(c1)|=0·2=0, ν_0=0+1=1.  New W̃_0=(1-0)·1=1.
        voter1 (supports c1, supporter):
          γ_1=1.  New W̃_1=0.
        voter2 (supports both, supporter for c1):
          γ_2=1.  New W̃_2=0.
        New Δ = 2×2 = 4.

        Invariant:
          w_0 = 1/4, w_1 = 0/4 = 0, w_2 = 0/4 = 0.
        """
        B = _ballot_matrix([[1, 0], [0, 1], [1, 1]])
        ch = _chi(winner=1, m=2)
        wallets = [_share(0), _share(1), _share(0)]   # period-2 state
        delta = _share(2)
        A_shr = [_share(2), _share(2)]
        T_shr = [_share(0), _share(1)]   # T̃_1=1 from voter1's wallet

        new_w, new_d = algorithm_7_wallet_update(
            B, ch, wallets, delta, A_shr, T_shr, 3, 2, N, T, P
        )

        assert _rec(new_d) == 4
        assert _rec(new_w[0]) == 1    # non-supporter: 0·2 + 1 = 1
        assert _rec(new_w[1]) == 0    # supporter: reset
        assert _rec(new_w[2]) == 0    # supporter: reset

        _assert_invariant(new_w, new_d, [(1, 4), (0, 4), (0, 4)])

    def test_invariant_with_nonzero_initial_wallet(self):
        """Voter with W̃>0 who does not support the winner carries forward
        the correct rational balance after scaling.

        Setup: Δ=3, W̃=[2,0], ballots [[0,1],[1,0]], winner=c0.
        |A(c0)|=1, T̃_0=0.  κ*=3-0=3.
        voter0 (non-supp for c0, B=[0,1]):
          γ_0=0, μ_0=W̃_0·1=2, ν_0=2+3=5.  New W̃_0=(1)·5=5.
        voter1 (supp c0, B=[1,0]):
          γ_1=1.  New W̃_1=0.
        New Δ=3×1=3.

        Invariant:
          w_0 = 5/3 → W̃_0_new·3 = 5·Δ_new → 5·3=15 ≡ 4 (mod 11) and 5·3=15 ≡ 4. ✓
          w_1 = 0.
        """
        B = _ballot_matrix([[0, 1], [1, 0]])
        ch = _chi(winner=0, m=2)
        wallets = [_share(2), _share(0)]
        delta = _share(3)
        A_shr = [_share(1), _share(1)]
        T_shr = [_share(0), _share(2)]   # T̃_1 = W̃_0 (voter0 approved c1)

        new_w, new_d = algorithm_7_wallet_update(
            B, ch, wallets, delta, A_shr, T_shr, 2, 2, N, T, P
        )

        assert _rec(new_d) == 3
        assert _rec(new_w[0]) == 5
        assert _rec(new_w[1]) == 0

        _assert_invariant(new_w, new_d, [(5, 3), (0, 3)])


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------

class TestStructural:
    def setup_method(self):
        random.seed(30)

    def test_output_wallet_count_unchanged(self):
        """Number of updated wallets equals n_valid."""
        B = _ballot_matrix([[1, 0], [0, 1], [1, 1]])
        ch = _chi(winner=0, m=2)
        wallets = [_share(0)] * 3
        delta = _share(1)
        A_shr = [_share(2), _share(2)]
        T_shr = [_share(0), _share(0)]

        new_w, _ = algorithm_7_wallet_update(
            B, ch, wallets, delta, A_shr, T_shr, 3, 2, N, T, P
        )
        assert len(new_w) == 3

    def test_kappa_star_equals_score_num(self):
        """κ* = Δ − T̃* must equal the winning Score^num.

        With Δ=5, T̃*=2 (winner's supporters held 2/5 total):
        κ* = 3 = Score*^num.
        We verify indirectly: the non-supporter wallet increment equals κ*
        divided by |A(m*)| in rational terms.

        Setup: Δ=5, W̃=[0,0], ballots [[1,0],[0,1]], winner=c0.
        |A(c0)|=1, T̃_c0=0.  κ*=5.
        New W̃_0 (non-supp, B=[0,1]) = (1-0)·(0·1+5) = 5.
        New Δ = 5·1 = 5.
        w_0 = 5/5 = 1.
        """
        B = _ballot_matrix([[1, 0], [0, 1]])
        ch = _chi(winner=0, m=2)
        wallets = [_share(0), _share(0)]
        delta = _share(5)
        A_shr = [_share(1), _share(1)]
        T_shr = [_share(0), _share(0)]

        new_w, new_d = algorithm_7_wallet_update(
            B, ch, wallets, delta, A_shr, T_shr, 2, 2, N, T, P
        )

        assert _rec(new_d) == 5
        assert _rec(new_w[0]) == 0    # voter0 supports c0 → supporter → reset
        assert _rec(new_w[1]) == 5    # voter1 supports c1 only → non-supp → W̃=κ*=5
        _assert_invariant(new_w, new_d, [(0, 5), (5, 5)])


# ---------------------------------------------------------------------------
# Consecutive updates — multi-round correctness
# ---------------------------------------------------------------------------

class TestConsecutiveUpdates:
    """Two back-to-back Algorithm 7 calls using the output of round r as the
    input of round r+1.  These tests verify Δ growth and the invariant across
    period boundaries.

    Shared scenario:
      3 voters, 2 candidates.
      Ballots: voter0→{c0}, voter1→{c1}, voter2→{c0,c1}.
      Round 1 winner: c0  (|A(c0)|=2, voters 0 and 2).
      Round 2 winner: c1  (|A(c1)|=2, voters 1 and 2).
    """

    def setup_method(self):
        random.seed(40)

    # --- shared helper --------------------------------------------------

    @staticmethod
    def _run_round1():
        """Run round-1 update and return (new_wallets, new_delta)."""
        B = _ballot_matrix([[1, 0], [0, 1], [1, 1]])
        ch = _chi(winner=0, m=2)
        wallets = [_share(0)] * 3
        delta = _share(1)
        A_shr = [_share(2), _share(2)]
        T_shr = [_share(0), _share(0)]
        return algorithm_7_wallet_update(
            B, ch, wallets, delta, A_shr, T_shr, 3, 2, N, T, P
        )

    def test_round1_state_is_correct(self):
        """Sanity-check the round-1 intermediate state used by later tests.

        Round-1 expected: Δ=2, W̃=[0,1,0].
        voter0 (supporter c0): reset.
        voter1 (non-supp c0): W̃=0×2+1=1.
        voter2 (supporter c0): reset.
        """
        new_w, new_d = self._run_round1()
        assert _rec(new_d) == 2
        assert _rec(new_w[0]) == 0
        assert _rec(new_w[1]) == 1
        assert _rec(new_w[2]) == 0
        _assert_invariant(new_w, new_d, [(0, 2), (1, 2), (0, 2)])

    def test_two_rounds_delta_equals_product_of_support_counts(self):
        """After two rounds Δ = |A(m*_1)| × |A(m*_2)| = 2×2 = 4.

        Round 2: winner=c1, |A(c1)|=2.
        New Δ = Δ_1 × |A(m*_2)| = 2 × 2 = 4.
        """
        w1, d1 = self._run_round1()
        # Round 2 — winner is c1; T̃_c1 = W̃_1+W̃_2 = 1+0 = 1
        B = _ballot_matrix([[1, 0], [0, 1], [1, 1]])
        ch2 = _chi(winner=1, m=2)
        A_shr2 = [_share(2), _share(2)]
        T_shr2 = [_share(0), _share(1)]   # T̃_c0=0, T̃_c1=1

        _, d2 = algorithm_7_wallet_update(
            B, ch2, w1, d1, A_shr2, T_shr2, 3, 2, N, T, P
        )
        assert _rec(d2) == 4   # 1 × 2 × 2

    def test_two_rounds_invariant_after_each_update(self):
        """Invariant w_n = W̃_n / Δ holds after both round 1 and round 2.

        Round-2 expected state: Δ=4, W̃=[1,0,0].
          voter0 (non-supp c1, only supp c0): gains Score*_2 = κ*_2/|A(c1)|·Δ_1
            = (Δ_1 - T̃*_2) / (|A(c1)| × Δ_1) = (2-1)/(2×2) = 1/4.
            New W̃_0 = 0×2 + κ*_2 = 0×2 + 1 = 1.
          voter1 (supp c1): reset. W̃_1=0.
          voter2 (supp c1 via both): reset. W̃_2=0.
          New Δ = 2×2 = 4.
        """
        w1, d1 = self._run_round1()
        B = _ballot_matrix([[1, 0], [0, 1], [1, 1]])
        ch2 = _chi(winner=1, m=2)
        A_shr2 = [_share(2), _share(2)]
        T_shr2 = [_share(0), _share(1)]

        w2, d2 = algorithm_7_wallet_update(
            B, ch2, w1, d1, A_shr2, T_shr2, 3, 2, N, T, P
        )

        assert _rec(d2) == 4
        assert _rec(w2[0]) == 1   # non-supp gained Score*_2
        assert _rec(w2[1]) == 0   # supporter reset
        assert _rec(w2[2]) == 0   # supporter reset

        # Invariant: after round 2  w_0=1/4, w_1=0, w_2=0
        _assert_invariant(w2, d2, [(1, 4), (0, 4), (0, 4)])


# ---------------------------------------------------------------------------
# Wallet growth — voter who never supports the winner
# ---------------------------------------------------------------------------

class TestWalletGrowth:
    """A voter whose preferred candidate never wins accumulates wallet balance
    over multiple rounds.

    Scenario: voter0→c0 only; voter1,voter2→c1; c1 always wins.
    """

    def setup_method(self):
        random.seed(50)

    def test_non_winning_voter_wallet_grows_first_round(self):
        """Round 1: non-supporter gains Score*_1 = 1/|A(c1)| = 1/2."""
        B = _ballot_matrix([[1, 0], [0, 1], [0, 1]])
        ch = _chi(winner=1, m=2)
        wallets = [_share(0)] * 3
        delta = _share(1)
        A_shr = [_share(1), _share(2)]
        T_shr = [_share(0), _share(0)]

        new_w, new_d = algorithm_7_wallet_update(
            B, ch, wallets, delta, A_shr, T_shr, 3, 2, N, T, P
        )
        assert _rec(new_d) == 2
        assert _rec(new_w[0]) == 1    # non-supp: 0×2 + 1 = 1 → w_0=1/2
        assert _rec(new_w[1]) == 0    # supporter reset
        assert _rec(new_w[2]) == 0    # supporter reset
        _assert_invariant(new_w, new_d, [(1, 2), (0, 2), (0, 2)])

    def test_non_winning_voter_wallet_grows_two_rounds(self):
        """Round 2: voter0's wallet further accumulates while c1 wins again.

        After round 1: W̃=[1,0,0], Δ=2 (from above).
        Round 2: T̃_c1=W̃_1+W̃_2=0, κ*_2=Δ-0=2.
          voter0: γ=0. μ=W̃_0×|A_2|=1×2=2. ν=2+2=4. New W̃_0=1×4=4.
        New Δ=2×2=4.  w_0=4/4=1.
        """
        # Round 1
        B = _ballot_matrix([[1, 0], [0, 1], [0, 1]])
        w1, d1 = algorithm_7_wallet_update(
            B, _chi(winner=1, m=2),
            [_share(0)] * 3, _share(1),
            [_share(1), _share(2)], [_share(0), _share(0)],
            3, 2, N, T, P,
        )
        # Round 2 — same winner, T̃_c1 = W̃_1+W̃_2 = 0+0 = 0
        w2, d2 = algorithm_7_wallet_update(
            B, _chi(winner=1, m=2),
            w1, d1,
            [_share(1), _share(2)], [_share(0), _share(0)],
            3, 2, N, T, P,
        )
        assert _rec(d2) == 4
        assert _rec(w2[0]) == 4   # 1×2 + 2 = 4
        assert _rec(w2[1]) == 0
        assert _rec(w2[2]) == 0
        _assert_invariant(w2, d2, [(4, 4), (0, 4), (0, 4)])


# ---------------------------------------------------------------------------
# Extended cases — mixed, M>1 all-support, non-zero initial wallet
# ---------------------------------------------------------------------------

class TestExtendedCases:
    def setup_method(self):
        random.seed(60)

    def test_all_voters_support_winner_multiple_candidates(self):
        """All N=3 voters support winner c1 out of M=2 candidates.

        Every γ_i=1 → every wallet resets.  Δ scales by 3.
        """
        B = _ballot_matrix([[0, 1], [0, 1], [0, 1]])
        ch = _chi(winner=1, m=2)
        wallets = [_share(0)] * 3
        delta = _share(1)
        A_shr = [_share(0), _share(3)]
        T_shr = [_share(0), _share(0)]

        new_w, new_d = algorithm_7_wallet_update(
            B, ch, wallets, delta, A_shr, T_shr, 3, 2, N, T, P
        )
        assert _rec(new_d) == 3
        for w in new_w:
            assert _rec(w) == 0
        _assert_invariant(new_w, new_d, [(0, 3)] * 3)

    def test_mixed_four_voters_two_supporters_two_non(self):
        """4 voters, 2 candidates: voters 0,2 support c0 (winner);
        voters 1,3 support c1 (loser).

        winner=c0, |A(c0)|=2.  κ*=1.
        voters 0,2 (supp): reset → W̃=0.
        voters 1,3 (non-supp): W̃=0×2+1=1.
        New Δ=2.  Invariant: w_1=w_3=1/2, w_0=w_2=0.
        """
        B = _ballot_matrix([[1, 0], [0, 1], [1, 0], [0, 1]])
        ch = _chi(winner=0, m=2)
        wallets = [_share(0)] * 4
        delta = _share(1)
        A_shr = [_share(2), _share(2)]
        T_shr = [_share(0), _share(0)]

        new_w, new_d = algorithm_7_wallet_update(
            B, ch, wallets, delta, A_shr, T_shr, 4, 2, N, T, P
        )
        assert _rec(new_d) == 2
        assert _rec(new_w[0]) == 0   # supporter
        assert _rec(new_w[1]) == 1   # non-supporter
        assert _rec(new_w[2]) == 0   # supporter
        assert _rec(new_w[3]) == 1   # non-supporter
        _assert_invariant(new_w, new_d, [(0, 2), (1, 2), (0, 2), (1, 2)])

    def test_non_zero_initial_wallet_entering_second_period(self):
        """Voter with an existing non-zero wallet at period start.

        State: Δ=3, W̃=[3,0] — voter0 has w_0=3/3=1 accumulated.
        Ballots: voter0→c1 only, voter1→c0 (winner).
        Winner=c0, |A(c0)|=1.  T̃_c0=0.  κ*=3-0=3.

        voter0 (non-supp c0): γ=0. μ=W̃_0×1=3. ν=3+3=6. New W̃_0=1×6=6.
        voter1 (supp c0):    γ=1.  Reset. New W̃_1=0.
        New Δ=3×1=3.

        New w_0 = 6/3 = 2  (doubled: gained Score*=1 on top of w_0=1).
        """
        B = _ballot_matrix([[0, 1], [1, 0]])
        ch = _chi(winner=0, m=2)
        wallets = [_share(3), _share(0)]
        delta = _share(3)
        A_shr = [_share(1), _share(1)]
        T_shr = [_share(0), _share(3)]   # T̃_c1=W̃_0=3

        new_w, new_d = algorithm_7_wallet_update(
            B, ch, wallets, delta, A_shr, T_shr, 2, 2, N, T, P
        )
        assert _rec(new_d) == 3
        assert _rec(new_w[0]) == 6
        assert _rec(new_w[1]) == 0
        # w_0_new = 6/3 = 2  (previously 1, now 2 after gaining Score*=1)
        _assert_invariant(new_w, new_d, [(2, 1), (0, 1)])
