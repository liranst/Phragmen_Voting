"""MPC Enforcement Tests — Verify no algorithm operates on plaintext secrets.

For each algorithm (2–8) this module:
  1. Confirms that passing a raw integer instead of a Shares object raises
     TypeError (primitives are not callable on plain ints).
  2. Verifies that every output is a Shares object — a list of (int, int)
     tuples — and never a bare integer.
  3. Specifically for Algorithm 5: proves that protocol_2_reconstruct is
     never called mid-protocol by running under a mock that would raise if
     invoked.

Standard MPC parameters: p=11 (prime), n=3 parties, t=2 threshold.
"""

import random
from typing import List
from unittest.mock import patch

import pytest

from mpc_secret_shares import (
    protocol_1_share as share,
    protocol_2_reconstruct as reconstruct,
    shares_one,
    shares_zero,
)
from protocol.algorithm2_validation import algorithm_2_input_validation
from protocol.algorithm3_initialization import algorithm_3_initialization
from protocol.algorithm4_score import algorithm_4_score_computation
from protocol.algorithm5_find_min import algorithm_5_find_minimum_score
from protocol.algorithm6_one_hot import algorithm_6_one_hot_winner
from protocol.algorithm7_wallet_update import algorithm_7_wallet_update
from protocol.algorithm8_reconstruct_winner import algorithm_8_reconstruct_winner
from protocol.types import BallotMatrix, ScorePair, Shares

P, N, T = 11, 3, 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _s(v: int) -> Shares:
    return share(v, N, T, P)


def _ballot(values: list[list[int]]) -> BallotMatrix:
    return [[_s(v) for v in row] for row in values]


def _score(num: int, den: int) -> ScorePair:
    return _s(num), _s(den)


def _is_shares(obj) -> bool:
    """Return True if obj is a non-empty list of (int, int) tuples."""
    return (
        isinstance(obj, list)
        and len(obj) > 0
        and all(isinstance(e, tuple) and len(e) == 2
                and all(isinstance(x, int) for x in e)
                for e in obj)
    )


# ---------------------------------------------------------------------------
# Algorithm 2
# ---------------------------------------------------------------------------

class TestAlgorithm2Enforcement:
    def setup_method(self):
        random.seed(42)

    def test_raw_int_ballot_raises_typeerror(self):
        """Passing a raw integer as a ballot entry (instead of Shares) must
        raise TypeError when secure_mult tries to iterate over it."""
        raw_ballot: BallotMatrix = [[5, 0], [0, 1]]   # integers, not Shares
        with pytest.raises(TypeError):
            algorithm_2_input_validation(raw_ballot, N, T, P)

    def test_output_b_hat_entries_are_shares(self):
        """Every entry in the returned B_hat must be a Shares object, not an int."""
        B = _ballot([[1, 0], [0, 1]])
        B_hat, n_valid = algorithm_2_input_validation(B, N, T, P)
        assert n_valid == 2
        for row in B_hat:
            for entry in row:
                assert _is_shares(entry), (
                    f"B_hat entry is {type(entry).__name__}, expected Shares (list of tuples)"
                )


# ---------------------------------------------------------------------------
# Algorithm 3
# ---------------------------------------------------------------------------

class TestAlgorithm3Enforcement:
    def test_output_delta_is_shares(self):
        """The returned Δ must be a Shares object, never a plain integer."""
        delta, wallets = algorithm_3_initialization(3, N, T, P)
        assert _is_shares(delta), (
            f"delta is {type(delta).__name__}, expected Shares"
        )

    def test_output_wallets_are_shares(self):
        """Every returned wallet must be a Shares object, never a plain integer."""
        delta, wallets = algorithm_3_initialization(3, N, T, P)
        for i, w in enumerate(wallets):
            assert _is_shares(w), (
                f"wallet[{i}] is {type(w).__name__}, expected Shares"
            )

    def test_raw_int_n_valid_still_produces_shares(self):
        """n_valid is a public integer (not a secret); outputs are still Shares."""
        delta, wallets = algorithm_3_initialization(1, N, T, P)
        assert _is_shares(delta)
        assert _is_shares(wallets[0])


# ---------------------------------------------------------------------------
# Algorithm 4
# ---------------------------------------------------------------------------

class TestAlgorithm4Enforcement:
    def setup_method(self):
        random.seed(10)

    def test_raw_int_wallet_raises_typeerror(self):
        """Passing a raw integer as a wallet share must raise TypeError."""
        B = _ballot([[1, 0], [0, 1]])
        raw_wallets = [5, 3]           # integers, not Shares
        delta = _s(1)
        with pytest.raises(TypeError):
            algorithm_4_score_computation(B, raw_wallets, delta, 2, 2, N, T, P)

    def test_raw_int_delta_raises_typeerror(self):
        """Passing a raw integer as Δ must raise TypeError."""
        B = _ballot([[1, 0], [0, 1]])
        wallets = [_s(0), _s(0)]
        with pytest.raises(TypeError):
            algorithm_4_score_computation(B, wallets, 7, 2, 2, N, T, P)

    def test_score_outputs_are_shares(self):
        """Every Score^num and Score^den in the returned list must be Shares."""
        B = _ballot([[1, 0], [0, 1], [1, 1]])
        wallets = [_s(0)] * 3
        delta = _s(1)
        scores, A_shares, T_tilde = algorithm_4_score_computation(
            B, wallets, delta, 3, 2, N, T, P
        )
        for m, (num, den) in enumerate(scores):
            assert _is_shares(num), f"scores[{m}].num is {type(num).__name__}"
            assert _is_shares(den), f"scores[{m}].den is {type(den).__name__}"
        for m, a in enumerate(A_shares):
            assert _is_shares(a), f"A_shares[{m}] is {type(a).__name__}"
        for m, t_m in enumerate(T_tilde):
            assert _is_shares(t_m), f"T_tilde[{m}] is {type(t_m).__name__}"


# ---------------------------------------------------------------------------
# Algorithm 5 — also verifies no reconstruct() is called mid-protocol
# ---------------------------------------------------------------------------

class TestAlgorithm5Enforcement:
    def setup_method(self):
        random.seed(20)

    def test_raw_int_score_raises_typeerror(self):
        """Passing raw integers as score shares must raise TypeError."""
        raw_scores = [(3, 5), (1, 4)]   # plain int tuples, not ScorePairs of Shares
        with pytest.raises(TypeError):
            algorithm_5_find_minimum_score(raw_scores, N, T, P)

    def test_output_is_shares(self):
        """The winning (best^num, best^den) tuple must contain Shares, not ints."""
        scores = [_score(1, 3), _score(2, 3), _score(3, 3)]
        win_num, win_den = algorithm_5_find_minimum_score(scores, N, T, P)
        assert _is_shares(win_num), f"win_num is {type(win_num).__name__}, expected Shares"
        assert _is_shares(win_den), f"win_den is {type(win_den).__name__}, expected Shares"

    def test_no_reconstruct_called_during_find_min(self):
        """Algorithm 5 must never call reconstruct() — it operates fully in the
        secret-shared domain so no party learns individual scores.

        The mock raises AssertionError if protocol_2_reconstruct is invoked;
        a clean return proves the algorithm is reconstruct-free.
        """
        scores = [_score(3, 3), _score(1, 3), _score(2, 3)]

        def _reconstruct_forbidden(*args, **kwargs):
            raise AssertionError(
                "reconstruct() was called inside algorithm_5_find_minimum_score — "
                "this leaks individual scores to the parties."
            )

        with patch(
            "protocol.algorithm5_find_min.secure_lt",
            wraps=__import__(
                "mpc_secret_shares",
                fromlist=["protocol_5_secure_compare"]
            ).protocol_5_secure_compare,
        ), patch(
            "mpc_secret_shares.protocol_2_reconstruct",
            side_effect=_reconstruct_forbidden,
        ):
            win_num, win_den = algorithm_5_find_minimum_score(scores, N, T, P)

        # Reconstructing the output here (outside the mock scope) confirms the
        # algorithm returned valid shares and produced the correct winner.
        assert reconstruct(win_num[:T], P) == 1
        assert reconstruct(win_den[:T], P) == 3

    def test_correct_winner_returned_as_shares(self):
        """Minimum score 1/4 < 1/3 < 1/2 is found and returned as Shares."""
        scores = [_score(1, 2), _score(1, 3), _score(1, 4)]
        win_num, win_den = algorithm_5_find_minimum_score(scores, N, T, P)
        assert reconstruct(win_num[:T], P) == 1
        assert reconstruct(win_den[:T], P) == 4


# ---------------------------------------------------------------------------
# Algorithm 6
# ---------------------------------------------------------------------------

class TestAlgorithm6Enforcement:
    def setup_method(self):
        random.seed(30)

    def test_raw_int_score_raises_typeerror(self):
        """Passing raw integers as score shares must raise TypeError."""
        raw_scores = [(1, 3), (2, 3)]
        win = _score(1, 3)
        with pytest.raises(TypeError):
            algorithm_6_one_hot_winner(raw_scores, win, [0, 1], N, T, P)

    def test_raw_int_winning_score_raises_typeerror(self):
        """Passing raw integers as the winning score must raise TypeError."""
        scores = [_score(1, 3), _score(2, 3)]
        with pytest.raises(TypeError):
            algorithm_6_one_hot_winner(scores, (1, 3), [0, 1], N, T, P)

    def test_chi_entries_are_shares(self):
        """Every χ_m in the returned list must be a Shares object, not an int."""
        scores = [_score(1, 3), _score(2, 3), _score(3, 3)]
        win = _score(1, 3)
        chi = algorithm_6_one_hot_winner(scores, win, [0, 1, 2], N, T, P)
        assert len(chi) == 3
        for m, c in enumerate(chi):
            assert _is_shares(c), f"chi[{m}] is {type(c).__name__}, expected Shares"


# ---------------------------------------------------------------------------
# Algorithm 7
# ---------------------------------------------------------------------------

class TestAlgorithm7Enforcement:
    def setup_method(self):
        random.seed(40)

    def test_raw_int_chi_raises_typeerror(self):
        """Passing raw integers as chi entries must raise TypeError."""
        B = _ballot([[1, 0], [0, 1]])
        raw_chi = [1, 0]              # integers, not Shares
        wallets = [_s(0), _s(0)]
        delta = _s(1)
        A_shr = [_s(1), _s(1)]
        T_shr = [_s(0), _s(0)]
        with pytest.raises(TypeError):
            algorithm_7_wallet_update(
                B, raw_chi, wallets, delta, A_shr, T_shr, 2, 2, N, T, P
            )

    def test_raw_int_wallet_raises_typeerror(self):
        """Passing raw integers as wallets must raise TypeError."""
        B = _ballot([[1, 0], [0, 1]])
        chi = [_s(1), _s(0)]
        raw_wallets = [5, 3]          # integers, not Shares
        delta = _s(1)
        A_shr = [_s(1), _s(1)]
        T_shr = [_s(0), _s(0)]
        with pytest.raises(TypeError):
            algorithm_7_wallet_update(
                B, chi, raw_wallets, delta, A_shr, T_shr, 2, 2, N, T, P
            )

    def test_output_wallets_are_shares(self):
        """Every returned wallet and the new Δ must be Shares, not plain ints."""
        B = _ballot([[1, 0], [0, 1], [1, 1]])
        chi = [_s(1), _s(0)]
        wallets = [_s(0)] * 3
        delta = _s(1)
        A_shr = [_s(2), _s(2)]
        T_shr = [_s(0), _s(0)]
        new_w, new_d = algorithm_7_wallet_update(
            B, chi, wallets, delta, A_shr, T_shr, 3, 2, N, T, P
        )
        assert _is_shares(new_d), f"new_delta is {type(new_d).__name__}, expected Shares"
        for i, w in enumerate(new_w):
            assert _is_shares(w), f"new_wallet[{i}] is {type(w).__name__}, expected Shares"


# ---------------------------------------------------------------------------
# Algorithm 8
# ---------------------------------------------------------------------------

class TestAlgorithm8Enforcement:
    def setup_method(self):
        random.seed(50)

    def test_raw_int_chi_raises_typeerror(self):
        """Passing raw integers instead of Shares must raise TypeError in
        reconstruct (which expects a list of tuples, not an int)."""
        raw_chi = [1, 0, 0]          # plain ints, not Shares
        with pytest.raises(TypeError):
            algorithm_8_reconstruct_winner(raw_chi, T, P)

    def test_output_is_plain_int(self):
        """Algorithm 8 is the authorised final reveal — its output must be a
        plain int (the winner index), not Shares."""
        chi = [_s(1 if i == 1 else 0) for i in range(3)]
        result = algorithm_8_reconstruct_winner(chi, T, P)
        assert isinstance(result, int), (
            f"winner is {type(result).__name__}, expected int"
        )
        assert not _is_shares(result), "winner must be a plain int, not Shares"
        assert result == 1
