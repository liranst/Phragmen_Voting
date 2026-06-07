"""Integration test — 2-period election with ballot change in period 2.

Reference: "Fairness Without Exposure: Privacy-Preserving Phragmén Voting",
Section 5 (full per-period loop: Algorithms 1–8).

This test exercises the complete protocol pipeline end-to-end across two
election periods and verifies three properties:

1. Correctness — each period elects the candidate with the minimum score.
2. Ballot independence — voters may change their votes between periods;
   the fresh ballot matrix in period 2 replaces period 1's matrix entirely.
3. State continuity — wallet balances accumulated in period 1 influence
   the outcome of period 2 even after voter 0 changes their vote.

Election scenario
-----------------
  N = 3 voters  (V0, V1, V2)
  M = 2 candidates  (C0, C1)
  D = 3 talliers,  threshold t = 2

  Period 1 ballots:
    V0 → {C0}           V0 approves C0 only
    V1 → {C0}           V1 approves C0 only
    V2 → {C1}           V2 approves C1 only

  Period 1 outcome:
    |A(C0)| = 2,  Score_C0 = 1/2
    |A(C1)| = 1,  Score_C1 = 1/1
    C0 wins (lower score).  V0 and V1 wallets reset; V2 gains Score* = 1/2.

  Period 2 ballots  (V0 changes their vote):
    V0 → {C1}           ← CHANGED from {C0} to {C1}
    V1 → {C0}           unchanged
    V2 → {C1}           unchanged

  Period 2 outcome (using wallets carried from period 1):
    |A(C0)| = 1,  T̃_C0 = W̃_V1 = 0   →  Score_C0 = (2-0)/(1×2) = 1
    |A(C1)| = 2,  T̃_C1 = W̃_V0+W̃_V2 = 0+1 = 1
                                     →  Score_C1 = (2-1)/(2×2) = 1/4
    C1 wins (lower score).  V0 and V2 wallets reset; V1 gains balance.

Field-size note: p=251 is used so that all cross-products in Algorithms
5 and 6 stay well below p across both periods.
"""

import os
import random
import tempfile

import pytest

from mpc_secret_shares import (
    protocol_1_share as share,
    protocol_2_reconstruct as reconstruct,
)
from protocol.algorithm1_permutation import algorithm_1_oblivious_candidate_permutation
from protocol.algorithm2_validation import algorithm_2_input_validation
from protocol.algorithm3_initialization import algorithm_3_initialization
from protocol.algorithm4_score import algorithm_4_score_computation
from protocol.algorithm5_find_min import algorithm_5_find_minimum_score
from protocol.algorithm6_one_hot import algorithm_6_one_hot_winner
from protocol.algorithm7_wallet_update import algorithm_7_wallet_update
from protocol.algorithm8_reconstruct_winner import algorithm_8_reconstruct_winner
from protocol.state_manager import (
    combine_tallier_states,
    load_tallier_state,
    save_tallier_state,
)
from protocol.types import BallotMatrix, Shares

# ---------------------------------------------------------------------------
# Fixed election parameters
# ---------------------------------------------------------------------------

P = 251   # prime field; large enough that cross-products stay below p
N = 3     # number of tallying parties
T = 2     # reconstruction threshold
M = 2     # number of candidates
SECURITY_BITS = 128

# Fixed seeds for Algorithm 1 — deterministic permutation in tests.
_ALG1_SEEDS = [bytes([d + 1]) * (SECURITY_BITS // 8) for d in range(N)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _share(v: int) -> Shares:
    return share(v, N, T, P)


def _rec(s: Shares) -> int:
    return reconstruct(s[:T], P)


def _ballot_matrix(values: list[list[int]]) -> BallotMatrix:
    return [[_share(v) for v in row] for row in values]


def _assert_invariant(
    wallets: list[Shares],
    delta: Shares,
    expected: list[tuple[int, int]],  # (W̃_i, Δ) as plain ints
) -> None:
    """Verify w_n = W̃_n / Δ for each voter via cross-multiplication (mod p)."""
    delta_val = _rec(delta)
    for i, (exp_w, exp_d) in enumerate(expected):
        w_tilde = _rec(wallets[i])
        assert (w_tilde * exp_d) % P == (exp_w * delta_val) % P, (
            f"Invariant violated voter {i}: W̃={w_tilde}, Δ={delta_val}, "
            f"expected W̃/Δ = {exp_w}/{exp_d}"
        )


# ---------------------------------------------------------------------------
# Main integration test
# ---------------------------------------------------------------------------

class TestTwoPeriodElection:

    def setup_method(self):
        random.seed(0)

    def test_full_two_period_run(self, tmp_path):
        """Complete 2-period election: winner changes when V0 changes ballot."""

        # ── One-time setup (Algorithms 1–3) ──────────────────────────────

        pi = algorithm_1_oblivious_candidate_permutation(
            SECURITY_BITS, M, N, seeds=_ALG1_SEEDS
        )
        assert sorted(pi) == list(range(M))

        # Period 1 ballots: V0→C0, V1→C0, V2→C1
        B_raw_p1 = _ballot_matrix([[1, 0], [1, 0], [0, 1]])
        B_hat_p1, n_valid = algorithm_2_input_validation(B_raw_p1, N, T, P)
        assert n_valid == 3   # no cheaters

        delta_0, wallets_0 = algorithm_3_initialization(n_valid, N, T, P)
        assert _rec(delta_0) == 1
        assert all(_rec(w) == 0 for w in wallets_0)

        # ── Period 1 ─────────────────────────────────────────────────────

        scores_1, A_shr_1, T_shr_1 = algorithm_4_score_computation(
            B_hat_p1, wallets_0, delta_0, n_valid, M, N, T, P
        )

        # Period 1 first-period invariant: all T̃_m = 0, Score_m^num = 1
        assert _rec(T_shr_1[0]) == 0
        assert _rec(T_shr_1[1]) == 0
        # |A(C0)| = 2, |A(C1)| = 1
        assert _rec(A_shr_1[0]) == 2
        assert _rec(A_shr_1[1]) == 1

        winning_score_1 = algorithm_5_find_minimum_score(scores_1, N, T, P)
        chi_1 = algorithm_6_one_hot_winner(
            scores_1, winning_score_1, pi, N, T, P
        )

        winner_1 = algorithm_8_reconstruct_winner(chi_1, T, P)
        assert winner_1 == 0, (
            "C0 should win period 1: it has 2 supporters vs C1's 1"
        )

        wallets_1, delta_1 = algorithm_7_wallet_update(
            B_hat_p1, chi_1, wallets_0, delta_0,
            A_shr_1, T_shr_1, n_valid, M, N, T, P
        )

        # Period 1 post-update checks:
        #   New Δ = 1 × |A(C0)| = 2
        #   V0, V1 (supporters): W̃ = 0    w_n = 0/2 = 0
        #   V2 (non-supporter):  W̃ = 1    w_n = 1/2
        assert _rec(delta_1) == 2
        assert _rec(wallets_1[0]) == 0   # V0 supporter → reset
        assert _rec(wallets_1[1]) == 0   # V1 supporter → reset
        assert _rec(wallets_1[2]) == 1   # V2 non-supporter → gained Score* = 1/2

        _assert_invariant(
            wallets_1, delta_1,
            [(0, 2), (0, 2), (1, 2)],    # (W̃_i, Δ) encoding w_n
        )

        # ── Save tallier state (end of period 1) ─────────────────────────

        state_paths = [
            str(tmp_path / f"tallier_{d}.json") for d in range(1, N + 1)
        ]
        for d in range(1, N + 1):
            save_tallier_state(d, delta_1, wallets_1, state_paths[d - 1])

        # Verify files were created
        for path in state_paths:
            assert os.path.exists(path)

        # ── Load tallier state (start of period 2) ────────────────────────

        loaded = [
            load_tallier_state(d, state_paths[d - 1]) for d in range(1, N + 1)
        ]
        delta_loaded, wallets_loaded = combine_tallier_states(loaded)

        # Reconstructed state must equal the original after period 1
        assert _rec(delta_loaded) == _rec(delta_1)
        for i in range(n_valid):
            assert _rec(wallets_loaded[i]) == _rec(wallets_1[i])

        # ── Period 2 — V0 changes ballot from {C0} to {C1} ───────────────

        # Fresh ballots: V0 now supports C1 instead of C0
        B_raw_p2 = _ballot_matrix([[0, 1], [1, 0], [0, 1]])
        B_hat_p2, n_valid_2 = algorithm_2_input_validation(B_raw_p2, N, T, P)
        assert n_valid_2 == 3   # still no cheaters

        scores_2, A_shr_2, T_shr_2 = algorithm_4_score_computation(
            B_hat_p2, wallets_loaded, delta_loaded, n_valid_2, M, N, T, P
        )

        # Period 2 score expectations:
        #   |A(C0)| = 1 (V1 only),  T̃_C0 = W̃_V1 = 0  → Score_C0 = 2/2 = 1
        #   |A(C1)| = 2 (V0, V2),   T̃_C1 = W̃_V0+W̃_V2 = 0+1 = 1
        #                                              → Score_C1 = 1/4
        assert _rec(A_shr_2[0]) == 1
        assert _rec(A_shr_2[1]) == 2
        assert _rec(T_shr_2[0]) == 0   # T̃_C0 = 0 (V1's wallet is 0)
        assert _rec(T_shr_2[1]) == 1   # T̃_C1 = 1 (V2's carried balance)

        winning_score_2 = algorithm_5_find_minimum_score(scores_2, N, T, P)
        chi_2 = algorithm_6_one_hot_winner(
            scores_2, winning_score_2, pi, N, T, P
        )

        winner_2 = algorithm_8_reconstruct_winner(chi_2, T, P)
        assert winner_2 == 1, (
            "C1 should win period 2: V2's accumulated balance lowers C1's score"
        )

        wallets_2, delta_2 = algorithm_7_wallet_update(
            B_hat_p2, chi_2, wallets_loaded, delta_loaded,
            A_shr_2, T_shr_2, n_valid_2, M, N, T, P
        )

        # Period 2 post-update checks:
        #   New Δ = 2 × |A(C1)| = 4
        #   V0 (supporter C1): reset.   w_V0 = 0/4
        #   V1 (non-supporter):         W̃_V1 = 0×2 + κ*₂ = 0+1 = 1.  w_V1 = 1/4
        #   V2 (supporter C1): reset.   w_V2 = 0/4
        assert _rec(delta_2) == 4
        assert _rec(wallets_2[0]) == 0   # V0 supporter → reset
        assert _rec(wallets_2[1]) == 1   # V1 non-supporter → gained 1/4
        assert _rec(wallets_2[2]) == 0   # V2 supporter → reset

        _assert_invariant(
            wallets_2, delta_2,
            [(0, 4), (1, 4), (0, 4)],
        )

        # ── Cross-period assertions ───────────────────────────────────────

        # Primary fairness check: different candidates win in the two periods
        assert winner_1 != winner_2, (
            "V0's ballot change must produce a different winner in period 2"
        )

        # Δ growth: Δ_2 = |A(m*_1)| × |A(m*_2)| = 2 × 2 = 4
        assert _rec(delta_2) == _rec(delta_1) * _rec(A_shr_2[winner_2]) % P

    def test_ballot_change_is_independent_of_wallet_state(self, tmp_path):
        """Ballot matrix is fully replaced each period — there is no memory
        of prior ballot entries in B_hat.

        This test verifies that when Algorithm 2 runs on the period-2 ballots,
        V0's old vote for C0 is NOT present.  The period-2 B_hat uses only the
        fresh period-2 submissions.
        """
        random.seed(1)

        # Run period 1 lightly (just to get state)
        B_p1 = _ballot_matrix([[1, 0], [1, 0], [0, 1]])
        B_hat_p1, n_valid = algorithm_2_input_validation(B_p1, N, T, P)
        delta_0, wallets_0 = algorithm_3_initialization(n_valid, N, T, P)
        scores_1, A_1, T_1 = algorithm_4_score_computation(
            B_hat_p1, wallets_0, delta_0, n_valid, M, N, T, P
        )
        pi = [0, 1]
        ws_1 = algorithm_5_find_minimum_score(scores_1, N, T, P)
        chi_1 = algorithm_6_one_hot_winner(scores_1, ws_1, pi, N, T, P)
        wallets_1, delta_1 = algorithm_7_wallet_update(
            B_hat_p1, chi_1, wallets_0, delta_0, A_1, T_1, n_valid, M, N, T, P
        )

        # Period 2: V0 submits entirely new ballot — old C0 support is gone
        B_p2 = _ballot_matrix([[0, 1], [1, 0], [0, 1]])
        B_hat_p2, _ = algorithm_2_input_validation(B_p2, N, T, P)

        # Verify period-2 ballot matrix reflects ONLY the new ballots
        scores_2, A_2, _ = algorithm_4_score_computation(
            B_hat_p2, wallets_1, delta_1, n_valid, M, N, T, P
        )

        # With new ballots: |A(C0)| = 1 (V1 only), NOT 2 as in period 1
        assert _rec(A_2[0]) == 1, (
            "V0's old C0 vote must not appear in period-2 ballot matrix"
        )
        assert _rec(A_2[1]) == 2, (
            "V0 now supports C1; |A(C1)| must be 2"
        )

    def test_state_roundtrip_preserves_all_shares(self, tmp_path):
        """save → load → combine must reconstruct all shares exactly."""
        random.seed(2)

        delta_0, wallets_0 = algorithm_3_initialization(3, N, T, P)

        # Corrupt the check: if reconstruct(combined) ≠ reconstruct(original)
        # then the state round-trip lost information.
        paths = [str(tmp_path / f"t{d}.json") for d in range(1, N + 1)]
        for d in range(1, N + 1):
            save_tallier_state(d, delta_0, wallets_0, paths[d - 1])

        loaded = [load_tallier_state(d, paths[d - 1]) for d in range(1, N + 1)]
        delta_rt, wallets_rt = combine_tallier_states(loaded)

        assert _rec(delta_rt) == _rec(delta_0)
        for i in range(3):
            assert _rec(wallets_rt[i]) == _rec(wallets_0[i])

    def test_load_wrong_tallier_id_raises(self, tmp_path):
        """load_tallier_state raises ValueError when file tallier_id mismatches."""
        delta_0, wallets_0 = algorithm_3_initialization(2, N, T, P)
        path = str(tmp_path / "tallier_1.json")
        save_tallier_state(1, delta_0, wallets_0, path)

        with pytest.raises(ValueError, match="tallier"):
            load_tallier_state(2, path)   # file stores tallier 1, not 2
