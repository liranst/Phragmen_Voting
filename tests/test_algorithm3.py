"""Tests for Algorithm 3 — Initialization.

Reference: Algorithm 3 in "Fairness Without Exposure: Privacy-Preserving
Phragmén Voting", Section 5.3.

Standard test parameters: p=11, n=3 parties, t=2 threshold.
"""

import pytest

from mpc_secret_shares import reconstruct
from protocol.algorithm3_initialization import algorithm_3_initialization

P, N, T = 11, 3, 2


class TestDeltaInitialization:
    def test_delta_reconstructs_to_one(self):
        """[[Δ]] must encode the secret 1."""
        delta, _ = algorithm_3_initialization(3, N, T, P)
        assert reconstruct(delta[:T], P) == 1

    def test_delta_has_n_shares(self):
        """[[Δ]] must contain exactly n share entries."""
        delta, _ = algorithm_3_initialization(3, N, T, P)
        assert len(delta) == N

    def test_delta_share_values_in_field(self):
        """Every share value of [[Δ]] must be in Z_p."""
        delta, _ = algorithm_3_initialization(3, N, T, P)
        assert all(0 <= v < P for _, v in delta)

    def test_delta_party_ids_are_one_indexed(self):
        """Share party IDs must be 1, 2, …, n."""
        delta, _ = algorithm_3_initialization(3, N, T, P)
        assert [pid for pid, _ in delta] == list(range(1, N + 1))


class TestWalletInitialization:
    def test_all_wallets_zero(self):
        """Every W̃_i must encode the secret 0."""
        _, wallets = algorithm_3_initialization(5, N, T, P)
        for w in wallets:
            assert reconstruct(w[:T], P) == 0

    def test_wallet_count_equals_n_valid(self):
        """Number of wallet sharings must equal n_valid."""
        for n_valid in (0, 1, 3, 10):
            _, wallets = algorithm_3_initialization(n_valid, N, T, P)
            assert len(wallets) == n_valid

    def test_each_wallet_has_n_shares(self):
        """Each wallet sharing must contain exactly n share entries."""
        _, wallets = algorithm_3_initialization(4, N, T, P)
        for w in wallets:
            assert len(w) == N

    def test_wallet_share_values_in_field(self):
        """Every share value of every W̃_i must be in Z_p."""
        _, wallets = algorithm_3_initialization(3, N, T, P)
        for w in wallets:
            assert all(0 <= v < P for _, v in w)

    def test_n_valid_zero_empty_wallets(self):
        """n_valid=0 → empty wallet list; Δ still initialised."""
        delta, wallets = algorithm_3_initialization(0, N, T, P)
        assert wallets == []
        assert reconstruct(delta[:T], P) == 1

    def test_n_valid_one(self):
        """Single voter → exactly one wallet sharing of 0."""
        _, wallets = algorithm_3_initialization(1, N, T, P)
        assert len(wallets) == 1
        assert reconstruct(wallets[0][:T], P) == 0


class TestInvariant:
    def test_wallet_invariant_holds(self):
        """w_n = W̃_n / Δ = 0 / 1 = 0 for all voters at initialisation."""
        delta, wallets = algorithm_3_initialization(4, N, T, P)
        delta_val = reconstruct(delta[:T], P)
        for w in wallets:
            w_tilde = reconstruct(w[:T], P)
            # Rational balance: w_n = W̃_n / Δ = 0 / 1 = 0
            assert w_tilde * pow(delta_val, P - 2, P) % P == 0

    def test_wallets_are_independent_sharings(self):
        """Each wallet sharing must be a distinct list object."""
        _, wallets = algorithm_3_initialization(3, N, T, P)
        assert wallets[0] is not wallets[1]
        assert wallets[1] is not wallets[2]
