"""Algorithm 7 — Wallet Update (Global Denominator Scaling).

Reference: Algorithm 7 in "Fairness Without Exposure: Privacy-Preserving
Phragmén Voting", Section 5.7 (pages 13–14).  Run every election period.

Updates every voter's scaled wallet W̃_i and the global denominator Δ after
the winner m* is chosen, maintaining the central invariant (paper Eq. 6):

    w_n = W̃_n / Δ    for all n ∈ [N_valid], all periods.

The update rule (paper Eq. 2) for a voter V_n is:

    w_n ← 0                                  if n ∈ A(m*)   (supporter: reset)
    w_n ← w_n + Score*                        if n ∉ A(m*)   (non-supporter: gain)

where Score* = (Δ − T̃*) / (|A(m*)| · Δ).

To avoid any field division the algorithm scales all wallet values by |A(m*)|
and increments the global denominator by the same factor:

    New W̃_i (supporter)     = 0
    New W̃_i (non-supporter) = W̃_i · |A(m*)| + κ*        where κ* = Δ − T̃*
    New Δ                   = Δ · |A(m*)|

Invariant verification:
    New w_n = New W̃_n / New Δ
    For supporters:     0 / (Δ · |A(m*)|)   = 0                         ✓
    For non-supporters: (W̃_n · |A(m*)| + κ*) / (Δ · |A(m*)|)
                      = W̃_n/Δ + κ*/(Δ · |A(m*)|)
                      = w_n + Score*                                     ✓
"""

from typing import List, Tuple

from mpc_secret_shares import (
    protocol_4_secure_mult as secure_mult,
    shares_add,
    shares_one,
    shares_sub,
    shares_zero,
)
from protocol.types import BallotMatrix, Shares


def algorithm_7_wallet_update(
    B_hat: BallotMatrix,
    chi: List[Shares],
    wallet_shares: List[Shares],
    delta_shares: Shares,
    A_shares: List[Shares],
    T_tilde_shares: List[Shares],
    n_valid: int,
    num_candidates: int,
    n: int,
    t: int,
    p: int,
) -> Tuple[List[Shares], Shares]:
    """Update voter wallets and the global denominator after electing a winner.

    Algorithm 7 of "Fairness Without Exposure: Privacy-Preserving Phragmén
    Voting", Section 5.7.

    Precondition: at least one voter supports the elected candidate
    (paper Section 5, page 8 — guaranteed by the protocol assumptions).

    Args:
        B_hat: N_valid × M ballot matrix of (t,n)-sharings.
        chi: List of M (t,n)-sharings — the one-hot winner indicator from
            Algorithm 6.  Exactly one entry encodes 1.
        wallet_shares: List of N_valid (t,n)-sharings of the current W̃_i.
        delta_shares: (t,n)-sharing of the current global denominator Δ.
        A_shares: List of M (t,n)-sharings of |A(m)| from Algorithm 4.
        T_tilde_shares: List of M (t,n)-sharings of T̃_m from Algorithm 4.
        n_valid: Number of honest voters N_valid.
        num_candidates: Number of candidates M.
        n: Number of MPC parties.
        t: Reconstruction threshold.
        p: Prime field modulus.

    Returns:
        A tuple ``(new_wallet_shares, new_delta_shares)`` where:

        * ``new_wallet_shares[i]`` is the updated (t,n)-sharing of W̃_i.
        * ``new_delta_shares`` is the updated (t,n)-sharing of Δ.

        The invariant w_n = W̃_n / Δ is preserved for every honest voter.
    """
    one: Shares = shares_one(n)

    # ------------------------------------------------------------------
    # Lines 1-5: extract the winner's supporter count and load sum.
    # The χ-weighted sums isolate the winner's row from Algorithm 4 output.
    # ------------------------------------------------------------------
    A_star: Shares = shares_zero(n)   # [[|A(m*)|]]            Line 1
    T_star: Shares = shares_zero(n)   # [[T̃*]]                 Line 2

    for m in range(num_candidates):                            # Line 3
        # Line 4: [[|A(m*)|]] ← [[|A(m*)|]] + SecureMult([[χ_m]], [[|A(m)|]])
        A_star = shares_add(
            A_star,
            secure_mult(chi[m], A_shares[m], n, t, p),
            p,
        )
        # Line 5: [[T̃*]] ← [[T̃*]] + SecureMult([[χ_m]], [[T̃_m]])
        T_star = shares_add(
            T_star,
            secure_mult(chi[m], T_tilde_shares[m], n, t, p),
            p,
        )

    # Line 6: [[κ*]] ← [[Δ]] − [[T̃*]]   (= Score*^num, local)
    kappa_star: Shares = shares_sub(delta_shares, T_star, p)

    # ------------------------------------------------------------------
    # Lines 7-13: per-voter wallet update.
    # γ_i = 1 iff voter i supports the winner; 0 otherwise.
    # Supporters are reset (× 0); non-supporters are incremented (× 1).
    # ------------------------------------------------------------------
    new_wallets: List[Shares] = []

    for i in range(n_valid):                                   # Line 7
        # Lines 8-10: [[γ_i]] = Σ_m SecureMult([[B̂_{i,m}]], [[χ_m]])
        gamma_i: Shares = shares_zero(n)                      # Line 8
        for m in range(num_candidates):                        # Line 9
            gamma_i = shares_add(
                gamma_i,
                secure_mult(B_hat[i][m], chi[m], n, t, p),
                p,
            )                                                  # Line 10

        # Line 11: [[μ_i]] ← SecureMult([[W̃_i]], [[|A(m*)|]])
        mu_i: Shares = secure_mult(
            wallet_shares[i], A_star, n, t, p
        )

        # Line 12: [[ν_i]] ← [[μ_i]] + [[κ*]]  (local)
        nu_i: Shares = shares_add(mu_i, kappa_star, p)

        # Line 13: [[W̃_i]] ← SecureMult([[1]] − [[γ_i]], [[ν_i]])
        # [[1]] − [[γ_i]] is local.  Supporters (γ_i=1) → 0; others → ν_i.
        one_minus_gamma: Shares = shares_sub(one, gamma_i, p)
        new_wallets.append(
            secure_mult(one_minus_gamma, nu_i, n, t, p)
        )                                                      # Line 13

    # Line 14: [[Δ]] ← SecureMult([[Δ]], [[|A(m*)|]])
    new_delta: Shares = secure_mult(delta_shares, A_star, n, t, p)

    return new_wallets, new_delta                              # Line 14
