"""Algorithm 3 — Initialization.

Reference: Algorithm 3 in "Fairness Without Exposure: Privacy-Preserving
Phragmén Voting", Section 5.3 (page 9).  Run once, prior to the first
election period only.

Sets the global denominator Δ to 1 and every voter's scaled wallet W̃_i to 0,
establishing the invariant w_n = W̃_n / Δ = 0 for all voters at the start.
No MPC secure operations are required — both values are public constants.
"""

from typing import List, Tuple

from mpc_primitives.mpc_project.mpc_secret_shares import (
    shares_one,
    shares_zero,
)
from protocol.types import Shares


def algorithm_3_initialization(
    n_valid: int,
    n: int,
    t: int,
    p: int,
) -> Tuple[Shares, List[Shares]]:
    """Initialise the global denominator and all voter wallets to their
    starting values.

    Algorithm 3 of "Fairness Without Exposure: Privacy-Preserving Phragmén
    Voting", Section 5.3.

    At the start of the first period all rational wallet balances w_n are 0.
    The protocol represents each balance as w_n = W̃_n / Δ, so initialising
    Δ = 1 and W̃_n = 0 satisfies the invariant without any field division.

    Args:
        n_valid: N_valid — number of honest voters after Algorithm 2.
        n: Number of MPC parties (talliers).
        t: Reconstruction threshold.
        p: Prime field modulus.

    Returns:
        A tuple ``(delta_shares, wallet_shares)`` where:

        * ``delta_shares`` is a (t,n)-sharing of Δ = 1  (Algorithm 3, Line 1).
        * ``wallet_shares`` is a list of N_valid (t,n)-sharings, each
          encoding W̃_i = 0  (Algorithm 3, Lines 2–3).
    """
    # Line 1: [[Δ]] ← [[1]]
    delta_shares: Shares = shares_one(n)

    # Lines 2-3: [[W̃_i]] ← [[0]] for every honest voter i.
    wallet_shares: List[Shares] = [shares_zero(n) for _ in range(n_valid)]

    return delta_shares, wallet_shares
