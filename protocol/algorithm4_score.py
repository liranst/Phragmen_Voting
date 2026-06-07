"""Algorithm 4 — Oblivious Score Computation (Global Denominator).

Reference: Algorithm 4 in "Fairness Without Exposure: Privacy-Preserving
Phragmén Voting", Section 5.4 (pages 10–11).  Run every election period.

Computes per-candidate scores in numerator/denominator form using the
common-denominator strategy (Section 5, page 7):

    Score_m = Score_m^num / Score_m^den
            = (Δ − T̃_m) / (|A(m)| · Δ)

where T̃_m = Σ_{i ∈ A(m)} W̃_i is the sum of scaled wallets of candidate
m's supporters, and Δ is the shared global denominator.  This eliminates
per-voter cross-multiplication from the critical path (O(MN) SecureMult
instead of O(MN²)).

Precondition (paper Section 5, page 8): at least one candidate must have
|A(m)| > 0 (i.e. at least one voter approves at least one candidate).
The degenerate all-zero ballot matrix is explicitly excluded by the paper;
calling this function with n_valid=0 or all-zero ballots is outside the
paper's guaranteed domain and will produce Score^den = 0 for every candidate,
making Algorithm 5's comparison meaningless.
"""

from typing import List, Tuple

from mpc_secret_shares import (
    secure_mult,
    shares_add,
    shares_sub,
    shares_zero,
)
from protocol.types import BallotMatrix, ScorePair, Shares


def algorithm_4_score_computation(
    B_hat: BallotMatrix,
    wallet_shares: List[Shares],
    delta_shares: Shares,
    n_valid: int,
    num_candidates: int,
    n: int,
    t: int,
    p: int,
) -> Tuple[List[ScorePair], List[Shares], List[Shares]]:
    """Compute per-candidate scores and supporting data for the current period.

    Algorithm 4 of "Fairness Without Exposure: Privacy-Preserving Phragmén
    Voting", Section 5.4.

    For each candidate m the algorithm accumulates:

    * ``[[|A(m)|]]`` — supporter count (local additions of ballot bits).
    * ``[[T̃_m]]`` — scaled wallet sum of supporters (SecureMult per voter).

    Then it derives the score representation from Equation (4) of the paper:

    * ``[[Score_m^num]] = [[Δ]] − [[T̃_m]]``       (local subtraction)
    * ``[[Score_m^den]] = SecureMult([[Δ]], [[|A(m)|]])``

    Args:
        B_hat: N_valid × M matrix of (t,n)-sharings of B̂_{i,m} ∈ {0,1}.
        wallet_shares: List of N_valid (t,n)-sharings of the scaled wallets
            W̃_i.  Index matches the row index of B_hat.
        delta_shares: (t,n)-sharing of the global denominator Δ.
        n_valid: Number of valid voters N_valid.  Must equal len(B_hat).
        num_candidates: Number of candidates M.
        n: Number of MPC parties.
        t: Reconstruction threshold.
        p: Prime field modulus.

    Returns:
        A tuple ``(scores, A_shares, T_tilde_shares)`` where:

        * ``scores[m]`` is ``(Score_m^num, Score_m^den)`` — a ScorePair of
          (t,n)-sharings for candidate m (Algorithm 4, Lines 8–9).
        * ``A_shares[m]`` is ``[[|A(m)|]]`` — supporter count for candidate m
          (Line 5), needed by Algorithm 7.
        * ``T_tilde_shares[m]`` is ``[[T̃_m]]`` — scaled wallet sum for
          candidate m (Line 7), needed by Algorithm 7.
    """
    scores: List[ScorePair] = []
    A_shares: List[Shares] = []
    T_tilde_shares: List[Shares] = []

    for m in range(num_candidates):                               # Line 1
        A_m: Shares = shares_zero(n)                             # Line 2
        T_m: Shares = shares_zero(n)                             # Line 3

        for i in range(n_valid):                                  # Line 4
            b_im: Shares = B_hat[i][m]

            # Line 5: [[|A(m)|]] ← [[|A(m)|]] + [[B̂_{i,m}]]  (local)
            A_m = shares_add(A_m, b_im, p)

            # Line 6: [[contrib]] ← SecureMult([[B̂_{i,m}]], [[W̃_i]])
            contrib: Shares = secure_mult(
                b_im, wallet_shares[i], n, t, p
            )

            # Line 7: [[T̃_m]] ← [[T̃_m]] + [[contrib]]  (local)
            T_m = shares_add(T_m, contrib, p)

        # Line 8: [[Score_m^num]] ← [[Δ]] − [[T̃_m]]  (local)
        score_num: Shares = shares_sub(delta_shares, T_m, p)

        # Line 9: [[Score_m^den]] ← SecureMult([[Δ]], [[|A(m)|]])
        score_den: Shares = secure_mult(
            delta_shares, A_m, n, t, p
        )

        scores.append((score_num, score_den))
        A_shares.append(A_m)
        T_tilde_shares.append(T_m)

    return scores, A_shares, T_tilde_shares
