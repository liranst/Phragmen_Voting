"""Algorithm 6 — One-Hot Winner Identification with Permuted Tie-Break.

Reference: Algorithm 6 in "Fairness Without Exposure: Privacy-Preserving
Phragmén Voting", Section 5.6 (pages 12–13).  Run every election period.

Produces a one-hot shared indicator {[[χ_m]]}_{m ∈ [M]} with χ_{m*} = 1 for
the unique elected candidate m* and χ_m = 0 for all others.

Two sequential passes:

Pass 1 (Lines 1–4, O(1) rounds — all M triples are independent and batched):
  For each candidate m, test whether Score_m equals the winning score Score*
  via cross-multiplication followed by SecureIsZero:
      α_m = Score_m^num · Score*^den
      β_m = Score*^num · Score_m^den
      e_m = SecureIsZero(α_m − β_m)       ← 1 iff Score_m = Score*

Pass 2 (Lines 5–11, O(M) rounds — sequential prefix scan):
  Walk the candidates in the public permutation order π.  The running prefix
  accumulator [[s]] starts at 0; the first π-ordered candidate m with e_m = 1
  gets χ_m = 1 and sets s = 1, suppressing all subsequent matches via
      χ_m = SecureMult(e_m, [[1]] − [[s]])

The permutation π (from Algorithm 1) breaks ties uniformly at random while
keeping the winner's identity secret until Algorithm 8.

Round complexity: O(M) dominated by the prefix scan (paper Remark 6.1).
Communication: 2M + (M−1) SecureMult and M SecureIsZero per period.
"""

from typing import List

from mpc_secret_shares import (
    secure_mult,
    is_zero,
    shares_add,
    shares_one,
    shares_sub,
)
from protocol.types import ScorePair, Shares


def algorithm_6_one_hot_winner(
    scores: List[ScorePair],
    winning_score: ScorePair,
    pi: List[int],
    n: int,
    t: int,
    p: int,
) -> List[Shares]:
    """Compute a one-hot shared indicator for the winning candidate.

    Algorithm 6 of "Fairness Without Exposure: Privacy-Preserving Phragmén
    Voting", Section 5.6.

    Args:
        scores: List of M ScorePairs ``(Score_m^num, Score_m^den)`` from
            Algorithm 4.  ``scores[m]`` corresponds to candidate m (0-indexed).
        winning_score: The minimum score ``(Score*^num, Score*^den)`` from
            Algorithm 5.
        pi: Public permutation of ``[0, M)`` from Algorithm 1 (0-indexed).
            Used to break ties obliviously; ``pi[0]`` is the highest-priority
            candidate in case of a tie.
        n: Number of MPC parties.
        t: Reconstruction threshold.
        p: Prime field modulus.  Cross-products Score_m^num · Score*^den must
            not exceed p (paper Section 7).

    Returns:
        ``chi`` — a list of M (t,n)-sharings where ``chi[m] = [[χ_m]]``.
        Exactly one entry encodes the secret 1; all others encode 0.

    Raises:
        ValueError: If ``scores`` is empty or ``pi`` has wrong length.
    """
    m_count = len(scores)
    if m_count == 0:
        raise ValueError("scores must be non-empty: at least one candidate required")
    if len(pi) != m_count:
        raise ValueError(
            f"pi has length {len(pi)}, expected {m_count} (one entry per candidate)"
        )

    star_num: Shares = winning_score[0]
    star_den: Shares = winning_score[1]
    one: Shares = shares_one(n)

    # ------------------------------------------------------------------
    # Lines 1-4: compute e_m = SecureIsZero(α_m − β_m) for each candidate.
    # All M triples (SecureMult, SecureMult, SecureIsZero) are mutually
    # independent → they can be batched in O(1) rounds.
    # ------------------------------------------------------------------
    e: List[Shares] = []
    for m in range(m_count):                                         # Line 1
        score_num, score_den = scores[m]

        # Line 2: α_m = SecureMult([[Score_m^num]], [[Score*^den]])
        alpha_m: Shares = secure_mult(
            score_num, star_den, n, t, p
        )

        # Line 3: β_m = SecureMult([[Score*^num]], [[Score_m^den]])
        beta_m: Shares = secure_mult(
            star_num, score_den, n, t, p
        )

        # Line 4: e_m = SecureIsZero([[α_m]] − [[β_m]])
        # α_m − β_m = 0  iff  Score_m = Score*  (same fraction)
        diff: Shares = shares_sub(alpha_m, beta_m, p)
        e.append(is_zero(diff, n, t, p))

    # ------------------------------------------------------------------
    # Lines 5-11: prefix scan in permutation order π.
    # chi is pre-allocated and filled by candidate index (not π-order).
    # ------------------------------------------------------------------
    chi: List[Shares] = [None] * m_count  # type: ignore[list-item]

    # Lines 5-7: initialise with the first candidate in π-order.
    m1: int = pi[0]                                                  # Line 5
    chi[m1] = e[m1]                                                  # Line 6
    s: Shares = e[m1]                                                # Line 7

    # Lines 8-11: suppress all but the first π-ordered tie-winner.
    for k in range(1, m_count):                                      # Line 8
        m = pi[k]                                                    # Line 9

        # Line 10: [[χ_m]] = SecureMult([[e_m]], [[1]] − [[s]])
        # [[1]] − [[s]] is local (affine subtraction of public 1).
        one_minus_s: Shares = shares_sub(one, s, p)
        chi[m] = secure_mult(e[m], one_minus_s, n, t, p)  # Line 10

        # Line 11: [[s]] ← [[s]] + [[χ_m]]  (local)
        s = shares_add(s, chi[m], p)                                 # Line 11

    return chi
