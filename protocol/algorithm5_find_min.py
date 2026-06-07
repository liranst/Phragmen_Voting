"""Algorithm 5 — Find Minimum Score.

Reference: Algorithm 5 in "Fairness Without Exposure: Privacy-Preserving
Phragmén Voting", Section 5.5 (page 12).  Run every election period.

Finds the winning (minimum) score among all M candidates using a sequential
left-to-right reduction.  At each step the running best is compared against
the next candidate via cross-multiplication to avoid division:

    Score_m < Score_best  iff  Score_m^num · best^den < best^num · Score_m^den

Both cross-products are non-negative (Lemma 7.1), so SecureLT is valid.

Round complexity: O(M) — this sequential chain is the dominant latency
bottleneck of the per-period loop (paper Remark 6.1).
"""

from typing import List, Tuple

from mpc_secret_shares import (
    protocol_4_secure_mult as secure_mult,
    protocol_5_secure_compare as secure_lt,
    shares_add,
    shares_sub,
)
from protocol.types import ScorePair, Shares


def algorithm_5_find_minimum_score(
    scores: List[ScorePair],
    n: int,
    t: int,
    p: int,
) -> ScorePair:
    """Find the shared winning (minimum) score across all candidates.

    Algorithm 5 of "Fairness Without Exposure: Privacy-Preserving Phragmén
    Voting", Section 5.5.

    Maintains a running best ``(best^num, best^den)`` initialised to the
    first candidate's score (Lines 1–2).  For each subsequent candidate m
    (Lines 3–8) the algorithm:

    1. Computes two cross-products to compare scores without division
       (Lines 4–5).
    2. Calls SecureLT to get a shared bit β_m = 1_{Score_m < Score_best}
       (Line 6).
    3. Updates the running best via a multiplexer:
       ``best ← best + β_m · (Score_m − best)``
       which selects Score_m when β_m = 1 and leaves best unchanged when 0
       (Lines 7–8).

    Args:
        scores: List of M ScorePairs ``(Score_m^num, Score_m^den)`` produced
            by Algorithm 4.  Must contain at least one entry.
        n: Number of MPC parties.
        t: Reconstruction threshold.
        p: Prime field modulus.

    Returns:
        A ScorePair ``(Score*^num, Score*^den)`` — a (t,n)-sharing of the
        minimum score among all candidates.

    Raises:
        ValueError: If ``scores`` is empty (no candidates).
    """
    if not scores:
        raise ValueError("scores must be non-empty: at least one candidate required")

    # Lines 1-2: initialise the running best to the first candidate's score.
    best_num: Shares = scores[0][0]
    best_den: Shares = scores[0][1]

    # Lines 3-8: compare each subsequent candidate against the running best.
    for m in range(1, len(scores)):                                  # Line 3
        score_num: Shares = scores[m][0]
        score_den: Shares = scores[m][1]

        # Line 4: cross_a = Score_m^num · best^den
        cross_a: Shares = secure_mult(
            score_num, best_den, n, t, p
        )

        # Line 5: cross_b = best^num · Score_m^den
        cross_b: Shares = secure_mult(
            best_num, score_den, n, t, p
        )

        # Line 6: β_m = SecureLT(cross_a, cross_b)
        # β_m = 1  iff  Score_m < Score_best  (candidate m is a better winner)
        beta_m: Shares = secure_lt(cross_a, cross_b, n, t, p)

        # Line 7: best^num ← best^num + β_m · (Score_m^num − best^num)
        diff_num: Shares = shares_sub(score_num, best_num, p)
        best_num = shares_add(
            best_num,
            secure_mult(beta_m, diff_num, n, t, p),
            p,
        )

        # Line 8: best^den ← best^den + β_m · (Score_m^den − best^den)
        diff_den: Shares = shares_sub(score_den, best_den, p)
        best_den = shares_add(
            best_den,
            secure_mult(beta_m, diff_den, n, t, p),
            p,
        )

    return best_num, best_den                                        # Lines 9-10
