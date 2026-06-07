"""Empirical validation of Section 7 — Field Size Requirements.

Reference: "Fairness Without Exposure: Privacy-Preserving Phragmén Voting",
Section 7 (Field Size Requirements and Boundedness).

The paper proves that the global denominator Delta grows as
    Delta_r = prod_{k=1}^{r-1} |A(m*_k)|

and the cross-products fed to SecureLT in Algorithm 5 are bounded by
    N * Delta^2_r.

For the protocol to be correct through period R the field must satisfy
    p > N * Delta^2_r  for all r <= R,

giving the asymptotic requirement  log p = Omega(R * log N).

This test runs the full per-period loop (Algorithms 4-8) in a while loop,
using a cleartext rational oracle to detect the first wrong winner, and
reports when and why the MPC output diverges from ground truth.

Ballot setup that maximises Delta growth (Delta doubles every round):
    V0 -> {C0}        V1 -> {C1}        V2 -> {C0, C1}
    |A(C0)| = |A(C1)| = 2 in every round  ->  Delta_r = 2^{r-1}.

Run with:  pytest tests/test_field_size.py -v -s
"""

import math
import random
from fractions import Fraction
from typing import List, Optional, Tuple

from mpc_secret_shares import (
    share,
    reconstruct,
)
from protocol.algorithm2_validation import algorithm_2_input_validation
from protocol.algorithm3_initialization import algorithm_3_initialization
from protocol.algorithm4_score import algorithm_4_score_computation
from protocol.algorithm5_find_min import algorithm_5_find_minimum_score
from protocol.algorithm6_one_hot import algorithm_6_one_hot_winner
from protocol.algorithm7_wallet_update import algorithm_7_wallet_update
from protocol.algorithm8_reconstruct_winner import algorithm_8_reconstruct_winner
from protocol.types import BallotMatrix, Shares


# ---------------------------------------------------------------------------
# Cleartext oracle — exact rational arithmetic via Python Fraction
# ---------------------------------------------------------------------------

def _cleartext_winner(
    wallets_frac: List[Fraction],
    ballots: List[List[int]],
    pi: List[int],
) -> int:
    """Return the correct winner index using exact rational Phragmén scoring."""
    scores = []
    for m in range(len(ballots[0])):
        supporters = [i for i in range(len(ballots)) if ballots[i][m] == 1]
        if not supporters:
            scores.append(Fraction(10 ** 9))       # effectively +∞
        else:
            total = sum(wallets_frac[i] for i in supporters)
            scores.append((Fraction(1) - total) / len(supporters))
    min_score = min(scores)
    for m in pi:
        if scores[m] == min_score:
            return m
    return pi[0]


def _update_cleartext_wallets(
    wallets_frac: List[Fraction],
    ballots: List[List[int]],
    winner: int,
) -> None:
    """Apply the Phragmén wallet-update rule in exact rational arithmetic."""
    supporters = [i for i in range(len(ballots)) if ballots[i][winner] == 1]
    total = sum(wallets_frac[i] for i in supporters)
    score_star = (Fraction(1) - total) / len(supporters)
    for i in range(len(wallets_frac)):
        if ballots[i][winner] == 1:
            wallets_frac[i] = Fraction(0)
        else:
            wallets_frac[i] += score_star


# ---------------------------------------------------------------------------
# Stress runner
# ---------------------------------------------------------------------------

def _run_stress(
    p: int,
    n: int = 3,
    t: int = 2,
    max_rounds: int = 50,
) -> Tuple[int, Optional[int]]:
    """Run the per-period loop until a wrong winner or max_rounds.

    Returns:
        (rounds_ok, delta_at_failure) — rounds_ok is the count of correctly
        answered rounds; delta_at_failure is Δ when the first wrong answer
        occurs (None if max_rounds completed correctly).
    """
    # Fixed balanced ballots: V0→C0, V1→C1, V2→both.
    # Each round |A(m*)| = 2, so Δ_r = 2^{r-1}.
    ballots = [[1, 0], [0, 1], [1, 1]]
    m = 2
    pi = [0, 1]        # C0 has tie-break priority
    n_valid = 3

    random.seed(42)

    def _share(v: int) -> Shares:
        return share(v, n, t, p)

    def _rec(s: Shares) -> int:
        return reconstruct(s[:t], p)

    def _make_ballot_shares() -> BallotMatrix:
        return [[_share(v) for v in row] for row in ballots]

    delta_shares, wallet_shares = algorithm_3_initialization(n_valid, n, t, p)
    wallets_frac = [Fraction(0)] * n_valid

    rounds_ok = 0
    delta_at_failure: Optional[int] = None

    print(f"\n  p = {p:>12d}   (log2(p) = {math.log2(p):.1f})")
    print(f"  {'r':>4}  {'Delta':>12}  {'N*Delta^2':>16}  {'< p?':>5}  result")
    print(f"  {'-'*4}  {'-'*12}  {'-'*16}  {'-'*5}  {'-'*20}")

    for r in range(1, max_rounds + 1):
        winner_true = _cleartext_winner(wallets_frac, ballots, pi)
        delta_val = _rec(delta_shares)
        max_cross = n * delta_val * delta_val
        safe = max_cross < p

        try:
            B_hat, _ = algorithm_2_input_validation(_make_ballot_shares(), n, t, p)
            scores, A_shr, T_shr = algorithm_4_score_computation(
                B_hat, wallet_shares, delta_shares, n_valid, m, n, t, p
            )
            winning_score = algorithm_5_find_minimum_score(scores, n, t, p)
            chi = algorithm_6_one_hot_winner(scores, winning_score, pi, n, t, p)
            winner_mpc = algorithm_8_reconstruct_winner(chi, t, p)
            wallet_shares, delta_shares = algorithm_7_wallet_update(
                B_hat, chi, wallet_shares, delta_shares, A_shr, T_shr,
                n_valid, m, n, t, p,
            )
            correct = (winner_mpc == winner_true)
            result = "OK  correct" if correct else f"FAIL  WRONG  (mpc=C{winner_mpc}, true=C{winner_true})"
        except Exception as exc:
            winner_mpc = -1
            correct = False
            result = f"FAIL  EXCEPTION: {type(exc).__name__}: {exc}"

        print(
            f"  {r:>4}  {delta_val:>12}  {max_cross:>16}  "
            f"{'yes' if safe else 'NO ':>5}  {result}"
        )

        # Update cleartext wallets using the oracle winner (stays accurate
        # even if MPC is already producing wrong results).
        _update_cleartext_wallets(wallets_frac, ballots, winner_true)

        if not correct:
            delta_at_failure = delta_val
            break

        rounds_ok += 1

    return rounds_ok, delta_at_failure


# ---------------------------------------------------------------------------
# Test entry point
# ---------------------------------------------------------------------------

PRIMES = [251, 65537, 2 ** 31 - 1]
N_PARTIES = 3


def test_field_size_stress():
    """Empirical validation of Section 7.

    Detects the round at which field overflow causes a wrong winner and
    compares empirical results against the theoretical bound from the paper.

    Output is printed to stdout; run with -s to see it.
    """
    print("\n")
    print("=" * 70)
    print("  Field Size Stress Test -- Empirical Validation of Section 7")
    print("=" * 70)
    print(f"  Ballot setup:  V0->{{C0}}, V1->{{C1}}, V2->{{C0,C1}}")
    print(f"  |A(m*)| = 2 every round  ->  Delta_r = 2^(r-1)")
    print(f"  Overflow condition: N*Delta^2 >= p  (cross-products in SecureLT wrap)")
    print(f"  Cleartext oracle: exact Fraction arithmetic (ground truth)")

    summary: List[Tuple[int, int, float, int, Optional[int]]] = []
    for p in PRIMES:
        rounds_ok, delta_fail = _run_stress(p, n=N_PARTIES, t=2, max_rounds=50)
        theoretical = math.log(p / N_PARTIES) / (2.0 * math.log(N_PARTIES))
        summary.append((p, N_PARTIES, theoretical, rounds_ok, delta_fail))

    # ── Summary table ─────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  Summary")
    print("=" * 70)
    header = f"  {'p':>12}  {'N':>2}  {'theory R_max':>13}  {'actual R_max':>12}  {'D at failure':>13}"
    print(header)
    print(f"  {'-'*12}  {'-'*2}  {'-'*13}  {'-'*12}  {'-'*13}")
    for p, n_val, theory, actual, delta_fail in summary:
        p_str = "2^31-1" if p == 2 ** 31 - 1 else str(p)
        delta_str = str(delta_fail) if delta_fail is not None else f">max (>={2**actual})"
        print(
            f"  {p_str:>12}  {n_val:>2}  {theory:>13.1f}  {actual:>12}  {delta_str:>13}"
        )
    print()
    print("  Formula: theoretical R_max = log(p/N) / (2*log N)")
    print("  Actual R_max = rounds completed before first wrong winner.")
    print("  Note: actual > theory because |A(m*)|=2 < N=3 in this ballot setup.")
    print("=" * 70)

    # ── Correctness assertions ─────────────────────────────────────────────
    # Every prime must answer at least one round correctly.
    for p, _, _, actual, _ in summary:
        assert actual >= 1, f"p={p}: failed on round 1 — unexpected"

    # Monotonicity: a larger field must last at least as long.
    assert summary[1][3] >= summary[0][3], \
        "p=65537 must survive ≥ as many rounds as p=251"
    assert summary[2][3] >= summary[1][3], \
        "p=2^31-1 must survive ≥ as many rounds as p=65537"
