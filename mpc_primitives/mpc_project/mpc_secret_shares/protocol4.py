"""
Protocol 4 — SecureMult (Secure Multiplication)

Compute a (t,n)-sharing of w = u·v given (t,n)-sharings of u and v,
without any party learning u, v, or w.

Uses the randomization technique: each party P_i contributes random values
r_i (shared at degree t) and R_i (shared at degree 2t−1).  The degree-2(t−1)
polynomial formed by local products is reduced back to degree t−1 via
reconstruction and subtraction of the mask.

Reference: "SoK: Secure Computation over Secret Shares", Protocol 4.
"""

import random
from typing import List, Tuple

from .protocol1 import protocol_1_share
from .protocol2 import protocol_2_reconstruct

Shares = List[Tuple[int, int]]


def protocol_4_secure_mult(
    shares_u: Shares,
    shares_v: Shares,
    n: int,
    t: int,
    P: int,
    print_: bool = False,
) -> Shares:
    """Return a (t,n)-sharing of u·v.

    Parameters
    ----------
    shares_u, shares_v : Shares
        (t,n)-sharings of the secrets u and v.
    n : int
        Number of parties.
    t : int
        Reconstruction threshold.
    P : int
        Prime field modulus.
    print_ : bool
        Verbose output for debugging.

    Returns
    -------
    Shares
        A (t,n)-sharing of w = u·v (mod P).
    """
    # ------------------------------------------------------------------
    # Lines 1-5: Each P_i generates r_i and shares it at both degree t
    #            (for the final mask) and degree 2t-1 (for the noise term).
    # ------------------------------------------------------------------
    all_shares_r_t:   List[Shares] = []
    all_shares_R_2t1: List[Shares] = []

    for i in range(1, n + 1):
        r_i = random.randint(0, P - 1)
        shares_r = protocol_1_share(r_i, n, t, P)
        all_shares_r_t.append(shares_r)
        shares_R = protocol_1_share(r_i, n, 2 * t - 1, P)
        all_shares_R_2t1.append(shares_R)

    if print_:
        print(f"[SecureMult] r sharings (t): {all_shares_r_t}")
        print(f"[SecureMult] R sharings (2t-1): {all_shares_R_2t1}")

    # ------------------------------------------------------------------
    # Lines 6-8: P_j sums its column across all sharings to obtain [[r]]
    #            and [[R]].
    # ------------------------------------------------------------------
    shares_r_final: Shares = []
    shares_R_final: Shares = []

    for j in range(1, n + 1):
        sum_r = sum(all_shares_r_t[i][j - 1][1]   for i in range(n)) % P
        sum_R = sum(all_shares_R_2t1[i][j - 1][1] for i in range(n)) % P
        shares_r_final.append((j, sum_r))
        shares_R_final.append((j, sum_R))

    # ------------------------------------------------------------------
    # Lines 9-10: P_i computes the masked local product
    #             z_i = u_i * v_i + R_i  (mod P)
    # This is evaluations of a degree-2(t-1) polynomial for u·v + R.
    # ------------------------------------------------------------------
    shares_z: Shares = []
    for i in range(1, n + 1):
        u_i = shares_u[i - 1][1]
        v_i = shares_v[i - 1][1]
        R_i = shares_R_final[i - 1][1]
        z_i = (u_i * v_i + R_i) % P
        shares_z.append((i, z_i))

    # ------------------------------------------------------------------
    # Lines 11-12: Reconstruct z = u·v + R publicly (needs 2t-1 shares).
    # ------------------------------------------------------------------
    required = 2 * t - 1
    z_reconstructed = protocol_2_reconstruct(shares_z[:required], P)
    if print_:
        print(f"[SecureMult] z (public) = {z_reconstructed}")

    # ------------------------------------------------------------------
    # Lines 13-14: Each P_i removes the mask: w_i = z − r_i  (mod P).
    #              The result is a fresh degree-(t-1) sharing of u·v.
    # ------------------------------------------------------------------
    shares_w: Shares = []
    for i in range(1, n + 1):
        r_i = shares_r_final[i - 1][1]
        w_i = (z_reconstructed - r_i) % P
        shares_w.append((i, w_i))

    return shares_w
