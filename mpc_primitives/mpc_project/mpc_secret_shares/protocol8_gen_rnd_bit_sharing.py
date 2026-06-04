"""
Protocol 8 — GenRndBitSharing

Generate a (t,n)-sharing of a uniformly random bit b ∈ {0, 1}.

Key insight: each party contributes r_i; their sum r = Σ r_i is unknown.
Local squaring yields a degree-2(t-1) sharing of r²; reconstruction reveals
r² without revealing r.  One square root r′ of r² is chosen consistently,
and the formula  b = ((p+1)/2)·(r′⁻¹·[[r]] + 1)  maps the two roots to
{0, 1} with equal probability.

Requires n ≥ 2t−1.

Reference: "SoK: Secure Computation over Secret Shares", Protocol 8.
"""

import random
import math
from typing import List, Tuple

from .protocol1 import protocol_1_share
from .protocol2 import protocol_2_reconstruct

Shares = List[Tuple[int, int]]


def _modular_sqrt(a: int, p: int) -> int:
    """Return one square root of *a* modulo prime *p* (Tonelli-Shanks).

    Raises ``ValueError`` if *a* is not a quadratic residue mod *p*.
    """
    a = a % p
    if a == 0:
        return 0
    if pow(a, (p - 1) // 2, p) != 1:
        raise ValueError(f"{a} is not a quadratic residue mod {p}")
    if p % 4 == 3:
        return pow(a, (p + 1) // 4, p)

    # Tonelli-Shanks
    Q, S = p - 1, 0
    while Q % 2 == 0:
        Q //= 2
        S += 1
    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1
    M = S
    c = pow(z, Q, p)
    t_val = pow(a, Q, p)
    R = pow(a, (Q + 1) // 2, p)
    while True:
        if t_val == 1:
            return R
        i, temp = 1, (t_val * t_val) % p
        while temp != 1:
            temp = (temp * temp) % p
            i += 1
        b = pow(c, 1 << (M - i - 1), p)
        M = i
        c = (b * b) % p
        t_val = (t_val * c) % p
        R = (R * b) % p


def protocol_8_gen_rnd_bit_sharing(
    n: int,
    t: int,
    p: int,
    print_: bool = False,
) -> Shares:
    """Return a (t,n)-sharing of a uniformly random bit b ∈ {0, 1}.

    Parameters
    ----------
    n, t, p : int
        MPC parameters.  Requires n ≥ 2t−1.
    print_ : bool
        Verbose debug output.

    Returns
    -------
    Shares
        A (t,n)-sharing [[b]] with b ∈ {0, 1} uniformly at random.
    """
    while True:
        # Lines 1-5: each P_i shares r_i; P_j accumulates [[r]]_j = Σ [[r_i]]_j
        all_shares: List[Shares] = []
        for _ in range(n):                                         # Line 1
            r_i = random.randint(1, p - 1)                        # Line 2
            all_shares.append(protocol_1_share(r_i, n, t, p))     # Line 3

        r_shares: Shares = []
        for j in range(n):                                         # Line 4
            x_j   = all_shares[0][j][0]
            total = sum(all_shares[i][j][1] for i in range(n)) % p
            r_shares.append((x_j, total))                         # Line 5

        # Line 6: local squaring → degree-2(t-1) sharing of r²
        r2_shares: Shares = [(x, (y * y) % p) for x, y in r_shares]

        # Lines 7-8: reconstruct r² (needs 2t-1 shares)
        needed = 2 * t - 1
        if len(r2_shares) < needed:
            raise RuntimeError(
                f"GenRndBitSharing requires n ≥ 2t-1 = {needed}, got n = {n}"
            )
        r2 = protocol_2_reconstruct(r2_shares[:needed], p)        # Line 7-8
        if print_:
            print(f"[Protocol8] r² = {r2}")

        if r2 == 0:                                                # Lines 9-10
            if print_:
                print("[Protocol8] r² = 0 → retrying")
            continue

        # Lines 11-12: compute r′ = sqrt(r²) and r′⁻¹
        r_prime     = _modular_sqrt(r2, p)                        # Line 11
        r_prime_inv = pow(r_prime, p - 2, p)                      # Line 12

        # Line 13: [[b]] = ((p+1)/2) · (r′⁻¹ · [[r]] + 1)
        scaled   = [(x, (r_prime_inv * y) % p) for x, y in r_shares]
        plus_one = [(x, (y + 1) % p) for x, y in scaled]
        half     = (p + 1) // 2
        b_shares = [(x, (half * y) % p) for x, y in plus_one]    # Line 13

        if print_:
            b_val = protocol_2_reconstruct(b_shares[:t], p)
            print(f"[Protocol8] random bit b = {b_val}")

        return b_shares
