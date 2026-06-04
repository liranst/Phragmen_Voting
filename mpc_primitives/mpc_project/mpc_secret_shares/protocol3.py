"""
Protocol 3 — RNG (Shared Random Number Generation)

Each party P_i contributes a secret random value r_i; the shared secret
is r = Σ r_i (mod p).  No single party knows r, yet every party holds a
valid share of it.

Reference: "SoK: Secure Computation over Secret Shares", Protocol 3.
"""

import random
from typing import List, Tuple

from .protocol1 import protocol_1_share

Shares = List[Tuple[int, int]]


def protocol_3_rng(
    n: int,
    t: int,
    p: int,
    print_: bool = False,
) -> Shares:
    """Generate a (t,n)-sharing of a uniformly random secret r ∈ Z_p.

    Parameters
    ----------
    n : int
        Number of parties.
    t : int
        Reconstruction threshold.
    p : int
        Prime field modulus.
    print_ : bool
        Verbose output for debugging.

    Returns
    -------
    Shares
        A (t,n)-sharing of r = Σ r_i (mod p).
    """
    # Each party P_i generates r_i and shares it
    all_generated_shares: List[Shares] = []
    for i in range(1, n + 1):
        r_i = random.randint(0, p - 1)          # Line 2: P_i draws r_i
        if print_:
            print(f"P{i}: r_{i} = {r_i}")
        shares_from_i = protocol_1_share(r_i, n, t, p)  # Line 3: Share(r_i)
        all_generated_shares.append(shares_from_i)

    # Each party P_j sums the shares it received from all P_i
    final_random_shares: Shares = []
    for j in range(1, n + 1):
        sum_y = 0
        for shares_from_i in all_generated_shares:
            x_val, y_val = shares_from_i[j - 1]
            sum_y = (sum_y + y_val) % p          # Line 5: [[r]]_j = Σ [[r_i]]_j
        if print_:
            print(f"P{j}: [[r]]_{j} = {sum_y}")
        final_random_shares.append((j, sum_y))

    return final_random_shares
