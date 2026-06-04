"""
Protocol 15 — Secure Bitwise Extraction of Square Roots

Compute a sharing of  w = ⌊√u⌋  given sharings of u and its bits.
Mirrors Algorithm 14 with every cleartext comparison replaced by
SecureCompare and every conditional update replaced by SecureMult.

Reference: "SoK: Secure Computation over Secret Shares", Protocol 15.
"""

import math
from typing import List, Tuple

from .protocol4 import protocol_4_secure_mult
from .protocol5_secure_compare import protocol_5_secure_compare

Shares = List[Tuple[int, int]]


def protocol_15_secure_sqrt(
    u_shares: Shares,
    u_bit_shares: List[Shares],
    n: int,
    t: int,
    p: int,
    s: int = None,
    print_: bool = False,
) -> Shares:
    """Return a (t,n)-sharing of  w = ⌊√u⌋.

    Parameters
    ----------
    u_shares : Shares
        (t,n)-sharing of u ∈ Z_p.
    u_bit_shares : list of Shares
        ``u_bit_shares[j]`` = sharing of bit j of u (LSB = 0, MSB = s-1).
    n, t, p : int
        MPC parameters.
    s : int, optional
        Number of bits (padded to even if needed).  Defaults to ⌈log₂(p)⌉.
    print_ : bool
        Verbose debug output.

    Returns
    -------
    Shares
        A (t,n)-sharing [[w]] of w = ⌊√u⌋.
    """
    if s is None:
        s = math.ceil(math.log2(p)) if p > 2 else 1

    if s % 2 == 1:                        # pad to even for bit-pair processing
        s += 1
        u_bit_shares = list(u_bit_shares) + [[(j + 1, 0) for j in range(n)]]

    w_shares: Shares = [(j + 1, 0) for j in range(n)]   # Line 1
    r_shares: Shares = [(j + 1, 0) for j in range(n)]   # Line 2

    j = s - 2
    while j >= 0:                                         # Line 3
        # Line 4: [[r]] ← 4[[r]] + 2[[u_{j+1}]] + [[u_j]]  (local)
        r_shares = [
            (x, (4 * yr + 2 * yu1 + yu) % p)
            for (x, yr), (_, yu1), (_, yu)
            in zip(r_shares, u_bit_shares[j + 1], u_bit_shares[j])
        ]

        # Line 5: [[y]] ← 4[[w]] + [[1]]  (local)
        y_shares: Shares = [(x, (4 * yw + 1) % p) for x, yw in w_shares]

        # Line 6: [[b]] = 1 − SecureCompare([[r]], [[y]])
        cmp = protocol_5_secure_compare(r_shares, y_shares, n, t, p, print_=False)
        b_shares: Shares = [(x, (1 - yc) % p) for x, yc in cmp]

        if print_:
            from .protocol2 import protocol_2_reconstruct
            print(f"[P15] j={j}: b={protocol_2_reconstruct(b_shares[:t], p)}")

        # Line 7: [[r]] ← [[r]] − SecureMult([[b]], [[y]])
        by = protocol_4_secure_mult(b_shares, y_shares, n, t, p, print_=False)
        r_shares = [
            (x, (yr - yby) % p)
            for (x, yr), (_, yby) in zip(r_shares, by)
        ]

        # Line 8: [[w]] ← 2[[w]] + [[b]]  (local)
        w_shares = [
            (x, (2 * yw + yb) % p)
            for (x, yw), (_, yb) in zip(w_shares, b_shares)
        ]

        j -= 2

    return w_shares
