"""
Protocol 13 — Secure Binary Long Division

Compute sharings of  q = ⌊u/v⌋  and  r = u mod v  given sharings of u, v,
and the individual bit-sharings of u.

Reference: "SoK: Secure Computation over Secret Shares", Protocol 13.
"""

import math
from typing import List, Tuple

from .protocol4 import protocol_4_secure_mult
from .protocol5_secure_compare import protocol_5_secure_compare

Shares = List[Tuple[int, int]]


def protocol_13_secure_division(
    u_shares: Shares,
    v_shares: Shares,
    u_bit_shares: List[Shares],
    n: int,
    t: int,
    p: int,
    s: int = None,
    print_: bool = False,
) -> Tuple[Shares, Shares, List[Tuple[int, Shares]]]:
    """Return sharings of  q = ⌊u/v⌋  and  r = u mod v.

    Parameters
    ----------
    u_shares : Shares
        (t,n)-sharing of u ∈ Z_p.
    v_shares : Shares
        (t,n)-sharing of v ∈ (0, p).
    u_bit_shares : list of Shares
        ``u_bit_shares[j]`` = (t,n)-sharing of the j-th bit of u
        (LSB = index 0, MSB = index s-1).
    n, t, p : int
        MPC parameters.
    s : int, optional
        Number of bits.  Defaults to ⌈log₂(p)⌉.
    print_ : bool
        Verbose debug output.

    Returns
    -------
    q_shares : Shares
        (t,n)-sharing of q = ⌊u/v⌋.
    r_shares : Shares
        (t,n)-sharing of r = u mod v.
    q_bit_shares : list of (int, Shares)
        Bit-sharings of q: each entry is ``(bit_index, [[q_j]])``.
    """
    if s is None:
        s = math.ceil(math.log2(p)) if p > 2 else 1

    q_shares: Shares = [(j + 1, 0) for j in range(n)]   # Line 1: [[q]] ← [[0]]
    r_shares: Shares = [(j + 1, 0) for j in range(n)]   # Line 2: [[r]] ← [[0]]
    q_bit_shares: List[Tuple[int, Shares]] = []

    for j in range(s - 1, -1, -1):                       # Line 3
        # Line 4: [[r]] ← 2[[r]] + [[u_j]]  (local)
        r_shares = [
            (x, (2 * yr + yu) % p)
            for (x, yr), (_, yu) in zip(r_shares, u_bit_shares[j])
        ]

        # Line 5: [[q_j]] = 1 − SecureCompare([[r]], [[v]])
        cmp = protocol_5_secure_compare(r_shares, v_shares, n, t, p, print_=False)
        q_j: Shares = [(x, (1 - yc) % p) for x, yc in cmp]
        q_bit_shares.append((j, q_j))

        if print_:
            from .protocol2 import protocol_2_reconstruct
            print(f"[P13] j={j}: q_j={protocol_2_reconstruct(q_j[:t], p)}")

        # Line 6: [[r]] ← [[r]] − SecureMult([[q_j]], [[v]])
        qv = protocol_4_secure_mult(q_j, v_shares, n, t, p, print_=False)
        r_shares = [
            (x, (yr - yqv) % p)
            for (x, yr), (_, yqv) in zip(r_shares, qv)
        ]

        # Line 7: [[q]] ← 2[[q]] + [[q_j]]  (local)
        q_shares = [
            (x, (2 * yq + yqj) % p)
            for (x, yq), (_, yqj) in zip(q_shares, q_j)
        ]

    return q_shares, r_shares, q_bit_shares
