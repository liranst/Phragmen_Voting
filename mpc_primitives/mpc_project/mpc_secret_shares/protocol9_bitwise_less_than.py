"""
Protocol 9 — Bitwise_LessThan

Compute a sharing of 1_{a < b} where *a* is a public integer and *b* is
given through (t,n)-sharings of its individual bits.

Reference: "SoK: Secure Computation over Secret Shares", Protocol 9.
"""

import math
from typing import List, Tuple

from .protocol4 import protocol_4_secure_mult

Shares = List[Tuple[int, int]]


def protocol_9_bitwise_less_than(
    a: int,
    b_bit_shares: List[Shares],
    n: int,
    t: int,
    p: int,
    print_: bool = False,
) -> Shares:
    """Return a (t,n)-sharing of ``1_{a < b}``.

    Parameters
    ----------
    a : int
        The public comparand (integer in Z_p).
    b_bit_shares : list of Shares
        ``b_bit_shares[i]`` = (t,n)-sharing of bit i of b (LSB = index 0).
    n, t, p : int
        MPC parameters.
    print_ : bool
        Verbose debug output.

    Returns
    -------
    Shares
        A (t,n)-sharing [[f]] where f = 1 if a < b, else 0.
    """
    s      = len(b_bit_shares)
    a_bits = [(a >> i) & 1 for i in range(s)]   # LSB first

    # Lines 1-5: c_i = b_i if a_i=0, else c_i = 1−b_i
    c_shares: List[Shares] = []
    for i in range(s):
        if a_bits[i] == 0:                                         # Line 2
            c_shares.append(b_bit_shares[i])                      # Line 3
        else:
            c_i = [(x, (1 - y) % p) for x, y in b_bit_shares[i]] # Line 5
            c_shares.append(c_i)

    # Lines 6-7: initialise d and e at MSB
    d_shares: List[Shares] = [None] * s
    e_shares: List[Shares] = [None] * s
    d_shares[s - 1] = c_shares[s - 1]
    e_shares[s - 1] = c_shares[s - 1]

    # Lines 8-10: sweep from bit s-2 down to 0
    for i in range(s - 2, -1, -1):                                # Line 8
        mult = protocol_4_secure_mult(
            d_shares[i + 1], c_shares[i], n, t, p, print_=False
        )
        d_i: Shares = [
            (x, (yd + yc - ym) % p)
            for (x, yd), (_, yc), (_, ym)
            in zip(d_shares[i + 1], c_shares[i], mult)
        ]                                                          # Line 9
        d_shares[i] = d_i
        e_i: Shares = [
            (x, (yd - yd1) % p)
            for (x, yd), (_, yd1) in zip(d_i, d_shares[i + 1])
        ]                                                          # Line 10
        e_shares[i] = e_i

    # Line 11: ê = Σ e_i  (local sum)
    hat_e: Shares = [(j + 1, 0) for j in range(n)]
    for i in range(s):
        hat_e = [(x, (he + ye) % p)
                 for (x, he), (_, ye) in zip(hat_e, e_shares[i])]

    # Line 12: g = 1 − Σ_{a_i=1} e_i  (local)
    sum_e_a: Shares = [(j + 1, 0) for j in range(n)]
    for i in range(s):
        if a_bits[i] == 1:
            sum_e_a = [(x, (sv + ye) % p)
                       for (x, sv), (_, ye) in zip(sum_e_a, e_shares[i])]
    g: Shares = [(x, (1 - sg) % p) for x, sg in sum_e_a]

    # Line 13: f = SecureMult(ê, g)
    return protocol_4_secure_mult(hat_e, g, n, t, p, print_=False)
