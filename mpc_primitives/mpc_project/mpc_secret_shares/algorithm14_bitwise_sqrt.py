"""
Algorithm 14 — Bitwise Extraction of Square Roots (Cleartext)

Compute floor(sqrt(u)) for a non-negative integer u given its binary
representation.  Processes pairs of bits at a time using the non-restoring
binary square-root algorithm.

Note: the number of bits *s* must be even.  If *s* is odd it is
automatically padded to the next even number.

Reference: "SoK: Secure Computation over Secret Shares", Algorithm 14.
"""

import math
from typing import List, Optional


def algorithm_14_bitwise_sqrt(
    u: int,
    u_bits: Optional[List[int]] = None,
    s: Optional[int] = None,
) -> int:
    """Return ``floor(sqrt(u))`` via bitwise (non-restoring) extraction.

    Parameters
    ----------
    u      : int
        Non-negative integer.
    u_bits : list of int, optional
        Bit decomposition of *u* with ``u_bits[i]`` = i-th bit (LSB=index 0).
        Computed from *u* automatically when omitted.
    s      : int, optional
        Number of bits to use (automatically padded to even if odd).
        Defaults to ``ceil(log2(u+1))`` rounded up to the nearest even number.

    Returns
    -------
    int
        ``floor(sqrt(u))``.
    """
    if u < 0:
        raise ValueError("u must be non-negative")
    if u == 0:
        return 0

    # Determine bit-length, padded to even
    if s is None:
        s = math.ceil(math.log2(u + 1)) if u > 0 else 1
        if s % 2 == 1:
            s += 1

    # Build or pad the bit array (index 0 = LSB)
    if u_bits is None:
        u_bits = [(u >> i) & 1 for i in range(s)]
    else:
        u_bits = list(u_bits) + [0] * max(0, s - len(u_bits))

    # ------------------------------------------------------------------
    # Lines 1-2
    # ------------------------------------------------------------------
    w = 0   # Line 1: running square-root estimate
    r = 0   # Line 2: running remainder

    # ------------------------------------------------------------------
    # Line 3: process bit-pairs (j+1, j) from MSB to LSB, step −2
    # ------------------------------------------------------------------
    j = s - 2
    while j >= 0:                                   # Line 3
        r = 4 * r + 2 * u_bits[j + 1] + u_bits[j] # Line 4
        y = 4 * w + 1                               # Line 5
        if y <= r:                                  # Line 6
            r = r - y                               # Line 7
            w = 2 * w + 1                           # Line 8
        else:
            w = 2 * w                               # Line 10
        j -= 2

    return w
