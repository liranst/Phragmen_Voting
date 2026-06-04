"""
Algorithm 14 — Bitwise Extraction of Square Roots (Cleartext)
Compute ⌊√u⌋ for a non-negative integer u given its bit representation.

This is a *cleartext* algorithm operating on regular Python integers.
It is used as a reference / test oracle for Protocol 15 (the secure version).

Pseudocode reference (line numbers match the paper):
---------------------------------------------------------------------------
Input:  u ∈ Z, u ∈ [0, p);
        u_{s-1}, ..., u_0  such that u = (u_{s-1}, ..., u_0)_2

Line 1: w ← 0
Line 2: r ← 0
Line 3: for all j = s-2, s-4, ..., 2, 0 do
Line 4:   r ← 4r + 2u_{j+1} + u_j
Line 5:   y ← 4w + 1
Line 6:   if y ≤ r then
Line 7:     r ← r − y
Line 8:     w ← 2w + 1
Line 9:   else
Line 10:    w ← 2w
Output: The rounded square root w = ⌊√u⌋
---------------------------------------------------------------------------

Intuition (non-restoring square root in binary)
-----------------------------------------------
The algorithm processes pairs of bits at a time, from the most significant
pair down.  It maintains:
    w  — the running square root estimate (so far)
    r  — the running remainder  (r = u_processed − w²)

At each step the next two bits (u_{j+1}, u_j) are appended to r:
    r ← 4r + 2u_{j+1} + u_j
which corresponds to left-shifting r by 2 and inserting the next bit-pair.

Then y = 4w + 1 is the "trial subtrahend":
    If we set the next root bit to 1, the new estimate is w' = 2w + 1
    and we need to subtract (2w+1)² − (2w)² = 4w + 1 = y from r.

    y ≤ r  →  the next bit is 1: r ← r − y,  w ← 2w + 1
    y > r  →  the next bit is 0:              w ← 2w

Note: s must be even (the loop uses pairs of bits).  If s is odd, pad the
bit list with a leading zero so that s becomes even.
"""

import math
from typing import List


def algorithm_14_bitwise_sqrt(
    u: int,
    u_bits: List[int] = None,
    s: int = None,
) -> int:
    """Return ⌊√u⌋ using bitwise (non-restoring) square root extraction.

    Parameters
    ----------
    u      : int
        Non-negative integer whose square root is to be computed.
    u_bits : List[int], optional
        Bit decomposition of u, with u_bits[i] = i-th bit (LSB = index 0).
        If omitted, computed from u automatically.
    s      : int, optional
        Number of bits to use.  Defaults to ⌈log₂(u+1)⌉ rounded up to
        the nearest even number.

    Returns
    -------
    int
        ⌊√u⌋  (the floor of the square root of u).
    """
    if u < 0:
        raise ValueError("u must be non-negative")
    if u == 0:
        return 0

    # Determine the number of bits (must be even for bit-pair processing)
    if s is None:
        s = math.ceil(math.log2(u + 1)) if u > 0 else 1
        if s % 2 == 1:
            s += 1          # pad to even so pairs always align

    # Build the bit array if not supplied
    if u_bits is None:
        u_bits = [(u >> i) & 1 for i in range(s)]
    else:
        # Pad with zeros if the supplied list is shorter than s
        u_bits = list(u_bits) + [0] * max(0, s - len(u_bits))

    # ------------------------------------------------------------------
    # Lines 1-2: Initialise w and r
    # ------------------------------------------------------------------
    w = 0   # Line 1
    r = 0   # Line 2

    # ------------------------------------------------------------------
    # Line 3: Process pairs of bits from MSB to LSB
    # j runs from s-2 down to 0 in steps of 2.
    # Each iteration handles the bit pair (u_{j+1}, u_j).
    # ------------------------------------------------------------------
    j = s - 2
    while j >= 0:                                                    # Line 3

        # Line 4: r ← 4r + 2u_{j+1} + u_j
        r = 4 * r + 2 * u_bits[j + 1] + u_bits[j]                  # Line 4

        # Line 5: y ← 4w + 1  (trial subtrahend for setting next root bit)
        y = 4 * w + 1                                               # Line 5

        if y <= r:                                                   # Line 6
            r = r - y                                               # Line 7
            w = 2 * w + 1                                           # Line 8
        else:
            w = 2 * w                                               # Line 10

        j -= 2

    return w


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random
    random.seed(0)

    all_ok = True
    for u in list(range(0, 50)) + [random.randint(0, 1000) for _ in range(20)]:
        expected = int(math.isqrt(u))
        got      = algorithm_14_bitwise_sqrt(u)
        status   = "✓" if got == expected else "✗"
        if got != expected:
            print(f"√{u}: got {got}, expected {expected} {status}")
            all_ok = False

    print(f"Tested {50 + 20} values.")
    print("Algorithm 14 self-test", "PASSED ✓" if all_ok else "FAILED ✗")
