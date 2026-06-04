"""
Protocol 15 — Secure Bitwise Extraction of Square Roots
Compute a sharing of ⌊√u⌋ given sharings of u and its bits.

Pseudocode reference (line numbers match the paper):
---------------------------------------------------------------------------
Input:  [[u]], and [[u_j]], j ∈ {0,1,...,s-1}
        such that u = (u_{s-1}, ..., u_0)_2  (MSB = index s-1)

Line 1: [[w]] ← [[0]]
Line 2: [[r]] ← [[0]]
Line 3: for all j = s-2, s-4, ..., 2, 0 do
Line 4:   [[r]] ← 4[[r]] + 2[[u_{j+1}]] + [[u_j]]
Line 5:   [[y]] ← 4[[w]] + [[1]]
Line 6:   [[b]] ← [[1]] − SecureCompare([[r]], [[y]])
Line 7:   [[r]] ← [[r]] − SecureMult([[b]], [[y]])
Line 8:   [[w]] ← 2[[w]] + [[b]]
Output:  A sharing [[w]] of w = ⌊√u⌋
---------------------------------------------------------------------------

This is the secure analogue of Algorithm 14.  The structure is identical;
every cleartext comparison is replaced by SecureCompare and every
conditional update is replaced by SecureMult.

Steps 4-5 are *local* (scalar multiplication and addition of a public
constant) because all coefficients are public scalars.

Step 6: [[b]] = 1_{r ≥ y} = 1 − 1_{r < y} = 1 − SecureCompare([[r]], [[y]]).

Step 7: [[r]] ← [[r]] − [[b]] · [[y]]
    b = 1  →  r ← r − y   (subtract the trial value)
    b = 0  →  r ← r − 0 = r  (keep r unchanged)

Step 8: [[w]] ← 2[[w]] + [[b]]
    Shifts the running estimate left and appends the new root bit.

Note: s must be even (pairs of bits processed per iteration).  Pad to an
even number if s is odd by appending a zero bit-sharing at position s.
"""

import math
from typing import List, Tuple

try:
    from MPC_over_secret_shares.Protocol4 import protocol_4_secure_mult
    from MPC_over_secret_shares.Protocol5_SecureCompare import protocol_5_secure_compare
except ImportError:
    from Protocol4 import protocol_4_secure_mult
    from Protocol5_SecureCompare import protocol_5_secure_compare

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
    """Return a (t,n)-sharing of ⌊√u⌋.

    Parameters
    ----------
    u_shares     : Shares
        A (t,n)-sharing of u ∈ Z_p.
    u_bit_shares : List[Shares]
        List of length s where u_bit_shares[j] = [[u_j]], a (t,n)-sharing
        of the j-th bit of u (LSB = index 0, MSB = index s-1).
    n, t, p      : int
        MPC parameters.
    s            : int, optional
        Number of bits.  Defaults to ⌈log₂(p)⌉ rounded up to even.
    print_       : bool
        Verbose debug output.

    Returns
    -------
    Shares
        A (t,n)-sharing [[w]] where w = ⌊√u⌋.
    """
    if s is None:
        s = math.ceil(math.log2(p)) if p > 2 else 1

    # Ensure s is even (pairs of bits per iteration)
    if s % 2 == 1:
        s += 1
        # Extend u_bit_shares with a sharing of 0 at the new top position
        u_bit_shares = list(u_bit_shares) + [[(j + 1, 0) for j in range(n)]]

    # ------------------------------------------------------------------
    # Lines 1-2: Initialise [[w]] = [[0]] and [[r]] = [[0]]
    # ------------------------------------------------------------------
    w_shares: Shares = [(j + 1, 0) for j in range(n)]              # Line 1
    r_shares: Shares = [(j + 1, 0) for j in range(n)]              # Line 2

    # ------------------------------------------------------------------
    # Line 3: Process bit-pairs from MSB-1 down to 0 (step -2)
    # ------------------------------------------------------------------
    j = s - 2
    while j >= 0:                                                    # Line 3

        # Line 4: [[r]] ← 4[[r]] + 2[[u_{j+1}]] + [[u_j]]  (local)
        r_shares = [
            (x, (4 * yr + 2 * yu1 + yu) % p)
            for (x, yr), (_, yu1), (_, yu)
            in zip(r_shares, u_bit_shares[j + 1], u_bit_shares[j])
        ]                                                            # Line 4

        # Line 5: [[y]] ← 4[[w]] + [[1]]  (local)
        # y = 4w + 1 is the trial subtrahend.
        y_shares: Shares = [(x, (4 * yw + 1) % p) for x, yw in w_shares]  # Line 5

        # Line 6: [[b]] ← [[1]] − SecureCompare([[r]], [[y]])
        # b = 1  iff  r ≥ y  (next root bit is 1)
        cmp_shares = protocol_5_secure_compare(
            r_shares, y_shares, n, t, p, print_=False
        )
        b_shares: Shares = [(x, (1 - yc) % p) for x, yc in cmp_shares]  # Line 6

        if print_:
            from Protocol2 import protocol_2_reconstruct
            b_val = protocol_2_reconstruct(b_shares[:t], p)
            print(f"[Protocol15] j={j}: b = {b_val}")

        # Line 7: [[r]] ← [[r]] − SecureMult([[b]], [[y]])  (conditional subtract)
        by_shares = protocol_4_secure_mult(
            b_shares, y_shares, n, t, p, print_=False
        )
        r_shares = [
            (x, (yr - yby) % p)
            for (x, yr), (_, yby) in zip(r_shares, by_shares)
        ]                                                            # Line 7

        # Line 8: [[w]] ← 2[[w]] + [[b]]  (local)
        w_shares = [
            (x, (2 * yw + yb) % p)
            for (x, yw), (_, yb) in zip(w_shares, b_shares)
        ]                                                            # Line 8

        j -= 2

    if print_:
        from Protocol2 import protocol_2_reconstruct
        w_val = protocol_2_reconstruct(w_shares[:t], p)
        print(f"[Protocol15] √u = {w_val}")

    return w_shares


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import math
    import random
    from Protocol1 import protocol_1_share
    from Protocol2 import protocol_2_reconstruct
    from Algorithm14_BitwiseSqrt import algorithm_14_bitwise_sqrt

    random.seed(55)
    P, N, T = 11, 3, 2
    S = math.ceil(math.log2(P))
    if S % 2 == 1:
        S += 1

    def share_bit(b_val: int) -> Shares:
        return protocol_1_share(b_val, N, T, P)

    cases = [0, 1, 2, 4, 5, 9]   # values in Z_11 with meaningful sqrt
    all_ok = True
    for u in cases:
        expected = algorithm_14_bitwise_sqrt(u, s=S)
        u_bits   = [(u >> i) & 1 for i in range(S)]
        u_bit_sh = [share_bit(bi) for bi in u_bits]
        sh_u     = protocol_1_share(u, N, T, P)

        w_sh = protocol_15_secure_sqrt(sh_u, u_bit_sh, N, T, P, s=S)
        w    = protocol_2_reconstruct(w_sh[:T], P)

        status = "✓" if w == expected else "✗"
        print(f"√{u} = {w}  (expected {expected}) {status}")
        all_ok = all_ok and (w == expected)

    print("Protocol 15 self-test", "PASSED ✓" if all_ok else "FAILED ✗")
