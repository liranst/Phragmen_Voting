"""
Protocol 7 — LSB
Secure computation of a sharing of the least significant bit (LSB) of a
shared secret x ∈ Z_p.

Pseudocode reference (line numbers match the paper):
---------------------------------------------------------------------------
Input:   [[x]] — a (t,n)-sharing of x ∈ Z_p
Parameter: s = ⌈log₂(p)⌉

Line 1:  for all 0 ≤ i ≤ s-1 do
Line 2:    The parties run [[r_i]] ← GenRndBitSharing()
Line 3:  for all j ∈ [n] do
Line 4:    P_j sets [[r]]_j = Σ_{i=0}^{s-1} 2^i [[r_i]]_j
Line 5:  [[c]] ← [[x]] + [[r]]
Line 6:  The parties run Reconstruct([[c]], 1; t)
Line 7:  P_1 broadcasts c
Line 8:  if c_0 = 0 then
Line 9:    [[d_0]] ← [[r_0]]
Line 10: else
Line 11:   [[d_0]] ← 1 − [[r_0]]
Line 12: [[e]]   ← Bitwise_LessThan(c, {[[r_i]]}_{0≤i≤s-1})
Line 13: [[y]]   ← SecureMult([[e]], [[d_0]])
Line 14: [[x_0]] ← [[e]] + [[d_0]] − 2[[y]]
Output:  A (t,n)-sharing of [[x_0]] = LSB(x)
---------------------------------------------------------------------------

Intuition
---------
r is a random s-bit integer whose bits [[r_i]] are individually shared.
After masking, c = x + r mod p is public.

Let x_0 and r_0 denote the LSBs of x and r respectively.

Case A (c < r, i.e. a modular wrap occurred — x + r ≥ p):
    x = c + p − r   →   x_0 = (c_0 + 1 − r_0) mod 2  = c_0 XOR r_0 XOR 1
                                                       = NOT(d_0)  where d_0 = c_0 XOR r_0
    [[e]] = 1  →  [[x_0]] = 1 + d_0 − 2·d_0 = 1 − d_0 = NOT(d_0)  ✓

Case B (c ≥ r, no wrap):
    x = c − r        →   x_0 = (c_0 − r_0) mod 2  = c_0 XOR r_0 = d_0
    [[e]] = 0  →  [[x_0]] = 0 + d_0 − 0            = d_0            ✓

The formula [[x_0]] = [[e]] + [[d_0]] − 2[[y]] is thus the XOR of [[e]] and [[d_0]].
"""

import math
from typing import List, Tuple

try:
    from MPC_over_secret_shares.Protocol2 import protocol_2_reconstruct
    from MPC_over_secret_shares.Protocol4 import protocol_4_secure_mult
    from MPC_over_secret_shares.Protocol8_GenRndBitSharing import protocol_8_gen_rnd_bit_sharing
    from MPC_over_secret_shares.Protocol9_BitwiseLessThan import protocol_9_bitwise_less_than
except ImportError:
    from Protocol2 import protocol_2_reconstruct
    from Protocol4 import protocol_4_secure_mult
    from Protocol8_GenRndBitSharing import protocol_8_gen_rnd_bit_sharing
    from Protocol9_BitwiseLessThan import protocol_9_bitwise_less_than

Shares = List[Tuple[int, int]]


def protocol_7_lsb(
    x_shares: Shares,
    n: int,
    t: int,
    p: int,
    print_: bool = False,
) -> Shares:
    """Return a (t,n)-sharing of the LSB of the secret encoded by x_shares.

    Parameters
    ----------
    x_shares : Shares
        A (t,n)-sharing of x ∈ Z_p.
    n, t, p  : int
        MPC parameters.  Requires n ≥ 2t-1 (for the sub-protocols).
    print_   : bool
        Verbose debug output.

    Returns
    -------
    Shares
        A (t,n)-sharing [[x_0]] where x_0 = x mod 2 = LSB(x).
    """
    s = math.ceil(math.log2(p)) if p > 2 else 1

    # ------------------------------------------------------------------
    # Lines 1-2: Generate s independent random-bit sharings [[r_i]]
    # ------------------------------------------------------------------
    r_bit_shares: List[Shares] = []
    for i in range(s):                                               # Line 1
        rb = protocol_8_gen_rnd_bit_sharing(n, t, p, print_=False)  # Line 2
        r_bit_shares.append(rb)

    # ------------------------------------------------------------------
    # Lines 3-4: Build [[r]] = Σ_{i=0}^{s-1} 2^i · [[r_i]]  (local)
    # Each party j sums its shares: [[r]]_j = Σ_i 2^i · [[r_i]]_j
    # ------------------------------------------------------------------
    r_shares: Shares = [(j + 1, 0) for j in range(n)]
    for i in range(s):
        power = pow(2, i, p)                                         # 2^i mod p
        scaled: Shares = [(x, (power * y) % p) for x, y in r_bit_shares[i]]
        r_shares = [(x, (yr + ys) % p)
                    for (x, yr), (_, ys) in zip(r_shares, scaled)]  # Line 4

    # ------------------------------------------------------------------
    # Line 5: [[c]] = [[x]] + [[r]]  (local)
    # ------------------------------------------------------------------
    c_shares: Shares = [(x, (yx + yr) % p)
                        for (x, yx), (_, yr) in zip(x_shares, r_shares)]

    # ------------------------------------------------------------------
    # Lines 6-7: Reconstruct c publicly (use t shares, degree t-1 poly)
    # ------------------------------------------------------------------
    c = protocol_2_reconstruct(c_shares[:t], p)                     # Lines 6-7
    if print_:
        print(f"[Protocol7] c (public) = {c}")

    # ------------------------------------------------------------------
    # Lines 8-11: Set [[d_0]] depending on the LSB c_0 of c
    #   c_0 = 0  →  d_0 = r_0           (Line 9)
    #   c_0 = 1  →  d_0 = 1 − r_0       (Line 11)
    # In both cases d_0 = c_0 XOR r_0.
    # ------------------------------------------------------------------
    c_0 = c & 1                                                      # LSB of c
    if c_0 == 0:                                                     # Line 8
        d0_shares: Shares = r_bit_shares[0]                         # Line 9
    else:
        d0_shares = [(x, (1 - y) % p) for x, y in r_bit_shares[0]] # Line 11

    # ------------------------------------------------------------------
    # Line 12: [[e]] = Bitwise_LessThan(c, {[[r_i]]})
    # Computes 1_{c < r}, i.e. whether the mask r exceeds the revealed c.
    # ------------------------------------------------------------------
    e_shares = protocol_9_bitwise_less_than(
        c, r_bit_shares, n, t, p, print_=False
    )                                                                # Line 12

    # ------------------------------------------------------------------
    # Line 13: [[y]] = SecureMult([[e]], [[d_0]])
    # ------------------------------------------------------------------
    y_shares = protocol_4_secure_mult(
        e_shares, d0_shares, n, t, p, print_=False
    )                                                                # Line 13

    # ------------------------------------------------------------------
    # Line 14: [[x_0]] = [[e]] + [[d_0]] − 2[[y]]   (local, XOR formula)
    # ------------------------------------------------------------------
    x0_shares: Shares = [
        (x, (ye + yd - 2 * yy) % p)
        for (x, ye), (_, yd), (_, yy)
        in zip(e_shares, d0_shares, y_shares)
    ]                                                                # Line 14

    if print_:
        x0_val = protocol_2_reconstruct(x0_shares[:t], p)
        print(f"[Protocol7] LSB(x) = {x0_val}")

    return x0_shares


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random
    from Protocol1 import protocol_1_share
    from Protocol2 import protocol_2_reconstruct

    random.seed(3)
    P, N, T = 11, 3, 2

    cases = [(0, 0), (1, 1), (2, 0), (3, 1), (4, 0), (5, 1), (7, 1), (10, 0)]
    all_ok = True
    for secret, expected_lsb in cases:
        sh  = protocol_1_share(secret, N, T, P)
        lsb_sh = protocol_7_lsb(sh, N, T, P, print_=False)
        lsb    = protocol_2_reconstruct(lsb_sh[:T], P)
        status = "✓" if lsb == expected_lsb else "✗"
        print(f"LSB({secret}) = {lsb}  (expected {expected_lsb}) {status}")
        all_ok = all_ok and (lsb == expected_lsb)
    print("Protocol 7 self-test", "PASSED ✓" if all_ok else "FAILED ✗")
