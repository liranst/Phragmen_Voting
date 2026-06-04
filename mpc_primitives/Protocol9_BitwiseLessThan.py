"""
Protocol 9 — Bitwise_LessThan
Computing a sharing of 1_{a < b} where a is a *public* integer and b is
given through (t,n)-sharings of its individual bits.

Pseudocode reference (line numbers match the paper):
---------------------------------------------------------------------------
Input:  a ∈ Z_p  and  {[[b_i]]}_{0 ≤ i ≤ s-1}
        where b_i ∈ {0,1} and b = Σ_{i=0}^{s-1} 2^i b_i ∈ Z_p
Parameter: s = ⌈log₂(p)⌉

Line 1:  for all 0 ≤ i ≤ s-1 do
Line 2:    if a_i = 0 then
Line 3:      [[c_i]] ← [[b_i]]
Line 4:    else
Line 5:      [[c_i]] ← 1 - [[b_i]]
Line 6:  [[d_{s-1}]] ← [[c_{s-1}]]
Line 7:  [[e_{s-1}]] ← [[c_{s-1}]]
Line 8:  for all i = s-2 down to 0 do
Line 9:    [[d_i]] ← [[d_{i+1}]] + [[c_i]] - SecureMult([[d_{i+1}]], [[c_i]])
Line 10:   [[e_i]] ← [[d_i]] - [[d_{i+1}]]
Line 11: [[ê]]  ← Σ_{i=0}^{s-1} [[e_i]]
Line 12: [[g]]  ← 1 - Σ_{i=0}^{s-1} [[e_i]] · a_i
Line 13: [[f]]  ← SecureMult([[ê]], [[g]])
Output:  A (t,n)-sharing [[f]] of f := 1_{a < b}
---------------------------------------------------------------------------

Intuition
---------
c_i = b_i XOR a_i  (the bits agree ⟺ c_i = 0).

d_i = OR(c_{s-1}, ..., c_i)  computed via the formula
      OR(x, y) = x + y − x·y  (which equals the logical OR when x,y ∈ {0,1}).

e_i = d_i − d_{i+1}  ∈ {0,1}: a "pulse" that is 1 exactly at the most
      significant position where a and b differ.

ê = Σ e_i is 1 if they differ anywhere (0 if a = b).

g = 1 − Σ (a_i · e_i):
    - At the decisive bit i*, e_{i*} = 1.
    - If a_{i*} = 0 (so b_{i*} = 1): g = 1, meaning b > a at that bit.
    - If a_{i*} = 1 (so b_{i*} = 0): g = 0, meaning a > b at that bit.

f = ê · g = 1  iff  a ≠ b  AND  b beats a at the MSB where they differ
          = 1_{a < b}.
"""

import math
from typing import List, Tuple

try:
    from MPC_over_secret_shares.Protocol4 import protocol_4_secure_mult
    from MPC_over_secret_shares.Protocol2 import protocol_2_reconstruct
except ImportError:
    from Protocol4 import protocol_4_secure_mult
    from Protocol2 import protocol_2_reconstruct

Shares = List[Tuple[int, int]]


def protocol_9_bitwise_less_than(
    a: int,
    b_bit_shares: List[Shares],
    n: int,
    t: int,
    p: int,
    print_: bool = False,
) -> Shares:
    """Return a (t,n)-sharing of 1_{a < b}.

    Parameters
    ----------
    a            : int
        The *public* comparand (integer in Z_p).
    b_bit_shares : List[Shares]
        List of length s where b_bit_shares[i] = [[b_i]], a (t,n)-sharing
        of the i-th bit of b (LSB = index 0, MSB = index s-1).
    n, t, p      : int
        MPC parameters.
    print_       : bool
        Verbose debug output.

    Returns
    -------
    Shares
        A (t,n)-sharing [[f]] where f = 1 if a < b, else 0.
    """
    s = len(b_bit_shares)

    # Extract bits of the public value a (LSB first)
    a_bits = [(a >> i) & 1 for i in range(s)]   # a_bits[i] = i-th bit of a

    if print_:
        print(f"[Protocol9] a = {a},  a_bits = {a_bits}")

    # ------------------------------------------------------------------
    # Lines 1-5: c_i ← b_i  if a_i = 0,  else  c_i ← 1 − b_i
    # ------------------------------------------------------------------
    c_shares: List[Shares] = []
    for i in range(s):
        if a_bits[i] == 0:                                          # Line 2
            c_shares.append(b_bit_shares[i])                       # Line 3
        else:
            # 1 − [[b_i]]: add public constant 1 and negate locally
            neg_b = [(x, (-y) % p) for x, y in b_bit_shares[i]]
            c_i   = [(x, (1 + y) % p) for x, y in neg_b]          # Line 5
            c_shares.append(c_i)

    # ------------------------------------------------------------------
    # Lines 6-7: initialise d and e at the MSB position
    # ------------------------------------------------------------------
    d_shares: List[Shares] = [None] * s                             # Line 6
    e_shares: List[Shares] = [None] * s                             # Line 7
    d_shares[s - 1] = c_shares[s - 1]
    e_shares[s - 1] = c_shares[s - 1]

    # ------------------------------------------------------------------
    # Lines 8-10: sweep from bit s-2 down to 0
    # ------------------------------------------------------------------
    for i in range(s - 2, -1, -1):                                  # Line 8
        # Line 9: [[d_i]] = [[d_{i+1}]] + [[c_i]] − SecureMult([[d_{i+1}]], [[c_i]])
        mult = protocol_4_secure_mult(
            d_shares[i + 1], c_shares[i], n, t, p, print_=False
        )
        d_i: Shares = [
            (x, (yd + yc - ym) % p)
            for (x, yd), (_, yc), (_, ym)
            in zip(d_shares[i + 1], c_shares[i], mult)
        ]
        d_shares[i] = d_i                                           # Line 9

        # Line 10: [[e_i]] = [[d_i]] − [[d_{i+1}]]
        e_i: Shares = [
            (x, (yd - yd1) % p)
            for (x, yd), (_, yd1)
            in zip(d_i, d_shares[i + 1])
        ]
        e_shares[i] = e_i                                           # Line 10

    # ------------------------------------------------------------------
    # Line 11: [[ê]] = Σ_{i=0}^{s-1} [[e_i]]  (local summation)
    # ------------------------------------------------------------------
    hat_e: Shares = [(j + 1, 0) for j in range(n)]
    for i in range(s):
        hat_e = [(x, (he + ye) % p)
                 for (x, he), (_, ye) in zip(hat_e, e_shares[i])]  # Line 11

    # ------------------------------------------------------------------
    # Line 12: [[g]] = 1 − Σ_{i: a_i=1} [[e_i]]  (local)
    # Only the terms where a_i = 1 contribute to the sum.
    # ------------------------------------------------------------------
    sum_e_a: Shares = [(j + 1, 0) for j in range(n)]
    for i in range(s):
        if a_bits[i] == 1:
            sum_e_a = [(x, (sv + ye) % p)
                       for (x, sv), (_, ye) in zip(sum_e_a, e_shares[i])]
    g: Shares = [(x, (1 - sg) % p) for x, sg in sum_e_a]           # Line 12

    if print_:
        hat_e_val = protocol_2_reconstruct(hat_e[:t], p)
        g_val     = protocol_2_reconstruct(g[:t], p)
        print(f"[Protocol9] ê = {hat_e_val},  g = {g_val}")

    # ------------------------------------------------------------------
    # Line 13: [[f]] = SecureMult([[ê]], [[g]])
    # ------------------------------------------------------------------
    f_shares = protocol_4_secure_mult(hat_e, g, n, t, p, print_=False)  # Line 13

    return f_shares


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random
    from Protocol1 import protocol_1_share
    from Protocol2 import protocol_2_reconstruct

    random.seed(7)
    P, N, T = 11, 3, 2
    S = math.ceil(math.log2(P))   # 4 bits for p=11

    def share_bit(b_val: int) -> Shares:
        return protocol_1_share(b_val, N, T, P)

    # Test: a=3, b=7  → 3 < 7 → expect 1
    a_val, b_val = 3, 7
    b_bits_plain = [(b_val >> i) & 1 for i in range(S)]
    b_bit_sh = [share_bit(bi) for bi in b_bits_plain]
    result = protocol_9_bitwise_less_than(a_val, b_bit_sh, N, T, P, print_=True)
    f = protocol_2_reconstruct(result[:T], P)
    print(f"1_{{{a_val} < {b_val}}} = {f}  (expected 1)")
    assert f == 1

    # Test: a=9, b=3  → 9 < 3 is False → expect 0
    a_val, b_val = 9, 3
    b_bits_plain = [(b_val >> i) & 1 for i in range(S)]
    b_bit_sh = [share_bit(bi) for bi in b_bits_plain]
    result = protocol_9_bitwise_less_than(a_val, b_bit_sh, N, T, P, print_=True)
    f = protocol_2_reconstruct(result[:T], P)
    print(f"1_{{{a_val} < {b_val}}} = {f}  (expected 0)")
    assert f == 0

    print("Protocol 9 self-test PASSED ✓")
