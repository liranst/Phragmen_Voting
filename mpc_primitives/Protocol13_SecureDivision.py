"""
Protocol 13 — Secure Binary Long Division
Computing sharings of quotient q = ⌊u/v⌋ and remainder r = u mod v,
given sharings of u, v, and the individual bit-sharings of u.

Pseudocode reference (line numbers match the paper):
---------------------------------------------------------------------------
Input:  [[u]], [[v]], v ∈ (0,p), and [[u_j]], j ∈ {0,1,...,s-1}
        such that u = (u_{s-1}, ..., u_0)_2  (u_{s-1} is the MSB)

Line 1: [[q]] ← [[0]]
Line 2: [[r]] ← [[0]]
Line 3: for all j = s-1 down to 0 do
Line 4:   [[r]] ← 2[[r]] + [[u_j]]
Line 5:   [[q_j]] ← [[1]] − SecureCompare([[r]], [[v]])
Line 6:   [[r]] ← [[r]] − SecureMult([[q_j]], [[v]])
Line 7:   [[q]] ← 2[[q]] + [[q_j]]
Output: A sharing of q = ⌊u/v⌋ and of r = u mod v,
        as well as sharings of q's bits {[[q_j]]}
---------------------------------------------------------------------------

Intuition (mirrors cleartext long division)
-------------------------------------------
We process bits of u from MSB to LSB, maintaining a running "partial
remainder" [[r]].

At each step j:
  • Shift r left by 1 bit and append the next bit of u:  r ← 2r + u_j
  • If r ≥ v, the quotient bit q_j = 1; subtract v from r.
  • If r < v, the quotient bit q_j = 0; r is unchanged.

SecureCompare([[r]], [[v]]) = 1 if r < v.
So  q_j = 1 − 1_{r < v} = 1_{r ≥ v}.

The conditional subtraction r ← r − q_j · v is implemented with
SecureMult to avoid revealing which branch is taken.

Note on field arithmetic
------------------------
All values stay within [0, p) throughout the algorithm when
0 ≤ u < p and 0 < v < p, so reduction mod p is valid.
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
    """Compute sharings of ⌊u/v⌋ and u mod v via secure binary long division.

    Parameters
    ----------
    u_shares      : Shares
        A (t,n)-sharing of u ∈ Z_p.
    v_shares      : Shares
        A (t,n)-sharing of v ∈ (0, p).
    u_bit_shares  : List[Shares]
        List of length s where u_bit_shares[j] = [[u_j]], a (t,n)-sharing
        of the j-th bit of u (LSB = index 0, MSB = index s-1).
    n, t, p       : int
        MPC parameters.
    s             : int, optional
        Number of bits.  Defaults to ⌈log₂(p)⌉.
    print_        : bool
        Verbose debug output.

    Returns
    -------
    q_shares : Shares
        A (t,n)-sharing of q = ⌊u/v⌋.
    r_shares : Shares
        A (t,n)-sharing of r = u mod v.
    q_bit_shares : List[Tuple[int, Shares]]
        List of (bit_index, shares) pairs for the bits of q.
        q_bit_shares[k] = (j, [[q_j]]) where j is the bit position.
    """
    if s is None:
        s = math.ceil(math.log2(p)) if p > 2 else 1

    # ------------------------------------------------------------------
    # Lines 1-2: Initialise [[q]] = [[0]] and [[r]] = [[0]]
    # ------------------------------------------------------------------
    q_shares: Shares = [(j + 1, 0) for j in range(n)]              # Line 1
    r_shares: Shares = [(j + 1, 0) for j in range(n)]              # Line 2

    q_bit_shares: List[Tuple[int, Shares]] = []

    # ------------------------------------------------------------------
    # Line 3: Process each bit of u from MSB (j = s-1) down to LSB (j = 0)
    # ------------------------------------------------------------------
    for j in range(s - 1, -1, -1):                                  # Line 3

        # Line 4: [[r]] ← 2[[r]] + [[u_j]]  (local)
        # Scale every share by 2, then add the j-th bit share.
        r_shares = [
            (x, (2 * yr + yu) % p)
            for (x, yr), (_, yu) in zip(r_shares, u_bit_shares[j])
        ]                                                            # Line 4
        if print_:
            from Protocol2 import protocol_2_reconstruct
            r_val = protocol_2_reconstruct(r_shares[:t], p)
            print(f"[Protocol13] j={j}: r (after shift+append) = {r_val}")

        # Line 5: [[q_j]] = [[1]] − SecureCompare([[r]], [[v]])
        # SecureCompare returns 1 if r < v, so 1 − SecureCompare = 1 iff r ≥ v.
        cmp_shares = protocol_5_secure_compare(
            r_shares, v_shares, n, t, p, print_=False
        )
        q_j_shares: Shares = [(x, (1 - yc) % p) for x, yc in cmp_shares]  # Line 5
        q_bit_shares.append((j, q_j_shares))

        if print_:
            from Protocol2 import protocol_2_reconstruct
            qj_val = protocol_2_reconstruct(q_j_shares[:t], p)
            print(f"[Protocol13] j={j}: q_j = {qj_val}")

        # Line 6: [[r]] ← [[r]] − SecureMult([[q_j]], [[v]])
        # If q_j = 1: r ← r − v.  If q_j = 0: r ← r − 0 = r.
        qv_shares = protocol_4_secure_mult(
            q_j_shares, v_shares, n, t, p, print_=False
        )
        r_shares = [
            (x, (yr - yqv) % p)
            for (x, yr), (_, yqv) in zip(r_shares, qv_shares)
        ]                                                            # Line 6

        # Line 7: [[q]] ← 2[[q]] + [[q_j]]  (local)
        q_shares = [
            (x, (2 * yq + yqj) % p)
            for (x, yq), (_, yqj) in zip(q_shares, q_j_shares)
        ]                                                            # Line 7

    if print_:
        from Protocol2 import protocol_2_reconstruct
        q_val = protocol_2_reconstruct(q_shares[:t], p)
        r_val = protocol_2_reconstruct(r_shares[:t], p)
        print(f"[Protocol13] Final: q = {q_val},  r = {r_val}")

    return q_shares, r_shares, q_bit_shares


# ---------------------------------------------------------------------------
# Quick self-test (small field, easy to verify)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random
    from Protocol1 import protocol_1_share
    from Protocol2 import protocol_2_reconstruct

    random.seed(77)
    P, N, T = 11, 3, 2
    S = math.ceil(math.log2(P))   # 4 bits

    def share_bit(b: int) -> Shares:
        return protocol_1_share(b, N, T, P)

    cases = [(7, 3), (8, 2), (6, 4), (10, 5)]
    all_ok = True
    for u, v in cases:
        u_bits = [(u >> i) & 1 for i in range(S)]
        u_bit_sh = [share_bit(bi) for bi in u_bits]

        sh_u = protocol_1_share(u, N, T, P)
        sh_v = protocol_1_share(v, N, T, P)

        q_sh, r_sh, _ = protocol_13_secure_division(
            sh_u, sh_v, u_bit_sh, N, T, P, s=S, print_=False
        )
        q = protocol_2_reconstruct(q_sh[:T], P)
        r = protocol_2_reconstruct(r_sh[:T], P)

        exp_q, exp_r = divmod(u, v)
        status = "✓" if (q == exp_q and r == exp_r) else "✗"
        print(f"divmod({u}, {v}) = ({q}, {r})  expected ({exp_q}, {exp_r}) {status}")
        all_ok = all_ok and (q == exp_q) and (r == exp_r)

    print("Protocol 13 self-test", "PASSED ✓" if all_ok else "FAILED ✗")
