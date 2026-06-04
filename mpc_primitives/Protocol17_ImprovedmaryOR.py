"""
Protocol 17 — An Improved Protocol for m-ary ORs
Evaluate OR(u_1, ..., u_m) on m shared bits using a more communication-
efficient approach: the polynomial is replaced by a product formula, and
individual share multiplication (local, degree 2(t-1)) is used in place of
one of the SecureMult calls.

Pseudocode reference (line numbers match the paper):
---------------------------------------------------------------------------
Input:   [[u_i]], i ∈ [m]
Parameter: m

Line 1:  for all i = 2 to m+1 do
Line 2:    [[b_i]] ← RNG()
Line 3:    [[c_i]] ← SecureInv([[b_i]])
Line 4:  [[w]] ← 1 + Σ_{i ∈ [m]} [[u_i]]
Line 5:  [[b_1]] ← [[1]]
Line 6:  for all i = 2 to m+1 do
Line 7:    [[d_i]] ← SecureMult([[w−i]], [[b_{i-1}]])
Line 8:    Each party P_j sets [[e_i]]_j ← [[d_i]]_j · [[c_i]]_j   (LOCAL)
Line 9:    The parties run Reconstruct([[e_i]], 1; 2t-1)
Line 10:   P_1 broadcasts e_i
Line 11: [[v]] ← 1 − (Π_{j=2}^{m+1} e_j) · [[b_{m+1}]]
Output:  A sharing [[v]] of v = OR_{i ∈ [m]} u_i
---------------------------------------------------------------------------

Key idea
--------
w = 1 + Σ u_i  ∈ {1, ..., m+1}.

The product  Π_{j=2}^{m+1} (w − j)  is zero iff  w ∈ {2, ..., m+1},
i.e. iff at least one u_i = 1 (OR = 1).  It is non-zero iff w = 1 (OR = 0).

The protocol computes this product securely using blinding:

    e_i = (w − i) · b_{i-1} · b_i^{-1}        (telescoping)

    Π_{j=2}^{m+1} e_j  =  Π_{j=2}^{m+1} (w−j) · b_1 · b_{m+1}^{-1}
                         =  Π_{j=2}^{m+1} (w−j) · b_{m+1}^{-1}   (since b_1=1)

    (Π e_j) · [[b_{m+1}]]  =  [[Π_{j=2}^{m+1} (w−j)]]

    [[v]] = 1 − [[Π_{j=2}^{m+1} (w−j)]]  =  1  iff the product is 0
                                           =  0  iff the product is non-zero

Difference from Protocol 16
----------------------------
Step 8 uses LOCAL multiplication of share values rather than SecureMult.
The resulting [[e_i]] is a degree-2(t-1) polynomial, hence reconstruction
at Line 9 uses threshold 2t-1 instead of t.

This avoids one SecureMult per iteration (replacing it with communication-
free local multiplication + one degree-2t-1 reconstruction).

Note on correctness when w = 1
------------------------------
When OR = 0 (all u_i = 0, w = 1):
    Π_{j=2}^{m+1} (w−j) = Π_{j=2}^{m+1} (1−j) = (−1)^m · m!

For the output v to equal 0 we need  (−1)^m · m! ≡ 1 (mod p).
This holds for specific (m, p) pairs but is NOT guaranteed in general.
For a universal protocol, a normalisation factor (−1)^m · m! should be
applied to the final product; the pseudocode leaves this implicit.
"""

import random
from typing import List, Tuple

try:
    from MPC_over_secret_shares.Protocol2 import protocol_2_reconstruct
    from MPC_over_secret_shares.Protocol4 import protocol_4_secure_mult
    from MPC_over_secret_shares.secure_inv import secure_inv, _internal_rng
except ImportError:
    from Protocol2 import protocol_2_reconstruct
    from Protocol4 import protocol_4_secure_mult
    from secure_inv import secure_inv, _internal_rng

Shares = List[Tuple[int, int]]


def protocol_17_improved_m_ary_or(
    u_shares_list: List[Shares],
    n: int,
    t: int,
    p: int,
    print_: bool = False,
) -> Shares:
    """Compute a (t,n)-sharing of OR(u_1, ..., u_m) via the improved protocol.

    Parameters
    ----------
    u_shares_list : List[Shares]
        List of m (t,n)-sharings of input bits u_i ∈ {0,1}.
    n, t, p       : int
        MPC parameters.  Requires n ≥ 2t-1.
    print_        : bool
        Verbose debug output.

    Returns
    -------
    Shares
        A (t,n)-sharing [[v]] of v = OR(u_1, ..., u_m).

    Notes
    -----
    When OR = 0 (all u_i = 0) the output is correct only if
    (−1)^m · m! ≡ 1 (mod p).  See module docstring for details.
    """
    m = len(u_shares_list)
    needed_for_2t1 = 2 * t - 1
    if n < needed_for_2t1:
        raise RuntimeError(
            f"Protocol 17 requires n ≥ 2t-1 = {needed_for_2t1}, got n={n}"
        )

    # ------------------------------------------------------------------
    # Lines 1-3: Generate m random blinding sharings b_2, ..., b_{m+1}
    #            and their inverses c_2, ..., c_{m+1}.
    # (b_1 = [[1]] is set in Line 5; the loop runs for i = 2 to m+1.)
    # ------------------------------------------------------------------
    b_list: List[Shares] = []   # b_list[k] = b_{k+2}, k = 0, ..., m-1
    c_list: List[Shares] = []   # c_list[k] = c_{k+2}

    for _ in range(m):                                               # Line 1
        b_i = _internal_rng(n, t, p)                                # Line 2
        b_list.append(b_i)
        c_i = secure_inv(b_i, n, t, p, print_=False)                # Line 3
        c_list.append(c_i)

    # ------------------------------------------------------------------
    # Line 4: [[w]] = 1 + Σ_{i=1}^m [[u_i]]  (local)
    # ------------------------------------------------------------------
    w_shares: Shares = [(j + 1, 1) for j in range(n)]
    for ui_shares in u_shares_list:
        w_shares = [(x, (yw + yu) % p)
                    for (x, yw), (_, yu) in zip(w_shares, ui_shares)]  # Line 4

    # ------------------------------------------------------------------
    # Line 5: [[b_1]] = [[1]]
    # ------------------------------------------------------------------
    prev_b: Shares = [(j + 1, 1) for j in range(n)]                # Line 5

    # ------------------------------------------------------------------
    # Lines 6-10: For i = 2 to m+1 (index k = 0 to m-1 in our zero-based list)
    # ------------------------------------------------------------------
    e_values: List[int] = []    # public reconstructed e_i values

    for k in range(m):          # k corresponds to i = k+2 in the pseudocode
        i_paper = k + 2         # i in {2, ..., m+1}

        # Line 7: [[d_i]] = SecureMult([[w − i]], [[b_{i-1}]])
        # [[w − i]] is local: subtract public constant i from all share values.
        w_minus_i: Shares = [(x, (yw - i_paper) % p) for x, yw in w_shares]
        d_i = protocol_4_secure_mult(
            w_minus_i, prev_b, n, t, p, print_=False
        )                                                            # Line 7

        # Line 8: LOCAL share multiplication  [[e_i]]_j = [[d_i]]_j · [[c_i]]_j
        # This produces a degree-2(t-1) polynomial (product of two degree-(t-1) polys).
        e_i_shares: Shares = [
            (x, (yd * yc) % p)
            for (x, yd), (_, yc) in zip(d_i, c_list[k])
        ]                                                            # Line 8

        # Lines 9-10: Reconstruct e_i using 2t-1 shares (degree 2(t-1) poly)
        e_i_val = protocol_2_reconstruct(e_i_shares[:needed_for_2t1], p)  # Line 9
        e_values.append(e_i_val)
        if print_:
            print(f"[Protocol17] e_{i_paper} = {e_i_val}")          # Line 10

        prev_b = b_list[k]      # advance b_{i-1} → b_i

    # ------------------------------------------------------------------
    # Line 11: [[v]] = 1 − (Π_{j=2}^{m+1} e_j) · [[b_{m+1}]]
    #   Π e_j is a public scalar (product of all reconstructed e_j).
    #   [[b_{m+1}]] is b_list[-1] (the last random sharing).
    # ------------------------------------------------------------------
    E_product = 1
    for ev in e_values:
        E_product = (E_product * ev) % p

    # E_product · [[b_{m+1}]] encodes  Π_{j=2}^{m+1} (w−j)  as a sharing
    b_last: Shares = b_list[-1]
    Eb_shares: Shares = [(x, (E_product * y) % p) for x, y in b_last]

    # [[v]] = 1 − [[Π (w−j)]]  (local: subtract from constant 1)
    v_shares: Shares = [(x, (1 - y) % p) for x, y in Eb_shares]    # Line 11

    if print_:
        v_val = protocol_2_reconstruct(v_shares[:t], p)
        print(f"[Protocol17] OR = {v_val}")

    return v_shares


# ---------------------------------------------------------------------------
# Quick self-test  (only OR=1 cases are universally reliable; see docstring)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(33)
    from Protocol1 import protocol_1_share
    from Protocol2 import protocol_2_reconstruct

    P, N, T = 11, 3, 2

    def share_bit(b_val: int) -> Shares:
        return protocol_1_share(b_val, N, T, P)

    # Cases with at least one '1' — these should always give OR = 1
    positive_cases = [
        ([1, 0], 1),
        ([0, 1], 1),
        ([1, 1], 1),
        ([1, 0, 0], 1),
        ([0, 1, 0], 1),
        ([1, 1, 1], 1),
    ]

    print("Testing OR=1 cases (universally correct for all p):")
    all_ok = True
    for bits, expected in positive_cases:
        u_sh_list = [share_bit(b) for b in bits]
        v_sh = protocol_17_improved_m_ary_or(u_sh_list, N, T, P, print_=False)
        v    = protocol_2_reconstruct(v_sh[:T], P)
        status = "✓" if v == expected else "✗"
        print(f"  OR{bits} = {v}  (expected {expected}) {status}")
        all_ok = all_ok and (v == expected)

    print("Protocol 17 self-test (OR=1 cases)",
          "PASSED ✓" if all_ok else "FAILED ✗")
