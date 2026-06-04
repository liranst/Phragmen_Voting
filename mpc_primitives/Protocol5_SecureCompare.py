"""
Protocol 5 — SecureCompare
Secure comparison of two shared secrets u and v.

Pseudocode reference (line numbers match the paper):
---------------------------------------------------------------------------
Input:  [[u]] and [[v]] — (t,n)-sharings of u, v ∈ Z_p

Line 1: [[a]] ← LessThan_Half_P([[u]])
Line 2: [[b]] ← LessThan_Half_P([[v]])
Line 3: [[c]] ← LessThan_Half_P([[u]] − [[v]])
Line 4: [[e]] ← SecureMult([[b]], [[c]])
Line 5: [[d]] ← [[b]] + [[c]] − [[e]]
Line 6: [[w]] ← SecureMult([[a]], [[d]] − [[e]])
Line 7: [[w]] ← [[w]] − [[d]] + [[1]]

Output: A (t,n)-sharing [[w]] of w = 1_{u < v}
---------------------------------------------------------------------------

Correctness argument (signed-magnitude interpretation in Z_p)
-------------------------------------------------------------
Treat values in Z_p as signed integers in (−p/2, p/2]:
    a = 1_{u ∈ (−p/2, 0)}  =  1_{u > p/2}  (in the unsigned view)
    b = 1_{v ∈ (−p/2, 0)}  =  1_{v > p/2}
    c = 1_{u − v ∈ (−p/2, 0)}  (no modular wrap ⟺ actual u − v < 0)

The four sign combinations (a,b) and how they relate to u<v:
  (0,0): both positive.  u<v iff u−v<0 iff c=1.   →  w = c
  (1,1): both negative.  u<v iff u−v<0 iff c=1.   →  w = c
  (0,1): u≥0, v<0.       u>v always.               →  w = 0
  (1,0): u<0, v≥0.       u<v always.               →  w = 1

Working through the formula for each case confirms w = 1_{u<v}.
"""

from typing import List, Tuple

try:
    from MPC_over_secret_shares.Protocol4 import protocol_4_secure_mult
    from MPC_over_secret_shares.Protocol6_LessThanHalfP import protocol_6_less_than_half_p
except ImportError:
    from Protocol4 import protocol_4_secure_mult
    from Protocol6_LessThanHalfP import protocol_6_less_than_half_p

Shares = List[Tuple[int, int]]


def protocol_5_secure_compare(
    u_shares: Shares,
    v_shares: Shares,
    n: int,
    t: int,
    p: int,
    print_: bool = False,
) -> Shares:
    """Return a (t,n)-sharing of 1_{u < v}.

    Parameters
    ----------
    u_shares, v_shares : Shares
        (t,n)-sharings of the secrets u, v ∈ Z_p.
    n, t, p  : int
        MPC parameters.
    print_   : bool
        Verbose debug output.

    Returns
    -------
    Shares
        A (t,n)-sharing [[w]] where w = 1 if u < v, else w = 0.
    """
    # ------------------------------------------------------------------
    # Line 1: [[a]] = LessThan_Half_P([[u]])
    # a = 1  iff  u ∈ [p/2, p)  (i.e. u is "negative" in signed view)
    # ------------------------------------------------------------------
    a_shares = protocol_6_less_than_half_p(u_shares, n, t, p, print_=False)  # Line 1

    # ------------------------------------------------------------------
    # Line 2: [[b]] = LessThan_Half_P([[v]])
    # ------------------------------------------------------------------
    b_shares = protocol_6_less_than_half_p(v_shares, n, t, p, print_=False)  # Line 2

    # ------------------------------------------------------------------
    # Line 3: [[c]] = LessThan_Half_P([[u]] − [[v]])
    # [[u]] − [[v]] is a local operation (affine subtraction of shares).
    # ------------------------------------------------------------------
    diff_shares: Shares = [(x, (yu - yv) % p)
                           for (x, yu), (_, yv) in zip(u_shares, v_shares)]
    c_shares = protocol_6_less_than_half_p(diff_shares, n, t, p, print_=False)  # Line 3

    # ------------------------------------------------------------------
    # Line 4: [[e]] = SecureMult([[b]], [[c]])
    # ------------------------------------------------------------------
    e_shares = protocol_4_secure_mult(
        b_shares, c_shares, n, t, p, print_=False
    )                                                                # Line 4

    # ------------------------------------------------------------------
    # Line 5: [[d]] = [[b]] + [[c]] − [[e]]   (local)
    # d = b OR c  (since b,c ∈ {0,1} and x OR y = x + y − x·y)
    # ------------------------------------------------------------------
    d_shares: Shares = [
        (x, (yb + yc - ye) % p)
        for (x, yb), (_, yc), (_, ye)
        in zip(b_shares, c_shares, e_shares)
    ]                                                                # Line 5

    # ------------------------------------------------------------------
    # Line 6: [[w]] = SecureMult([[a]], [[d]] − [[e]])
    # [[d]] − [[e]] is local.
    # ------------------------------------------------------------------
    de_diff: Shares = [(x, (yd - ye) % p)
                       for (x, yd), (_, ye) in zip(d_shares, e_shares)]
    w_shares = protocol_4_secure_mult(
        a_shares, de_diff, n, t, p, print_=False
    )                                                                # Line 6

    # ------------------------------------------------------------------
    # Line 7: [[w]] = [[w]] − [[d]] + [[1]]
    # Adding the public constant 1 is local (add to all share values).
    # ------------------------------------------------------------------
    w_shares = [(x, (yw - yd + 1) % p)
                for (x, yw), (_, yd) in zip(w_shares, d_shares)]   # Line 7

    if print_:
        from Protocol2 import protocol_2_reconstruct
        w_val = protocol_2_reconstruct(w_shares[:t], p)
        print(f"[Protocol5] 1_{{u < v}} = {w_val}")

    return w_shares


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random
    from Protocol1 import protocol_1_share
    from Protocol2 import protocol_2_reconstruct

    random.seed(99)
    P, N, T = 11, 3, 2

    test_cases = [
        (2, 5, 1),   # 2 < 5  → 1
        (5, 2, 0),   # 5 < 2  → 0
        (3, 3, 0),   # 3 < 3  → 0 (equal)
        (0, 1, 1),   # 0 < 1  → 1
        (10, 1, 0),  # 10 < 1 → 0 (in Z_11, integers 0..10)
        (1, 10, 1),  # 1 < 10 → 1
    ]

    all_ok = True
    for u, v, expected in test_cases:
        sh_u = protocol_1_share(u, N, T, P)
        sh_v = protocol_1_share(v, N, T, P)
        w_sh = protocol_5_secure_compare(sh_u, sh_v, N, T, P)
        w    = protocol_2_reconstruct(w_sh[:T], P)
        status = "✓" if w == expected else "✗"
        print(f"1_{{{u} < {v}}} = {w}  (expected {expected}) {status}")
        all_ok = all_ok and (w == expected)
    print("Protocol 5 self-test", "PASSED ✓" if all_ok else "FAILED ✗")
