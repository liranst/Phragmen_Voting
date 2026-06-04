"""
Protocol 17 — Improved m-ary OR

An alternative to Protocol 16 that avoids the polynomial coefficient
computation.  Each iteration uses a *local* (non-interactive) share
multiplication followed by a degree-2(t-1) reconstruction, saving one
SecureMult per round compared to Protocol 16.

Note: the OR=0 case is correct only when (−1)^m · m! ≡ 1 (mod p).
See module docstring in the standalone file for a full derivation.

Reference: "SoK: Secure Computation over Secret Shares", Protocol 17.
"""

from typing import List, Tuple

from .protocol2 import protocol_2_reconstruct
from .protocol4 import protocol_4_secure_mult
from .secure_inv import secure_inv, _internal_rng

Shares = List[Tuple[int, int]]


def protocol_17_improved_m_ary_or(
    u_shares_list: List[Shares],
    n: int,
    t: int,
    p: int,
    print_: bool = False,
) -> Shares:
    """Return a (t,n)-sharing of  OR(u_1, …, u_m).

    Parameters
    ----------
    u_shares_list : list of Shares
        m (t,n)-sharings of input bits u_i ∈ {0, 1}.
    n, t, p : int
        MPC parameters.  Requires n ≥ 2t-1.
    print_ : bool
        Verbose debug output.

    Notes
    -----
    The OR=1 case is always correct.  For OR=0 correctness requires
    (−1)^m · m! ≡ 1 (mod p); verify for your specific (m, p) choice.
    """
    m = len(u_shares_list)
    needed = 2 * t - 1
    if n < needed:
        raise RuntimeError(
            f"Protocol 17 requires n ≥ 2t-1 = {needed}, got n = {n}"
        )

    # Lines 1-3: b_i = RNG(), c_i = b_i^{-1}  for i = 2, …, m+1
    b_list: List[Shares] = []
    c_list: List[Shares] = []
    for _ in range(m):
        b_i = _internal_rng(n, t, p)
        b_list.append(b_i)
        c_list.append(secure_inv(b_i, n, t, p, print_=False))

    # Line 4: [[w]] = 1 + Σ [[u_i]]  (local)
    w_shares: Shares = [(j + 1, 1) for j in range(n)]
    for ui in u_shares_list:
        w_shares = [(x, (yw + yu) % p) for (x, yw), (_, yu) in zip(w_shares, ui)]

    # Line 5: [[b_1]] = [[1]]
    prev_b: Shares = [(j + 1, 1) for j in range(n)]
    e_values: List[int] = []

    # Lines 6-10: for i = 2 to m+1
    for k in range(m):
        i_paper = k + 2                                   # i in {2, …, m+1}

        # Line 7: [[d_i]] = SecureMult([[w−i]], [[b_{i-1}]])
        w_minus_i = [(x, (yw - i_paper) % p) for x, yw in w_shares]
        d_i = protocol_4_secure_mult(w_minus_i, prev_b, n, t, p, print_=False)

        # Line 8: local share product → degree-2(t-1) polynomial
        e_i_sh: Shares = [
            (x, (yd * yc) % p) for (x, yd), (_, yc) in zip(d_i, c_list[k])
        ]

        # Lines 9-10: reconstruct with 2t-1 shares
        e_i = protocol_2_reconstruct(e_i_sh[:needed], p)
        e_values.append(e_i)
        if print_:
            print(f"[P17] e_{i_paper} = {e_i}")

        prev_b = b_list[k]

    # Line 11: [[v]] = 1 − (Π e_j) · [[b_{m+1}]]
    E = 1
    for ev in e_values:
        E = (E * ev) % p

    b_last = b_list[-1]
    eb     = [(x, (E * y) % p) for x, y in b_last]
    return [(x, (1 - y) % p) for x, y in eb]
