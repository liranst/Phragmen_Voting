"""
Protocol 5 — SecureCompare

Securely compute  w = 1_{u < v}  without revealing u, v, or w.

Uses LessThan_Half_P (Protocol 6) as a building block to determine the
"sign" of each operand and their difference in the signed interpretation
of Z_p.

Reference: "SoK: Secure Computation over Secret Shares", Protocol 5.
"""

from typing import List, Tuple

from .protocol4 import protocol_4_secure_mult
from .protocol6_less_than_half_p import protocol_6_less_than_half_p

Shares = List[Tuple[int, int]]


def protocol_5_secure_compare(
    u_shares: Shares,
    v_shares: Shares,
    n: int,
    t: int,
    p: int,
    print_: bool = False,
) -> Shares:
    """Return a (t,n)-sharing of  w = 1_{u < v}.

    Parameters
    ----------
    u_shares, v_shares : Shares
        (t,n)-sharings of u, v ∈ Z_p.
    n, t, p : int
        MPC parameters.
    print_ : bool
        Verbose debug output.

    Returns
    -------
    Shares
        A (t,n)-sharing [[w]] where w = 1 if u < v, else w = 0.
    """
    # Line 1: [[a]] = 1_{u ≥ p/2}  (u is "negative" in signed view)
    a_shares = protocol_6_less_than_half_p(
        u_shares, n, t, p, print_=False
    )

    # Line 2: [[b]] = 1_{v ≥ p/2}
    b_shares = protocol_6_less_than_half_p(
        v_shares, n, t, p, print_=False
    )

    # Line 3: [[c]] = 1_{(u−v) ≥ p/2};  [[u]]−[[v]] is local
    diff_shares: Shares = [
        (x, (yu - yv) % p) for (x, yu), (_, yv) in zip(u_shares, v_shares)
    ]
    c_shares = protocol_6_less_than_half_p(
        diff_shares, n, t, p, print_=False
    )

    # Line 4: [[e]] = SecureMult([[b]], [[c]])
    e_shares = protocol_4_secure_mult(
        b_shares, c_shares, n, t, p, print_=False
    )

    # Line 5: [[d]] = [[b]] + [[c]] − [[e]]  =  b OR c  (local)
    d_shares: Shares = [
        (x, (yb + yc - ye) % p)
        for (x, yb), (_, yc), (_, ye) in zip(b_shares, c_shares, e_shares)
    ]

    # Line 6: [[w]] = SecureMult([[a]], [[d]] − [[e]])
    de_diff: Shares = [
        (x, (yd - ye) % p) for (x, yd), (_, ye) in zip(d_shares, e_shares)
    ]
    w_shares = protocol_4_secure_mult(
        a_shares, de_diff, n, t, p, print_=False
    )

    # Line 7: [[w]] = [[w]] − [[d]] + 1  (local)
    w_shares = [
        (x, (yw - yd + 1) % p)
        for (x, yw), (_, yd) in zip(w_shares, d_shares)
    ]

    if print_:
        from .protocol2 import protocol_2_reconstruct
        print(f"[Protocol5] 1_{{u < v}} = {protocol_2_reconstruct(w_shares[:t], p)}")

    return w_shares
