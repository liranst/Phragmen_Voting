"""
Protocol 6 — LessThan_Half_P

Compute a sharing of  w = 1_{a < p/2}.

Uses the identity: for an odd prime p and a ∈ Z_p,
    a < p/2  ⟺  2a mod p  is even  ⟺  LSB(2a mod p) = 0.

So  w = 1 − LSB(2[[a]]).

Reference: "SoK: Secure Computation over Secret Shares", Protocol 6.
"""

from typing import List, Tuple

from .protocol7_lsb import protocol_7_lsb

Shares = List[Tuple[int, int]]


def protocol_6_less_than_half_p(
    a_shares: Shares,
    n: int,
    t: int,
    p: int,
    print_: bool = False,
) -> Shares:
    """Return a (t,n)-sharing of  w = 1  if a < p/2, else w = 0.

    Parameters
    ----------
    a_shares : Shares
        A (t,n)-sharing of a ∈ Z_p.
    n, t, p : int
        MPC parameters.
    print_ : bool
        Verbose debug output.

    Returns
    -------
    Shares
        A (t,n)-sharing [[w]] where w = 1_{a < p/2}.
    """
    # Line 1, step i:  2[[a]]  (local scalar multiplication)
    two_a_shares: Shares = [(x, (2 * y) % p) for x, y in a_shares]

    # Line 1, step ii: LSB(2[[a]])  via Protocol 7
    lsb_shares = protocol_7_lsb(two_a_shares, n, t, p, print_=print_)

    # Line 1, step iii:  [[w]] = 1 − LSB  (local)
    w_shares: Shares = [(x, (1 - y) % p) for x, y in lsb_shares]

    if print_:
        from .protocol2 import protocol_2_reconstruct
        print(f"[Protocol6] 1_{{a < p/2}} = {protocol_2_reconstruct(w_shares[:t], p)}")

    return w_shares
