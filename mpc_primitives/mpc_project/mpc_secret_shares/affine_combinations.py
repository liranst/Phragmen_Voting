"""
affine_combinations — Protocol 3.1: Local Affine Combination of Shares.

Computes a (t,n)-sharing of  w = α + β·u + γ·v  from sharings of u and v.
This is a *local* operation — no communication is required.

Reference: "SoK: Secure Computation over Secret Shares", Protocol 3.1.
"""

from typing import List, Tuple

Shares = List[Tuple[int, int]]


def protocol_3_1_affine_combination(
    alpha: int,
    beta: int,
    gamma: int,
    shares_u: Shares,
    shares_v: Shares,
    p: int,
) -> Shares:
    """Return a sharing of  w = α + β·u + γ·v  (local, no communication).

    Parameters
    ----------
    alpha, beta, gamma : int
        Public constants.
    shares_u : Shares
        A (t,n)-sharing of u.
    shares_v : Shares
        A (t,n)-sharing of v.
    p : int
        Prime field modulus.

    Returns
    -------
    Shares
        A (t,n)-sharing of w = α + β·u + γ·v (mod p).
    """
    w_shares: Shares = []
    for (x_u, y_u), (x_v, y_v) in zip(shares_u, shares_v):
        if x_u != x_v:
            raise ValueError(
                f"Participant ID mismatch: {x_u} ≠ {x_v}.  "
                "Both share lists must be ordered by participant ID."
            )
        y_new = (alpha + beta * y_u + gamma * y_v) % p
        w_shares.append((x_u, y_new))
    return w_shares
