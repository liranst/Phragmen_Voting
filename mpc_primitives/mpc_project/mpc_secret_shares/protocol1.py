"""
Protocol 1 — Share (Shamir Secret Sharing)

The dealer splits a secret S into n shares using a random degree-(t-1)
polynomial over Z_p with f(0) = S.  Any t shares suffice to reconstruct S;
fewer than t shares reveal nothing.

Reference: "SoK: Secure Computation over Secret Shares", Protocol 1.
"""

import random
from typing import List, Tuple

Shares = List[Tuple[int, int]]


def protocol_1_share(
    S: int,
    n: int,
    t: int,
    P: int,
    print_: bool = False,
) -> Shares:
    """Split secret *S* into *n* Shamir shares with threshold *t* over Z_P.

    Parameters
    ----------
    S : int
        The secret value in ``[0, P)``.
    n : int
        Number of shares to produce (one per party).
    t : int
        Reconstruction threshold: any *t* shares recover *S*.
    P : int
        Prime field modulus.
    print_ : bool
        Verbose output for debugging.

    Returns
    -------
    Shares
        List of *n* tuples ``(participant_id, share_value)``,
        with ``participant_id`` ∈ ``{1, ..., n}``.
    """
    # Build a random degree-(t-1) polynomial with f(0) = S
    coefficients = []
    poly_str = f"{S}"
    for j in range(1, t):
        a_j = random.randint(0, P - 1)
        coefficients.append(a_j)
        poly_str += f" + {a_j}x^{j}"

    if print_:
        print(f"Secret S = {S}")
        print(f"Polynomial: f(x) = {poly_str}  (mod {P})")

    def f(x: int) -> int:
        result = S
        for j in range(1, t):
            result += coefficients[j - 1] * (x ** j)
        return result % P

    shares: Shares = []
    for j in range(1, n + 1):
        share_value = f(j)
        shares.append((j, share_value))
        if print_:
            print(f"  Share P{j}: f({j}) = {share_value}")

    return shares
