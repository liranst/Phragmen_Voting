"""
Protocol 2 — Reconstruct (Lagrange Interpolation)

Recover the secret S = f(0) from a set of at least *t* Shamir shares using
Lagrange interpolation over Z_p.

Reference: "SoK: Secure Computation over Secret Shares", Protocol 2.
"""

from typing import List, Tuple

Shares = List[Tuple[int, int]]


def protocol_2_reconstruct(
    shares: Shares,
    p: int,
    print_: bool = False,
) -> int:
    """Reconstruct the secret from *shares* via Lagrange interpolation.

    Parameters
    ----------
    shares : Shares
        At least *t* share tuples ``(x_i, y_i)``.
    p : int
        Prime field modulus.
    print_ : bool
        Verbose output for debugging.

    Returns
    -------
    int
        The reconstructed secret S = f(0) in ``[0, p)``.
    """
    secret = 0

    for i in range(len(shares)):
        x_i, y_i = shares[i]
        numerator   = 1
        denominator = 1

        for j in range(len(shares)):
            if i != j:
                x_j, _ = shares[j]
                numerator   = (numerator   * (-x_j))      % p
                denominator = (denominator * (x_i - x_j)) % p

        denominator_inv  = pow(denominator, p - 2, p)  # Fermat's little theorem
        lagrange_weight  = (numerator * denominator_inv) % p
        term             = (y_i * lagrange_weight) % p

        if print_:
            print(f"  L_{i+1}(0) = {lagrange_weight},  term = {term}")

        secret = (secret + term) % p

    return secret
