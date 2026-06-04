"""
Protocol 10 — IsZero (Equality to Zero)

Compute a (t,n)-sharing of  w = 1_{u=0}  using Fermat's little theorem:
    u^(p-1) ≡ 1 (mod p)  for u ≠ 0,   u^(p-1) ≡ 0 (mod p)  for u = 0.

The exponentiation is performed via square-and-multiply on the shared
value [[u]], with an MSB-skip optimisation that avoids 2 redundant
SecureMult calls.

Optimisation detail
-------------------
The MSB of p-1 is always 1.  The standard loop would initialise [[z]]←[[1]]
and on the first (MSB) iteration perform:
    square:   SecureMult([[1]], [[1]]) → [[1]]   (no-op)
    multiply: SecureMult([[1]], [[u]]) → [[u]]   (just sets z ← u)
Instead, [[z]] is initialised directly to [[u]] and the loop starts from
bit s-2, saving exactly 2 SecureMult calls.

Reference: "SoK: Secure Computation over Secret Shares", Protocol 10.
"""

from typing import List, Tuple

from .protocol4 import protocol_4_secure_mult

Shares = List[Tuple[int, int]]


def protocol_10_is_zero(
    shares_u: Shares,
    n: int,
    t: int,
    p: int,
    print_: bool = False,
) -> Shares:
    """Return a (t,n)-sharing of  w = 1  if u = 0, else w = 0.

    Parameters
    ----------
    shares_u : Shares
        A (t,n)-sharing of the secret u ∈ Z_p.
    n, t, p : int
        MPC parameters.
    print_ : bool
        Verbose debug output.

    Returns
    -------
    Shares
        A (t,n)-sharing [[w]] where w = 1_{u=0}.
    """
    p_minus_1 = p - 1
    binary_b  = bin(p_minus_1)[2:]      # e.g. p=11 → '1010'

    if print_:
        print(f"[Protocol10] Exponent p-1 = {p_minus_1} = {binary_b}₂")

    # Line 1: [[z]] ← [[u]]  (skip the always-1 MSB)
    shares_z = list(shares_u)
    if print_:
        print(f"[Protocol10] Init [[z]] = [[u]] = {shares_z}")

    # Lines 2-5: square-and-multiply, starting from bit s-2 (index 1)
    for idx, bit in enumerate(binary_b[1:]):
        if print_:
            print(f"[Protocol10] iter {idx+1}: bit={bit}")

        # Line 3: [[z]] ← SecureMult([[z]], [[z]])
        shares_z = protocol_4_secure_mult(
            shares_z, shares_z, n, t, p, print_=False
        )
        if print_:
            print(f"  after square: {shares_z}")

        # Lines 4-5: if bit = 1, [[z]] ← SecureMult([[z]], [[u]])
        if bit == '1':
            shares_z = protocol_4_secure_mult(
                shares_z, shares_u, n, t, p, print_=False
            )
            if print_:
                print(f"  after multiply: {shares_z}")

    # Line 6: [[w]] = [[1]] − [[z]]  (local affine operation)
    # u = 0 → z = 0     → w = 1  ✓
    # u ≠ 0 → z = 1     → w = 0  ✓  (Fermat's little theorem)
    shares_w: Shares = [(i, (1 - shares_z[i - 1][1]) % p)
                        for i in range(1, n + 1)]

    if print_:
        print(f"[Protocol10] [[w]] = {shares_w}")

    return shares_w
