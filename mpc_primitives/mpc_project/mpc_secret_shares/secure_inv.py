"""
secure_inv — SecureInv: compute a sharing of u⁻¹.

Uses the standard randomisation trick:
    1. Generate random [[r]] via RNG.
    2. s = Reconstruct(SecureMult([[r]], [[u]])).   ← reveals r·u, not u.
    3. s_inv = s⁻¹ mod p.                           ← local cleartext.
    4. [[u⁻¹]] = s_inv · [[r]].                     ← local scalar multiply.

Correctness: s_inv · r = (r·u)⁻¹ · r = u⁻¹.

Also exports ``_internal_rng``, a bug-free RNG used by several protocols
internally (Protocol 3's reference implementation has a double-add issue).
"""

import random
from typing import List, Tuple

from .protocol1 import protocol_1_share
from .protocol2 import protocol_2_reconstruct
from .protocol4 import protocol_4_secure_mult

Shares = List[Tuple[int, int]]


def _internal_rng(n: int, t: int, p: int) -> Shares:
    """Generate a (t,n)-sharing of a uniformly random non-zero element of Z_p*.

    Each party P_i contributes r_i; the shared secret is r = Σ r_i (mod p).
    Retries until r ≠ 0 so the result is always invertible.
    """
    while True:
        all_shares: List[Shares] = []
        for _ in range(n):
            r_i = random.randint(0, p - 1)
            all_shares.append(protocol_1_share(r_i, n, t, p))

        result: Shares = []
        for j in range(n):
            x_j   = all_shares[0][j][0]
            total = sum(all_shares[i][j][1] for i in range(n)) % p
            result.append((x_j, total))

        # Verify r ≠ 0 (probability 1/p of needing a retry)
        secret = protocol_2_reconstruct(result[:t], p)
        if secret != 0:
            return result


def secure_inv(
    shares_u: Shares,
    n: int,
    t: int,
    p: int,
    print_: bool = False,
) -> Shares:
    """Return a (t,n)-sharing of u⁻¹ given a (t,n)-sharing of u.

    Parameters
    ----------
    shares_u : Shares
        Sharing of u ∈ Z_p, u ≠ 0.
    n, t, p : int
        MPC parameters.
    print_ : bool
        Verbose debug output.

    Returns
    -------
    Shares
        A (t,n)-sharing of u⁻¹ mod p.

    Raises
    ------
    ValueError
        If the reconstructed product s = r·u equals 0.
    """
    r_shares  = _internal_rng(n, t, p)
    s_shares  = protocol_4_secure_mult(r_shares, shares_u, n, t, p, print_=False)
    s         = protocol_2_reconstruct(s_shares[:t], p)

    if print_:
        print(f"[SecureInv] s = r·u = {s}")
    if s == 0:
        raise ValueError(
            "SecureInv: s = r·u = 0.  Either u = 0 or the random mask r = 0 "
            "(retry with a fresh call)."
        )

    s_inv = pow(s, p - 2, p)
    return [(x, (s_inv * y) % p) for x, y in r_shares]
