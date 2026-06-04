"""
secure_inv.py — SecureInv: compute a sharing of u^{-1} given a sharing of u.

This sub-protocol is used internally by Protocol 16 and Protocol 17.

Technique (standard randomization trick):
  1. Generate a random non-zero [[r]] via RNG.
  2. s = Reconstruct( SecureMult([[r]], [[u]]) )   ← reveals r*u, not u alone.
  3. s_inv = s^{-1} mod p                           ← local cleartext computation.
  4. [[u^{-1}]] = s_inv * [[r]]                     ← local scalar multiplication.

Correctness: s_inv * r = (r*u)^{-1} * r = u^{-1}.
Security:    s = r*u is masked by the random r; it reveals nothing about u.

Assumption:  u ≠ 0 (mod p).  If s = 0 after reconstruction, the random r
             accidentally hit 0 or u = 0; a ValueError is raised.
"""

import random
from typing import List, Tuple

try:
    from MPC_over_secret_shares.Protocol1 import protocol_1_share
    from MPC_over_secret_shares.Protocol2 import protocol_2_reconstruct
    from MPC_over_secret_shares.Protocol4 import protocol_4_secure_mult
except ImportError:
    from Protocol1 import protocol_1_share
    from Protocol2 import protocol_2_reconstruct
    from Protocol4 import protocol_4_secure_mult

Shares = List[Tuple[int, int]]


def _internal_rng(n: int, t: int, p: int) -> Shares:
    """Generate a (t,n)-sharing of a uniformly random element of Z_p.

    This is a self-contained RNG that avoids the double-addition bug present
    in the existing Protocol3 reference implementation.

    Each party P_i contributes a secret random r_i; the shared secret is
    r = sum_i r_i (mod p).
    """
    all_shares: List[Shares] = []
    for _ in range(n):
        r_i = random.randint(0, p - 1)
        all_shares.append(protocol_1_share(r_i, n, t, p))

    result: Shares = []
    for j in range(n):
        x_j = all_shares[0][j][0]          # participant ID = j+1
        total = sum(all_shares[i][j][1] for i in range(n)) % p
        result.append((x_j, total))
    return result


def secure_inv(shares_u: Shares, n: int, t: int, p: int,
               print_: bool = False) -> Shares:
    """Return a (t,n)-sharing of u^{-1} given a (t,n)-sharing of u.

    Parameters
    ----------
    shares_u : Shares
        A (t,n)-sharing of the secret u ∈ Z_p, u ≠ 0.
    n, t, p  : int
        Standard MPC parameters (number of parties, threshold, prime field).
    print_   : bool
        If True, print intermediate values for debugging.

    Returns
    -------
    Shares
        A (t,n)-sharing of u^{-1} mod p.

    Raises
    ------
    ValueError
        If the reconstructed product s = r*u equals 0, which indicates
        u = 0 or (with negligible probability) the random mask r = 0.
    """
    # Step 1 — generate random mask [[r]]
    r_shares = _internal_rng(n, t, p)

    # Step 2 — compute [[s]] = [[r]] * [[u]]  (requires interaction)
    s_shares = protocol_4_secure_mult(r_shares, shares_u, n, t, p,
                                      print_=False)

    # Step 3 — reconstruct s = r * u  (becomes public to all parties)
    s = protocol_2_reconstruct(s_shares[:t], p)
    if print_:
        print(f"[SecureInv] s = r*u = {s}")

    if s == 0:
        raise ValueError(
            "SecureInv: reconstructed product s = 0. "
            "Either u = 0 (not invertible) or the random mask r = 0 "
            "(retry with a fresh call)."
        )

    # Step 4 — s^{-1} is now public; use it to unmask [[r]]
    s_inv = pow(s, p - 2, p)                         # Fermat's little theorem
    inv_shares = [(x, (s_inv * y) % p) for x, y in r_shares]

    # Correctness check (optional):
    #   s_inv * r = (r*u)^{-1} * r = u^{-1}  ✓
    if print_:
        from Protocol2 import protocol_2_reconstruct as _rec
        recovered = _rec(inv_shares[:t], p)
        print(f"[SecureInv] u^{{-1}} (reconstructed) = {recovered}")

    return inv_shares


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)

    P, N, T = 11, 3, 2
    secret_u = 3          # 3^{-1} mod 11 = 4  (since 3*4 = 12 ≡ 1)
    expected  = pow(secret_u, P - 2, P)
    print(f"u = {secret_u},  expected u^{{-1}} mod {P} = {expected}")

    shares_u = protocol_1_share(secret_u, N, T, P)
    inv_sh   = secure_inv(shares_u, N, T, P, print_=True)

    recovered = protocol_2_reconstruct(inv_sh[:T], P)
    print(f"Recovered u^{{-1}} = {recovered}")
    assert recovered == expected, "SecureInv self-test FAILED"
    print("SecureInv self-test PASSED ✓")
