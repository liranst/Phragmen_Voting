"""
Protocol 8 — GenRndBitSharing
Generating a (t,n)-sharing of a uniformly random bit b ∈ {0, 1}.

Pseudocode reference (line numbers match the paper):
---------------------------------------------------------------------------
Line 1:  for all i ∈ [n] do
Line 2:    P_i generates a random r_i ∈ Z_p
Line 3:    P_i runs Share(r_i, i; t)
Line 4:  for all j ∈ [n] do
Line 5:    P_j sets [[r]]_j = Σ_{i=1}^{n} [[r_i]]_j
Line 6:    P_j sets [[r²]]_j ← [[r]]_j * [[r]]_j
Line 7:  The parties run Reconstruct([[r²]], 1; 2t-1)
Line 8:  P_1 broadcasts r²
Line 9:  if r² = 0 then
Line 10:   go to Line 1
Line 11: Compute a square root r' of r²
Line 12: Compute (r')^{-1}
Line 13: [[b]] ← ((p+1)/2) * ((r')^{-1} * [[r]] + 1)
Output:  A (t,n)-sharing of a secret random bit b ∈ {0,1}
---------------------------------------------------------------------------

Mathematical intuition
----------------------
After step 5, [[r]] encodes r = Σ r_i ∈ Z_p (random, unknown to any subset
of fewer than t parties).

Step 6 forms evaluations of the degree-2(t-1) polynomial (Σ f_i(x))² whose
constant term is r².  Reconstruction with 2t-1 shares recovers r² without
revealing r's sign.

Step 11 picks one of the two square roots r' ∈ {r, p−r} consistently.
Step 12-13 then exploit:

    ((p+1)/2) · (r'⁻¹ · r + 1) =
        ((p+1)/2) · 2  = p+1 ≡ 1  (mod p)   if  r' = r
        ((p+1)/2) · 0  = 0                   if  r' = p−r = −r

The bit is 0 or 1 with equal probability because r is symmetric around 0
in Z_p*.

Requirement: n ≥ 2t−1  (so that the degree-2(t-1) polynomial can be
             reconstructed from n evaluations).
"""

import random
import math
from typing import List, Tuple

try:
    from MPC_over_secret_shares.Protocol1 import protocol_1_share
    from MPC_over_secret_shares.Protocol2 import protocol_2_reconstruct
except ImportError:
    from Protocol1 import protocol_1_share
    from Protocol2 import protocol_2_reconstruct

Shares = List[Tuple[int, int]]


# ---------------------------------------------------------------------------
# Helper — modular square root (Tonelli-Shanks)
# ---------------------------------------------------------------------------

def _modular_sqrt(a: int, p: int) -> int:
    """Return one square root of a modulo p (prime), or raise ValueError.

    Uses the Tonelli-Shanks algorithm which works for any odd prime p.
    For p ≡ 3 (mod 4) the faster formula r = a^{(p+1)/4} is used.
    """
    a = a % p
    if a == 0:
        return 0
    if pow(a, (p - 1) // 2, p) != 1:
        raise ValueError(f"{a} is not a quadratic residue mod {p}")

    # Fast path: p ≡ 3 (mod 4)
    if p % 4 == 3:
        return pow(a, (p + 1) // 4, p)

    # Tonelli-Shanks: write p-1 = Q * 2^S with Q odd
    Q, S = p - 1, 0
    while Q % 2 == 0:
        Q //= 2
        S += 1

    # Find a quadratic non-residue z
    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1

    M = S
    c = pow(z, Q, p)
    t = pow(a, Q, p)
    R = pow(a, (Q + 1) // 2, p)

    while True:
        if t == 1:
            return R
        # Find least i such that t^{2^i} ≡ 1
        i, temp = 1, (t * t) % p
        while temp != 1:
            temp = (temp * temp) % p
            i += 1
        b = pow(c, 1 << (M - i - 1), p)
        M = i
        c = (b * b) % p
        t = (t * c) % p
        R = (R * b) % p


# ---------------------------------------------------------------------------
# Protocol 8
# ---------------------------------------------------------------------------

def protocol_8_gen_rnd_bit_sharing(n: int, t: int, p: int,
                                   print_: bool = False) -> Shares:
    """Generate a (t,n)-sharing of a uniformly random bit b ∈ {0, 1}.

    Parameters
    ----------
    n, t, p : int
        n — number of parties, t — threshold, p — prime field modulus.
        Requires n ≥ 2t−1.
    print_  : bool
        Verbose debug output when True.

    Returns
    -------
    Shares
        A (t,n)-sharing [[b]] where the secret b ∈ {0, 1} uniformly.
    """
    while True:           # Retry loop (Lines 9-10: restart if r² = 0)

        # ------------------------------------------------------------------
        # Lines 1-5: Every party P_i contributes a random r_i and shares it.
        #            Party P_j accumulates its column: [[r]]_j = Σ_i [[r_i]]_j
        # ------------------------------------------------------------------
        all_shares: List[Shares] = []
        for i in range(n):                                          # Line 1
            r_i = random.randint(1, p - 1)                         # Line 2 (non-zero)
            all_shares.append(protocol_1_share(r_i, n, t, p))      # Line 3

        # Line 4-5: each party sums its column across all sharings
        r_shares: Shares = []
        for j in range(n):
            x_j   = all_shares[0][j][0]                            # participant ID
            total = sum(all_shares[i][j][1] for i in range(n)) % p
            r_shares.append((x_j, total))

        # ------------------------------------------------------------------
        # Line 6: Local squaring — produces evaluations of the degree-2(t-1)
        #         polynomial g(x) = f(x)² where f encodes r.
        # ------------------------------------------------------------------
        r2_shares: Shares = [(x, (y * y) % p) for x, y in r_shares]  # Line 6

        # ------------------------------------------------------------------
        # Lines 7-8: Reconstruct r² using 2t-1 shares (degree 2(t-1) poly).
        # ------------------------------------------------------------------
        needed = 2 * t - 1                                          # Line 7
        if len(r2_shares) < needed:
            raise RuntimeError(
                f"Need n ≥ 2t-1 = {needed} parties for GenRndBitSharing, "
                f"but only n = {n} are available."
            )
        r2 = protocol_2_reconstruct(r2_shares[:needed], p)          # Lines 7-8
        if print_:
            print(f"[Protocol8] r² = {r2}")

        # ------------------------------------------------------------------
        # Lines 9-10: If r² = 0, the random sum r collapsed to 0; retry.
        # ------------------------------------------------------------------
        if r2 == 0:                                                  # Line 9
            if print_:
                print("[Protocol8] r² = 0 → retrying")
            continue                                                 # Line 10

        # ------------------------------------------------------------------
        # Line 11: Compute a square root r' of r²  (r' ∈ {r, p-r})
        # ------------------------------------------------------------------
        r_prime = _modular_sqrt(r2, p)                              # Line 11

        # ------------------------------------------------------------------
        # Line 12: Compute (r')^{-1} mod p
        # ------------------------------------------------------------------
        r_prime_inv = pow(r_prime, p - 2, p)                       # Line 12

        # ------------------------------------------------------------------
        # Line 13: [[b]] = ((p+1)/2) · (r'⁻¹ · [[r]] + 1)
        #   Step A: r'⁻¹ · [[r]]  — local scalar multiplication
        #   Step B: + 1            — add public constant 1 to all shares
        #   Step C: · (p+1)/2     — local scalar multiplication
        # ------------------------------------------------------------------
        # Step A
        scaled: Shares = [(x, (r_prime_inv * y) % p) for x, y in r_shares]
        # Step B: add constant 1 to all share values
        plus_one: Shares = [(x, (y + 1) % p) for x, y in scaled]
        # Step C: multiply by (p+1)/2 (the modular inverse of 2)
        half = (p + 1) // 2                                        # valid since p odd prime
        b_shares: Shares = [(x, (half * y) % p) for x, y in plus_one]  # Line 13

        if print_:
            reconstructed = protocol_2_reconstruct(b_shares[:t], p)
            print(f"[Protocol8] Generated random bit b = {reconstructed}")

        return b_shares


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(0)
    P, N, T = 11, 3, 2
    results = []
    trials = 20
    print(f"Generating {trials} random bits over Z_{P} with n={N}, t={T}:")
    for _ in range(trials):
        sh = protocol_8_gen_rnd_bit_sharing(N, T, P, print_=False)
        b  = protocol_2_reconstruct(sh[:T], P)
        assert b in (0, 1), f"Got b={b}, expected 0 or 1"
        results.append(b)
    print(f"Bits: {results}")
    print(f"Sum (should be ~{trials//2}): {sum(results)}")
    print("Protocol 8 self-test PASSED ✓")
