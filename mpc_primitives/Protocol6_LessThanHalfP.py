"""
Protocol 6 — LessThan_Half_P
Secure comparison of a shared secret a with ⌊p/2⌋.

Pseudocode reference (line numbers match the paper):
---------------------------------------------------------------------------
Input:   [[a]] — a (t,n)-sharing of a ∈ Z_p
Line 1:  [[w]] ← 1 − LSB(2[[a]])
Output:  A (t,n)-sharing [[w]] of w = 1_{a < p/2}
---------------------------------------------------------------------------

Intuition
---------
For an odd prime p and a ∈ Z_p:

• If a < p/2:  2a < p, so 2a mod p = 2a, which is even → LSB(2a mod p) = 0
               → w = 1 − 0 = 1  ✓

• If a > p/2:  2a > p, so 2a mod p = 2a − p.
               Since p is odd: 2a − p ≡ 2a − 1 (mod 2) → LSB = 1
               → w = 1 − 1 = 0  ✓

So w = 1_{a < p/2} exactly.

The sub-call LSB( 2[[a]] ) first computes the sharing of 2a mod p locally
(scalar multiplication by 2), then applies Protocol 7 to extract its LSB.
"""

from typing import List, Tuple

try:
    from MPC_over_secret_shares.Protocol7_LSB import protocol_7_lsb
except ImportError:
    from Protocol7_LSB import protocol_7_lsb

Shares = List[Tuple[int, int]]


def protocol_6_less_than_half_p(
    a_shares: Shares,
    n: int,
    t: int,
    p: int,
    print_: bool = False,
) -> Shares:
    """Return a (t,n)-sharing of 1_{a < p/2}.

    Parameters
    ----------
    a_shares : Shares
        A (t,n)-sharing of the secret a ∈ Z_p.
    n, t, p  : int
        MPC parameters.
    print_   : bool
        Verbose debug output.

    Returns
    -------
    Shares
        A (t,n)-sharing [[w]] where w = 1 if a < p/2, else w = 0.
    """
    # ------------------------------------------------------------------
    # Line 1, step (i): 2[[a]] — local scalar multiplication by 2
    # Encodes the secret 2a mod p.
    # ------------------------------------------------------------------
    two_a_shares: Shares = [(x, (2 * y) % p) for x, y in a_shares]

    # ------------------------------------------------------------------
    # Line 1, step (ii): LSB( 2[[a]] ) via Protocol 7
    # Returns a sharing of the LSB of the integer 2a mod p.
    # ------------------------------------------------------------------
    lsb_shares = protocol_7_lsb(two_a_shares, n, t, p, print_=print_)

    # ------------------------------------------------------------------
    # Line 1, step (iii): [[w]] = 1 − LSB(2[[a]])  (local)
    # Subtracting the LSB bit from 1 flips 0→1 and 1→0.
    # ------------------------------------------------------------------
    w_shares: Shares = [(x, (1 - y) % p) for x, y in lsb_shares]   # Line 1

    if print_:
        from Protocol2 import protocol_2_reconstruct
        w_val = protocol_2_reconstruct(w_shares[:t], p)
        print(f"[Protocol6] 1_{{a < p/2}} = {w_val}")

    return w_shares


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random
    from Protocol1 import protocol_1_share
    from Protocol2 import protocol_2_reconstruct

    random.seed(5)
    P, N, T = 11, 3, 2
    half_p = P / 2   # 5.5 for p=11

    # For p=11: a < p/2 = 5.5  →  a ∈ {0, 1, 2, 3, 4, 5} give w=1
    #                           →  a ∈ {6, 7, 8, 9, 10}   give w=0
    all_ok = True
    for secret in range(P):
        expected = 1 if secret < half_p else 0
        sh = protocol_1_share(secret, N, T, P)
        w_sh = protocol_6_less_than_half_p(sh, N, T, P)
        w = protocol_2_reconstruct(w_sh[:T], P)
        status = "✓" if w == expected else "✗"
        print(f"1_{{{secret} < {half_p}}} = {w}  (expected {expected}) {status}")
        all_ok = all_ok and (w == expected)
    print("Protocol 6 self-test", "PASSED ✓" if all_ok else "FAILED ✗")
