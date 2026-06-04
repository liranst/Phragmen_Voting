"""
Protocol 7 — LSB (Least Significant Bit extraction)

Securely compute a sharing of  x₀ = x mod 2  (the LSB of x).

The trick: mask x with a random r whose bits are individually shared.
Reconstruct c = x + r (mod p) publicly, then recover x₀ from c₀ (LSB of c)
and [[r₀]] using a bitwise carry-detection step.

Reference: "SoK: Secure Computation over Secret Shares", Protocol 7.
"""

import math
from typing import List, Tuple

from .protocol2 import protocol_2_reconstruct
from .protocol4 import protocol_4_secure_mult
from .protocol8_gen_rnd_bit_sharing import protocol_8_gen_rnd_bit_sharing
from .protocol9_bitwise_less_than import protocol_9_bitwise_less_than

Shares = List[Tuple[int, int]]


def protocol_7_lsb(
    x_shares: Shares,
    n: int,
    t: int,
    p: int,
    print_: bool = False,
) -> Shares:
    """Return a (t,n)-sharing of  x₀ = x mod 2.

    Parameters
    ----------
    x_shares : Shares
        A (t,n)-sharing of x ∈ Z_p.
    n, t, p : int
        MPC parameters.  Requires n ≥ 2t-1.
    print_ : bool
        Verbose debug output.

    Returns
    -------
    Shares
        A (t,n)-sharing [[x₀]] where x₀ = LSB(x).
    """
    s = math.ceil(math.log2(p)) if p > 2 else 1

    # Lines 1-2: generate s random bit sharings [[r_i]] with r_int < p
    # Rejection-sample until Σ 2^i·r_i < p to ensure correct carry detection.
    from .protocol2 import protocol_2_reconstruct as _recon
    while True:
        r_bit_shares = [
            protocol_8_gen_rnd_bit_sharing(n, t, p, print_=False)
            for _ in range(s)
        ]
        r_int = sum(
            (1 << i) * _recon(r_bit_shares[i][:t], p)
            for i in range(s)
        )
        if r_int < p:
            break

    # Lines 3-4: [[r]] = Σ 2ⁱ [[r_i]]  (local)
    r_shares: Shares = [(j + 1, 0) for j in range(n)]
    for i in range(s):
        power = pow(2, i, p)
        scaled = [(x, (power * y) % p) for x, y in r_bit_shares[i]]
        r_shares = [(x, (yr + ys) % p)
                    for (x, yr), (_, ys) in zip(r_shares, scaled)]

    # Line 5: [[c]] = [[x]] + [[r]]  (local)
    c_shares: Shares = [(x, (yx + yr) % p)
                        for (x, yx), (_, yr) in zip(x_shares, r_shares)]

    # Lines 6-7: reconstruct c publicly
    c = protocol_2_reconstruct(c_shares[:t], p)
    if print_:
        print(f"[Protocol7] c (public) = {c}")

    # Lines 8-11: d₀ = c₀ XOR r₀
    c_0 = c & 1
    if c_0 == 0:
        d0_shares: Shares = r_bit_shares[0]
    else:
        d0_shares = [(x, (1 - y) % p) for x, y in r_bit_shares[0]]

    # Line 12: [[e]] = Bitwise_LessThan(c, {[[r_i]]}) = 1_{c < r}
    e_shares = protocol_9_bitwise_less_than(
        c, r_bit_shares, n, t, p, print_=False
    )

    # Line 13: [[y]] = SecureMult([[e]], [[d₀]])
    y_shares = protocol_4_secure_mult(
        e_shares, d0_shares, n, t, p, print_=False
    )

    # Line 14: [[x₀]] = [[e]] + [[d₀]] − 2[[y]]  (XOR formula, local)
    x0_shares: Shares = [
        (x, (ye + yd - 2 * yy) % p)
        for (x, ye), (_, yd), (_, yy)
        in zip(e_shares, d0_shares, y_shares)
    ]

    if print_:
        x0_val = protocol_2_reconstruct(x0_shares[:t], p)
        print(f"[Protocol7] LSB(x) = {x0_val}")

    return x0_shares
