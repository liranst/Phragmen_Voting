"""
Protocol 16 — m-ary OR via Polynomial Evaluation

Evaluate  v = OR(u_1, …, u_m)  on m shared bits using a masked degree-m
polynomial evaluation.

Reference: "SoK: Secure Computation over Secret Shares", Protocol 16.
"""

import random
from typing import List, Tuple, Optional

from .protocol2 import protocol_2_reconstruct
from .protocol4 import protocol_4_secure_mult
from .secure_inv import secure_inv, _internal_rng

Shares = List[Tuple[int, int]]


def compute_or_polynomial_coefficients(m: int, p: int) -> List[int]:
    """Return coefficients [α_0, …, α_m] of the OR polynomial over Z_p.

    The polynomial satisfies  f(1) = 0  and  f(j) = 1  for j = 2, …, m+1.

    Parameters
    ----------
    m : int
        Number of input bits.
    p : int
        Prime field modulus (must be > m+1).
    """
    xs = list(range(1, m + 2))
    ys = [0] + [1] * m
    result = [0] * (m + 1)

    for i in range(m + 1):
        if ys[i] == 0:
            continue
        num_poly = [1]
        denom = 1
        for j in range(m + 1):
            if j == i:
                continue
            xj = xs[j]
            new_poly = [0] * (len(num_poly) + 1)
            for k, coeff in enumerate(num_poly):
                new_poly[k]     = (new_poly[k]     - xj * coeff) % p
                new_poly[k + 1] = (new_poly[k + 1] +      coeff) % p
            num_poly = new_poly
            denom    = (denom * (xs[i] - xj)) % p

        denom_inv = pow(denom, p - 2, p)
        for k in range(m + 1):
            if k < len(num_poly):
                result[k] = (result[k] + num_poly[k] * denom_inv) % p

    return result


def protocol_16_m_ary_or(
    u_shares_list: List[Shares],
    n: int,
    t: int,
    p: int,
    alpha_coeffs: Optional[List[int]] = None,
    print_: bool = False,
) -> Shares:
    """Return a (t,n)-sharing of  OR(u_1, …, u_m).

    Parameters
    ----------
    u_shares_list : list of Shares
        m (t,n)-sharings of input bits u_i ∈ {0, 1}.
    n, t, p : int
        MPC parameters.
    alpha_coeffs : list of int, optional
        Pre-computed polynomial coefficients.  Computed automatically if omitted.
    print_ : bool
        Verbose debug output.
    """
    m = len(u_shares_list)
    if alpha_coeffs is None:
        alpha_coeffs = compute_or_polynomial_coefficients(m, p)

    # Lines 1-3: generate m random blinding values and their inverses
    b_list: List[Shares] = []
    c_list: List[Shares] = []
    for _ in range(m):
        b_i = _internal_rng(n, t, p)
        b_list.append(b_i)
        c_list.append(secure_inv(b_i, n, t, p, print_=False))

    # Line 4: [[w]] = 1 + Σ [[u_i]]  (local)
    w_shares: Shares = [(j + 1, 1) for j in range(n)]
    for ui in u_shares_list:
        w_shares = [(x, (yw + yu) % p) for (x, yw), (_, yu) in zip(w_shares, ui)]

    # Line 5: [[b_0]] = [[1]]
    prev_b: Shares = [(j + 1, 1) for j in range(n)]
    e_values: List[int] = []

    # Lines 6-10: build masked powers of w
    for i in range(m):
        d_i = protocol_4_secure_mult(w_shares, prev_b, n, t, p, print_=False)
        e_i_shares = protocol_4_secure_mult(d_i, c_list[i], n, t, p, print_=False)
        e_i = protocol_2_reconstruct(e_i_shares[:t], p)
        e_values.append(e_i)
        if print_:
            print(f"[P16] e_{i+1} = {e_i}")
        prev_b = b_list[i]

    # Lines 11-12: [[w^i]] = (Π_{j=1}^i e_j) · [[b_i]]
    w_power_shares: List[Shares] = [[(j + 1, 1) for j in range(n)]]  # w^0 = 1
    running = 1
    for i in range(m):
        running = (running * e_values[i]) % p
        w_power_shares.append([(x, (running * y) % p) for x, y in b_list[i]])

    # Line 13: [[v]] = Σ α_i [[w^i]]  (local)
    v_shares: Shares = [(j + 1, 0) for j in range(n)]
    for i, alpha_i in enumerate(alpha_coeffs):
        if alpha_i == 0:
            continue
        scaled = [(x, (alpha_i * y) % p) for x, y in w_power_shares[i]]
        v_shares = [(x, (yv + ys) % p)
                    for (x, yv), (_, ys) in zip(v_shares, scaled)]

    return v_shares
