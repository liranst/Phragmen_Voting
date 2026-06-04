"""
Protocol 16 — Computing an m-ary OR
Evaluate OR(u_1, ..., u_m) on m secret bits u_i ∈ {0,1}, each given as a
(t,n)-sharing, using a degree-m polynomial evaluation over Z_p.

Pseudocode reference (line numbers match the paper):
---------------------------------------------------------------------------
Input:   [[u_i]], i ∈ [m]
Parameter: f(x) = Σ_{i=0}^{m} α_i x^i
           where f(1) = 0  and  f(j) = 1 for j = 2, ..., m+1

Line 1:  for all i ∈ [m] do
Line 2:    [[b_i]] ← RNG()
Line 3:    [[c_i]] ← SecureInv([[b_i]])
Line 4:  [[w]] ← 1 + Σ_{i ∈ [m]} [[u_i]]
Line 5:  [[b_0]] ← [[1]]
Line 6:  for all i ∈ [m] do
Line 7:    [[d_i]] ← SecureMult([[w]], [[b_{i-1}]])
Line 8:    [[e_i]] ← SecureMult([[d_i]], [[c_i]])
Line 9:    The parties run Reconstruct([[e_i]], 1; t)
Line 10:   P_1 broadcasts e_i
Line 11: for all i = 2 to m do           (include i=1 as w^1 = e_1 * [[b_1]])
Line 12:   [[w^i]] ← (Π_{j=1}^{i} e_j) * [[b_i]]
Line 13: [[v]] ← Σ_{i=0}^{m} α_i [[w^i]]
Output:  A sharing [[v]] of v = OR_{i ∈ [m]} u_i
---------------------------------------------------------------------------

Key idea (random masking for polynomial evaluation)
----------------------------------------------------
w = 1 + Σ u_i  ∈ {1, 2, ..., m+1}  (public: w=1 iff all u_i=0).

The parties want to compute f(w) without revealing w.  They use a sequence
of random blinding values b_1, ..., b_m (masked products):

    e_i = w · b_{i-1} · b_i^{-1}          (telescoping mask)

Product of first i values (telescoping):
    Π_{j=1}^i e_j = w^i · b_0 · b_i^{-1} = w^i · b_i^{-1}

So  [[w^i]] = (Π_{j=1}^i e_j) · [[b_i]] = w^i · b_i^{-1} · [[b_i]] = [[w^i]] ✓.

The e_j are reconstructed and broadcast — they reveal only w masked by the
random blinding chain, not w itself.

Finally,  [[v]] = Σ α_i [[w^i]]  is a local affine combination.

Polynomial coefficients
-----------------------
The α_i are computed inside this function via Lagrange interpolation of the
step function  f(1)=0, f(2)=f(3)=...=f(m+1)=1  over Z_p.
They can also be passed in externally via the `alpha_coeffs` parameter.
"""

import random
from typing import List, Tuple, Optional

try:
    from MPC_over_secret_shares.Protocol1 import protocol_1_share
    from MPC_over_secret_shares.Protocol2 import protocol_2_reconstruct
    from MPC_over_secret_shares.Protocol4 import protocol_4_secure_mult
    from MPC_over_secret_shares.secure_inv import secure_inv, _internal_rng
except ImportError:
    from Protocol1 import protocol_1_share
    from Protocol2 import protocol_2_reconstruct
    from Protocol4 import protocol_4_secure_mult
    from secure_inv import secure_inv, _internal_rng

Shares = List[Tuple[int, int]]


# ---------------------------------------------------------------------------
# Helper — Lagrange interpolation of the OR polynomial over Z_p
# ---------------------------------------------------------------------------

def compute_or_polynomial_coefficients(m: int, p: int) -> List[int]:
    """Return [α_0, α_1, ..., α_m] of the degree-m polynomial f over Z_p
    such that f(1) = 0 and f(j) = 1 for j = 2, ..., m+1.

    Parameters
    ----------
    m : int
        Number of input bits (arity of the OR gate).
    p : int
        Prime field modulus.  Must be > m+1.

    Returns
    -------
    List[int]
        Polynomial coefficients [α_0, α_1, ..., α_m] in Z_p,
        where the polynomial is  f(x) = Σ α_i x^i.
    """
    # Evaluation points: x = 1, 2, ..., m+1
    # Corresponding values: y(1) = 0,  y(j) = 1  for j ≥ 2
    xs = list(range(1, m + 2))
    ys = [0] + [1] * m

    # Lagrange interpolation: build each basis polynomial L_i(x)
    # then accumulate  f = Σ y_i L_i.
    # Polynomials are stored as coefficient lists, coeffs[k] = coefficient of x^k.

    result = [0] * (m + 1)   # degree-m polynomial has m+1 coefficients

    for i in range(m + 1):
        if ys[i] == 0:
            continue        # skip: y_i = 0 contributes nothing

        # Build the numerator polynomial Π_{j≠i} (x − x_j)
        num_poly = [1]      # start with the constant polynomial 1
        denom    = 1

        for j in range(m + 1):
            if j == i:
                continue

            # Multiply num_poly by (x − xs[j])
            # If current poly is [a0, a1, ...], new poly is:
            #   [−xs[j]*a0,  a0 − xs[j]*a1,  a1 − xs[j]*a2, ..., a_{k-1}]
            xj = xs[j]
            new_poly = [0] * (len(num_poly) + 1)
            for k, coeff in enumerate(num_poly):
                new_poly[k]     = (new_poly[k]     - xj * coeff) % p  # − x_j term
                new_poly[k + 1] = (new_poly[k + 1] +      coeff) % p  # x term

            num_poly = new_poly
            denom    = (denom * (xs[i] - xj)) % p

        denom_inv = pow(denom, p - 2, p)   # modular inverse via Fermat

        # Add  y_i · L_i  to result (y_i = 1, so just add L_i coefficients)
        for k in range(m + 1):
            if k < len(num_poly):
                result[k] = (result[k] + num_poly[k] * denom_inv) % p

    return result


# ---------------------------------------------------------------------------
# Protocol 16
# ---------------------------------------------------------------------------

def protocol_16_m_ary_or(
    u_shares_list: List[Shares],
    n: int,
    t: int,
    p: int,
    alpha_coeffs: Optional[List[int]] = None,
    print_: bool = False,
) -> Shares:
    """Compute a (t,n)-sharing of OR(u_1, ..., u_m).

    Parameters
    ----------
    u_shares_list : List[Shares]
        List of m (t,n)-sharings, one per input bit u_i ∈ {0,1}.
    n, t, p       : int
        MPC parameters.
    alpha_coeffs  : List[int], optional
        Pre-computed polynomial coefficients [α_0, ..., α_m].
        If omitted, they are computed internally via Lagrange interpolation.
    print_        : bool
        Verbose debug output.

    Returns
    -------
    Shares
        A (t,n)-sharing [[v]] where v = OR(u_1, ..., u_m).
    """
    m = len(u_shares_list)

    # Compute the OR polynomial coefficients if not supplied
    if alpha_coeffs is None:
        alpha_coeffs = compute_or_polynomial_coefficients(m, p)
    if print_:
        print(f"[Protocol16] m={m}, alpha_coeffs = {alpha_coeffs}")

    # ------------------------------------------------------------------
    # Lines 1-3: Generate m random blinding values and their inverses.
    # b_i = RNG(),  c_i = SecureInv(b_i)  for i = 1, ..., m
    # ------------------------------------------------------------------
    b_list: List[Shares] = []   # b_1, ..., b_m   (index 0 = b_1)
    c_list: List[Shares] = []   # c_1 = b_1^{-1}, ..., c_m = b_m^{-1}

    for i in range(m):                                               # Line 1
        b_i = _internal_rng(n, t, p)                                # Line 2
        b_list.append(b_i)
        c_i = secure_inv(b_i, n, t, p, print_=False)                # Line 3
        c_list.append(c_i)

    # ------------------------------------------------------------------
    # Line 4: [[w]] = 1 + Σ_{i=1}^m [[u_i]]  (local)
    # w ∈ {1, ..., m+1}: 1 iff all bits are 0, ≥ 2 iff at least one is 1.
    # ------------------------------------------------------------------
    w_shares: Shares = [(j + 1, 1) for j in range(n)]              # start at 1
    for ui_shares in u_shares_list:
        w_shares = [(x, (yw + yu) % p)
                    for (x, yw), (_, yu) in zip(w_shares, ui_shares)]  # Line 4

    # ------------------------------------------------------------------
    # Line 5: [[b_0]] = [[1]]  (the identity mask)
    # ------------------------------------------------------------------
    b0_shares: Shares = [(j + 1, 1) for j in range(n)]             # Line 5

    # ------------------------------------------------------------------
    # Lines 6-10: Build masked powers of w.
    #   d_i = w · b_{i-1}      (SecureMult)
    #   e_i = d_i · c_i        (SecureMult, then reconstruct)
    # After reconstruction, e_i is a *public* blinded power of w.
    # ------------------------------------------------------------------
    e_values: List[int]  = []   # public e_1, ..., e_m (after reconstruction)
    d_shares_list: List[Shares] = []
    prev_b: Shares = b0_shares  # b_{i-1}, starting with b_0 = [[1]]

    for i in range(m):                                              # Line 6
        # Line 7: [[d_i]] = SecureMult([[w]], [[b_{i-1}]])
        d_i = protocol_4_secure_mult(w_shares, prev_b, n, t, p, print_=False)
        d_shares_list.append(d_i)                                   # Line 7

        # Line 8: [[e_i]] = SecureMult([[d_i]], [[c_i]])
        e_i_shares = protocol_4_secure_mult(
            d_i, c_list[i], n, t, p, print_=False
        )                                                            # Line 8

        # Lines 9-10: Reconstruct e_i publicly (use t shares)
        e_i_val = protocol_2_reconstruct(e_i_shares[:t], p)         # Lines 9-10
        e_values.append(e_i_val)
        if print_:
            print(f"[Protocol16] e_{i+1} = {e_i_val}")

        prev_b = b_list[i]      # advance b_{i-1} → b_i for next iteration

    # ------------------------------------------------------------------
    # Lines 11-12: Compute [[w^i]] = (Π_{j=1}^i e_j) · [[b_i]]
    #   for i = 1, ..., m  (the pseudocode starts the explicit loop at i=2,
    #   but the formula i=1 is identical and is included here for uniformity).
    # ------------------------------------------------------------------
    # [[w^0]] = [[1]]
    w_power_shares: List[Shares] = [[(j + 1, 1) for j in range(n)]]   # [[1]]

    running_product = 1   # Π_{j=1}^i e_j, updated incrementally
    for i in range(m):                                               # Line 11
        running_product = (running_product * e_values[i]) % p
        # [[w^{i+1}]] = running_product · [[b_{i+1}]]
        w_i_shares: Shares = [
            (x, (running_product * y) % p) for x, y in b_list[i]
        ]                                                            # Line 12
        w_power_shares.append(w_i_shares)

    # ------------------------------------------------------------------
    # Line 13: [[v]] = Σ_{i=0}^m α_i [[w^i]]  (local affine combination)
    # ------------------------------------------------------------------
    v_shares: Shares = [(j + 1, 0) for j in range(n)]
    for i, alpha_i in enumerate(alpha_coeffs):
        if alpha_i == 0:
            continue
        scaled: Shares = [(x, (alpha_i * y) % p) for x, y in w_power_shares[i]]
        v_shares = [(x, (yv + ys) % p)
                    for (x, yv), (_, ys) in zip(v_shares, scaled)]  # Line 13

    if print_:
        v_val = protocol_2_reconstruct(v_shares[:t], p)
        print(f"[Protocol16] OR = {v_val}")

    return v_shares


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(21)
    from Protocol1 import protocol_1_share
    from Protocol2 import protocol_2_reconstruct

    P, N, T = 11, 3, 2

    def share_bit(b_val: int) -> Shares:
        return protocol_1_share(b_val, N, T, P)

    test_cases = [
        ([0, 0], 0),
        ([1, 0], 1),
        ([0, 1], 1),
        ([1, 1], 1),
        ([0, 0, 0], 0),
        ([1, 0, 0], 1),
        ([0, 0, 1], 1),
    ]

    all_ok = True
    for bits, expected in test_cases:
        m = len(bits)
        u_sh_list = [share_bit(b) for b in bits]
        v_sh = protocol_16_m_ary_or(u_sh_list, N, T, P, print_=False)
        v    = protocol_2_reconstruct(v_sh[:T], P)
        status = "✓" if v == expected else "✗"
        print(f"OR{bits} = {v}  (expected {expected}) {status}")
        all_ok = all_ok and (v == expected)

    print("Protocol 16 self-test", "PASSED ✓" if all_ok else "FAILED ✗")
