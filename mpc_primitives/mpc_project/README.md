# mpc-secret-shares

A pure-Python library for **Secure Multiparty Computation (MPC) over Secret Shares**.

Implements the protocols described in:

> _SoK: Secure Computation over Secret Shares_

All protocols operate over a prime finite field **Z_p** using **Shamir's (t,n)-secret sharing**:
any _t_ out of _n_ parties can reconstruct a secret; fewer than _t_ parties learn nothing.

---

## Installation

```bash
pip install mpc-secret-shares
```

---

## Quick Start

```python
import random
from mpc_secret_shares import share, reconstruct, secure_mult, secure_lt, is_zero

random.seed(42)

# MPC parameters
p = 11   # prime field Z_11
n = 3    # three parties
t = 2    # any 2 shares reconstruct the secret

# Share two secrets
sh_u = share(S=4, n=n, t=t, P=p)
sh_v = share(S=5, n=n, t=t, P=p)

# Secure multiplication: 4 × 5 = 20 ≡ 9 (mod 11)
sh_w = secure_mult(sh_u, sh_v, n=n, t=t, P=p)
print(f"4 × 5 mod 11 = {reconstruct(sh_w[:t], p)}")   # → 9

# Secure comparison: 4 < 5 ?
sh_cmp = secure_lt(sh_u, sh_v, n=n, t=t, p=p)
print(f"4 < 5 → {bool(reconstruct(sh_cmp[:t], p))}")  # → True

# IsZero test
sh_zero = share(S=0, n=n, t=t, P=p)
print(f"is_zero(0) = {reconstruct(is_zero(sh_zero, n=n, t=t, p=p)[:t], p)}")  # → 1
```

---

## API Reference

### v0.2.0 clean names

| Clean name | Description |
|---|---|
| `share(S, n, t, P)` | Shamir secret sharing |
| `reconstruct(shares, p)` | Lagrange interpolation |
| `secure_mult(u, v, n, t, p)` | Secure multiplication |
| `secure_lt(u, v, n, t, p)` | Secure `1_{u < v}` |
| `less_than_half_p(u, n, t, p)` | Secure `1_{u < p/2}` |
| `lsb(u, n, t, p)` | Secure least-significant-bit |
| `random_bit(n, t, p)` | Shared random bit |
| `bitwise_lt(a, b_bits, n, t, p)` | Bitwise comparison |
| `is_zero(u, n, t, p)` | Secure equality to zero |
| `secure_inv(u, n, t, p)` | Secure modular inverse |
| `secure_div(u, v, u_bits, n, t, p)` | Secure floor division |
| `secure_sqrt(u, u_bits, n, t, p)` | Secure integer square root |
| `mary_or(bits, n, t, p)` | Secure m-ary OR |
| `improved_mary_or(bits, n, t, p)` | Communication-efficient OR |
| `kth_ranked(dataset, M, k)` | k-th ranked element (cleartext) |

### Unchanged names

| Name | Description |
|---|---|
| `protocol_3_rng(n, t, p)` | Shared random number |
| `protocol_3_1_affine_combination(a, b, c, u, v, p)` | Local α + β·u + γ·v |
| `binary_long_division(u, v)` | Cleartext binary long division |
| `algorithm_14_bitwise_sqrt(u)` | Cleartext integer square root |
| `compute_or_polynomial_coefficients(m, p)` | OR polynomial coefficients |
| `square_and_multiply(x, c, n)` | Modular exponentiation |
| `shares_add / sub / scalar / ...` | Local share arithmetic |

### Deprecated names (removed in v1.0)

All `protocol_N_*` and `algorithm_N_*` names that have a clean alias above
still work but emit a `DeprecationWarning` on every call.

```python
# Old style (works, but warns):
from mpc_secret_shares import protocol_1_share
sh = protocol_1_share(7, n=3, t=2, P=11)
# DeprecationWarning: 'protocol_1_share' is deprecated; use 'share' instead.

# New style:
from mpc_secret_shares import share
sh = share(7, n=3, t=2, P=11)
```

---

## Running the Tests

```bash
cd mpc_project
pip install pytest
pytest tests/ -v
```

---

## License

MIT — see [LICENSE](LICENSE).
