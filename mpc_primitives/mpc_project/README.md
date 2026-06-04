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

Or install directly from source:

```bash
git clone https://github.com/yourusername/mpc-secret-shares.git
cd mpc-secret-shares/mpc_project
pip install .
```

---

## Quick Start

```python
import random
from mpc_secret_shares import (
    protocol_1_share,
    protocol_2_reconstruct,
    protocol_4_secure_mult,
    protocol_5_secure_compare,
    protocol_10_is_zero,
)

random.seed(42)

# --- MPC parameters ---
p = 11   # prime field  Z_11
n = 3    # three parties
t = 2    # any 2 shares reconstruct the secret

# --- Share two secrets ---
shares_u = protocol_1_share(S=4, n=n, t=t, P=p)
shares_v = protocol_1_share(S=5, n=n, t=t, P=p)

# --- Secure multiplication: 4 * 5 = 20 ≡ 9 (mod 11) ---
shares_w = protocol_4_secure_mult(shares_u, shares_v, n=n, t=t, P=p)
w = protocol_2_reconstruct(shares_w[:t], p)
print(f"4 × 5 mod 11 = {w}")   # → 9

# --- Secure comparison: 4 < 5 ? ---
shares_cmp = protocol_5_secure_compare(shares_u, shares_v, n=n, t=t, p=p)
cmp_result = protocol_2_reconstruct(shares_cmp[:t], p)
print(f"4 < 5 → {bool(cmp_result)}")   # → True

# --- IsZero test ---
shares_zero = protocol_1_share(S=0, n=n, t=t, P=p)
shares_iz = protocol_10_is_zero(shares_zero, n=n, t=t, p=p)
print(f"IsZero(0) = {protocol_2_reconstruct(shares_iz[:t], p)}")  # → 1
```

---

## Implemented Protocols

| Module | Protocol | Description |
|--------|----------|-------------|
| `protocol1` | Share | Shamir secret sharing |
| `protocol2` | Reconstruct | Lagrange interpolation |
| `protocol3` | RNG | Shared random number generation |
| `protocol4` | SecureMult | Secure multiplication |
| `protocol5_secure_compare` | SecureCompare | Secure `1_{u < v}` |
| `protocol6_less_than_half_p` | LessThan_Half_P | Secure `1_{a < p/2}` |
| `protocol7_lsb` | LSB | Secure least-significant-bit extraction |
| `protocol8_gen_rnd_bit_sharing` | GenRndBitSharing | Random bit generation |
| `protocol9_bitwise_less_than` | Bitwise_LessThan | Bitwise comparison |
| `protocol10` | IsZero | Secure equality to zero |
| `protocol12` | BinaryLongDiv | Cleartext binary long division |
| `protocol13_secure_division` | SecureDivision | Secure floor division |
| `algorithm14_bitwise_sqrt` | BitwiseSqrt | Cleartext integer square root |
| `protocol15_secure_sqrt` | SecureSqrt | Secure integer square root |
| `protocol16_mary_or` | m-ary OR | Secure OR via polynomial evaluation |
| `protocol17_improved_mary_or` | Improved OR | Communication-efficient OR |
| `algorithm18_kth_ranked` | KthRanked | Cleartext k-th ranked element |
| `affine_combinations` | AffineCombo | Local α + β·u + γ·v |
| `share_utils` | — | Local share arithmetic helpers |
| `secure_inv` | SecureInv | Secure modular inverse |

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
