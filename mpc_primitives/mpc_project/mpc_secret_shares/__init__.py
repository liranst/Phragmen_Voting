"""
mpc_secret_shares
=================
A Python library for Secure Multiparty Computation (MPC) over secret shares.

Based on: "SoK: Secure Computation over Secret Shares"

All primary functions are importable directly from this package, e.g.::

    from mpc_secret_shares import protocol_1_share, protocol_4_secure_mult

Typical MPC parameters (used throughout the API):
    n  — number of parties
    t  — reconstruction threshold  (any t shares recover the secret)
    p  — prime field modulus
"""

__version__ = "0.1.1"

# ── Foundational sharing / reconstruction ──────────────────────────────────
from .protocol1 import protocol_1_share
from .protocol2 import protocol_2_reconstruct
from .protocol3 import protocol_3_rng

# ── Core building blocks ───────────────────────────────────────────────────
from .protocol4          import protocol_4_secure_mult
from .secure_inv         import secure_inv
from .affine_combinations import protocol_3_1_affine_combination
from .share_utils        import (
    shares_zero,
    shares_one,
    shares_const,
    shares_add,
    shares_sub,
    shares_scalar,
    shares_add_const,
    shares_negate,
    shares_sum,
)

# ── Bit extraction ─────────────────────────────────────────────────────────
from .protocol8_gen_rnd_bit_sharing import protocol_8_gen_rnd_bit_sharing
from .protocol9_bitwise_less_than   import protocol_9_bitwise_less_than
from .protocol7_lsb                 import protocol_7_lsb

# ── Comparison & equality ──────────────────────────────────────────────────
from .protocol6_less_than_half_p import protocol_6_less_than_half_p
from .protocol5_secure_compare   import protocol_5_secure_compare
from .protocol10                 import protocol_10_is_zero

# ── Arithmetic protocols ───────────────────────────────────────────────────
from .protocol12              import binary_long_division
from .protocol13_secure_division import protocol_13_secure_division

# ── Square root protocols ──────────────────────────────────────────────────
from .algorithm14_bitwise_sqrt import algorithm_14_bitwise_sqrt
from .protocol15_secure_sqrt   import protocol_15_secure_sqrt

# ── OR protocols ───────────────────────────────────────────────────────────
from .protocol16_mary_or         import (
    protocol_16_m_ary_or,
    compute_or_polynomial_coefficients,
)
from .protocol17_improved_mary_or import protocol_17_improved_m_ary_or

# ── Statistics / ranking ───────────────────────────────────────────────────
from .algorithm18_kth_ranked import algorithm_18_kth_ranked

# ── Misc utilities ─────────────────────────────────────────────────────────
from .square_and_multiply import square_and_multiply

__all__ = [
    # version
    "__version__",
    # protocol 1-4
    "protocol_1_share",
    "protocol_2_reconstruct",
    "protocol_3_rng",
    "protocol_4_secure_mult",
    # share arithmetic helpers
    "shares_zero", "shares_one", "shares_const",
    "shares_add", "shares_sub", "shares_scalar",
    "shares_add_const", "shares_negate", "shares_sum",
    "protocol_3_1_affine_combination",
    # secure inverse
    "secure_inv",
    # bit / comparison
    "protocol_8_gen_rnd_bit_sharing",
    "protocol_9_bitwise_less_than",
    "protocol_7_lsb",
    "protocol_6_less_than_half_p",
    "protocol_5_secure_compare",
    "protocol_10_is_zero",
    # division & sqrt
    "binary_long_division",
    "protocol_13_secure_division",
    "algorithm_14_bitwise_sqrt",
    "protocol_15_secure_sqrt",
    # OR
    "protocol_16_m_ary_or",
    "compute_or_polynomial_coefficients",
    "protocol_17_improved_m_ary_or",
    # ranking
    "algorithm_18_kth_ranked",
    # utilities
    "square_and_multiply",
]
