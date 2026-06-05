"""
mpc_secret_shares
=================
A Python library for Secure Multiparty Computation (MPC) over secret shares.

Based on: "SoK: Secure Computation over Secret Shares"

v0.2.0 introduces a clean short-name API.  All original ``protocol_*`` /
``algorithm_*`` names remain importable but emit a ``DeprecationWarning``
and will be removed in v1.0.

Clean API::

    from mpc_secret_shares import share, reconstruct, secure_mult

Typical MPC parameters (used throughout the API):
    n  — number of parties
    t  — reconstruction threshold  (any t shares recover the secret)
    p  — prime field modulus
"""

import functools
import warnings

__version__ = "0.2.0"

# ── Internal imports ───────────────────────────────────────────────────────────
from .protocol1  import protocol_1_share               as _share_impl
from .protocol2  import protocol_2_reconstruct         as _reconstruct_impl
from .protocol3  import protocol_3_rng
from .protocol4  import protocol_4_secure_mult         as _secure_mult_impl
from .secure_inv import secure_inv
from .affine_combinations import protocol_3_1_affine_combination
from .share_utils import (
    shares_zero, shares_one, shares_const,
    shares_add, shares_sub, shares_scalar,
    shares_add_const, shares_negate, shares_sum,
)
from .protocol8_gen_rnd_bit_sharing import protocol_8_gen_rnd_bit_sharing  as _random_bit_impl
from .protocol9_bitwise_less_than   import protocol_9_bitwise_less_than    as _bitwise_lt_impl
from .protocol7_lsb                 import protocol_7_lsb                  as _lsb_impl
from .protocol6_less_than_half_p    import protocol_6_less_than_half_p     as _less_than_half_p_impl
from .protocol5_secure_compare      import protocol_5_secure_compare       as _secure_lt_impl
from .protocol10                    import protocol_10_is_zero              as _is_zero_impl
from .protocol12                    import binary_long_division
from .protocol13_secure_division    import protocol_13_secure_division      as _secure_div_impl
from .algorithm14_bitwise_sqrt      import algorithm_14_bitwise_sqrt
from .protocol15_secure_sqrt        import protocol_15_secure_sqrt          as _secure_sqrt_impl
from .protocol16_mary_or            import (
    protocol_16_m_ary_or            as _mary_or_impl,
    compute_or_polynomial_coefficients,
)
from .protocol17_improved_mary_or   import protocol_17_improved_m_ary_or   as _improved_mary_or_impl
from .algorithm18_kth_ranked        import algorithm_18_kth_ranked         as _kth_ranked_impl
from .square_and_multiply           import square_and_multiply

# ── Deprecation helper ─────────────────────────────────────────────────────────

def _deprecated(new_name: str, func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"'{func.__name__}' is deprecated; use '{new_name}' instead. "
            "It will be removed in v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)
    return wrapper

# ── Clean API (v0.2.0+) ────────────────────────────────────────────────────────
share            = _share_impl
reconstruct      = _reconstruct_impl
secure_mult      = _secure_mult_impl
secure_lt        = _secure_lt_impl
less_than_half_p = _less_than_half_p_impl
lsb              = _lsb_impl
random_bit       = _random_bit_impl
bitwise_lt       = _bitwise_lt_impl
is_zero          = _is_zero_impl
# secure_inv keeps its name — no alias needed
secure_div       = _secure_div_impl
secure_sqrt      = _secure_sqrt_impl
mary_or          = _mary_or_impl
improved_mary_or = _improved_mary_or_impl
kth_ranked       = _kth_ranked_impl

# ── Deprecated old names (backward-compatible) ─────────────────────────────────
protocol_1_share               = _deprecated("share",            _share_impl)
protocol_2_reconstruct         = _deprecated("reconstruct",      _reconstruct_impl)
protocol_4_secure_mult         = _deprecated("secure_mult",      _secure_mult_impl)
protocol_5_secure_compare      = _deprecated("secure_lt",        _secure_lt_impl)
protocol_6_less_than_half_p    = _deprecated("less_than_half_p", _less_than_half_p_impl)
protocol_7_lsb                 = _deprecated("lsb",              _lsb_impl)
protocol_8_gen_rnd_bit_sharing = _deprecated("random_bit",       _random_bit_impl)
protocol_9_bitwise_less_than   = _deprecated("bitwise_lt",       _bitwise_lt_impl)
protocol_10_is_zero            = _deprecated("is_zero",          _is_zero_impl)
protocol_13_secure_division    = _deprecated("secure_div",       _secure_div_impl)
protocol_15_secure_sqrt        = _deprecated("secure_sqrt",      _secure_sqrt_impl)
protocol_16_m_ary_or           = _deprecated("mary_or",          _mary_or_impl)
protocol_17_improved_m_ary_or  = _deprecated("improved_mary_or", _improved_mary_or_impl)
algorithm_18_kth_ranked        = _deprecated("kth_ranked",       _kth_ranked_impl)

__all__ = [
    "__version__",
    # ── Clean API ──────────────────────────────────────────────────────────────
    "share", "reconstruct", "secure_mult", "secure_lt",
    "less_than_half_p", "lsb", "random_bit", "bitwise_lt",
    "is_zero", "secure_inv", "secure_div", "secure_sqrt",
    "mary_or", "improved_mary_or", "kth_ranked",
    # ── No-alias names (unchanged) ─────────────────────────────────────────────
    "protocol_3_rng",
    "protocol_3_1_affine_combination",
    "binary_long_division",
    "algorithm_14_bitwise_sqrt",
    "compute_or_polynomial_coefficients",
    "square_and_multiply",
    # ── Share arithmetic helpers ───────────────────────────────────────────────
    "shares_zero", "shares_one", "shares_const",
    "shares_add", "shares_sub", "shares_scalar",
    "shares_add_const", "shares_negate", "shares_sum",
    # ── Deprecated old names (still exported for backward compat) ──────────────
    "protocol_1_share", "protocol_2_reconstruct",
    "protocol_4_secure_mult", "protocol_5_secure_compare",
    "protocol_6_less_than_half_p", "protocol_7_lsb",
    "protocol_8_gen_rnd_bit_sharing", "protocol_9_bitwise_less_than",
    "protocol_10_is_zero", "protocol_13_secure_division",
    "protocol_15_secure_sqrt", "protocol_16_m_ary_or",
    "protocol_17_improved_m_ary_or", "algorithm_18_kth_ranked",
]
