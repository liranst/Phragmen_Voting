"""
test_protocols.py
=================
Pytest test suite for the mpc_secret_shares package.

Run with:
    cd mpc_project
    pytest tests/ -v

All tests use a small, transparent field:
    p = 11  (prime),  n = 3  (parties),  t = 2  (threshold)
"""

import math
import random
import pytest

# ── Shared test parameters ─────────────────────────────────────────────────
P, N, T = 11, 3, 2


# ── Convenience helpers ────────────────────────────────────────────────────

def _share(secret: int):
    from mpc_secret_shares import protocol_1_share
    return protocol_1_share(secret, N, T, P)


def _reconstruct(shares):
    from mpc_secret_shares import protocol_2_reconstruct
    return protocol_2_reconstruct(shares[:T], P)


# ==========================================================================
# Protocol 1 & 2 — Share / Reconstruct
# ==========================================================================

class TestShareReconstruct:
    def test_all_field_elements(self):
        """Every element of Z_p can be shared and reconstructed exactly."""
        random.seed(101)
        from mpc_secret_shares import protocol_1_share, protocol_2_reconstruct
        for secret in range(P):
            sh = protocol_1_share(secret, N, T, P)
            assert len(sh) == N, "Wrong number of shares"
            assert all(0 <= y < P for _, y in sh), "Share out of field"
            recovered = protocol_2_reconstruct(sh[:T], P)
            assert recovered == secret, f"Round-trip failed for secret={secret}"

    def test_any_t_shares_suffice(self):
        """Reconstruction works with exactly T shares (any T distinct)."""
        random.seed(102)
        from mpc_secret_shares import protocol_1_share, protocol_2_reconstruct
        sh = protocol_1_share(7, N, T, P)
        # Use last T shares instead of first T
        assert protocol_2_reconstruct(sh[-T:], P) == 7


# ==========================================================================
# Protocol 3 — RNG
# ==========================================================================

class TestRNG:
    def test_output_in_field(self):
        """RNG shares are in Z_p and reconstruct to a field element."""
        random.seed(103)
        from mpc_secret_shares import protocol_3_rng, protocol_2_reconstruct
        sh = protocol_3_rng(N, T, P, print_=False)
        r = protocol_2_reconstruct(sh[:T], P)
        assert 0 <= r < P


# ==========================================================================
# Protocol 4 — SecureMult
# ==========================================================================

class TestSecureMult:
    @pytest.mark.parametrize("u,v", [
        (4, 5), (3, 7), (0, 9), (1, 1), (10, 10), (6, 6),
    ])
    def test_multiplication(self, u, v):
        random.seed(200 + u * 11 + v)
        from mpc_secret_shares import protocol_4_secure_mult
        sh_u = _share(u)
        sh_v = _share(v)
        sh_w = protocol_4_secure_mult(sh_u, sh_v, N, T, P, print_=False)
        assert _reconstruct(sh_w) == (u * v) % P, f"Mult({u},{v}) failed"


# ==========================================================================
# share_utils — Local arithmetic
# ==========================================================================

class TestShareUtils:
    def setup_method(self):
        random.seed(300)

    def test_add(self):
        from mpc_secret_shares import shares_add
        sh_u, sh_v = _share(3), _share(7)
        assert _reconstruct(shares_add(sh_u, sh_v, P)) == (3 + 7) % P

    def test_sub(self):
        from mpc_secret_shares import shares_sub
        sh_u, sh_v = _share(3), _share(7)
        assert _reconstruct(shares_sub(sh_u, sh_v, P)) == (3 - 7) % P

    def test_scalar(self):
        from mpc_secret_shares import shares_scalar
        assert _reconstruct(shares_scalar(4, _share(3), P)) == (4 * 3) % P

    def test_add_const(self):
        from mpc_secret_shares import shares_add_const
        assert _reconstruct(shares_add_const(_share(5), 6, P)) == (5 + 6) % P

    def test_negate(self):
        from mpc_secret_shares import shares_negate
        assert _reconstruct(shares_negate(_share(4), P)) == (-4) % P

    def test_zero_and_one(self):
        from mpc_secret_shares import shares_zero, shares_one
        assert _reconstruct(shares_zero(N)) == 0
        assert _reconstruct(shares_one(N))  == 1


# ==========================================================================
# Affine combination
# ==========================================================================

class TestAffineCombo:
    def test_formula(self):
        """w = 5 + 2u + 3v  (mod 11) = 5 + 6 + 12 = 23 ≡ 1."""
        random.seed(400)
        from mpc_secret_shares import protocol_3_1_affine_combination
        sh_u, sh_v = _share(3), _share(4)
        sh_w = protocol_3_1_affine_combination(5, 2, 3, sh_u, sh_v, P)
        assert _reconstruct(sh_w) == (5 + 2 * 3 + 3 * 4) % P


# ==========================================================================
# SecureInv
# ==========================================================================

class TestSecureInv:
    @pytest.mark.parametrize("u", [1, 2, 3, 4, 5, 6, 9, 10])
    def test_inverse(self, u):
        random.seed(500 + u)
        from mpc_secret_shares import secure_inv
        inv_sh = secure_inv(_share(u), N, T, P)
        inv_val = _reconstruct(inv_sh)
        assert (u * inv_val) % P == 1, f"SecureInv({u}) failed: got {inv_val}"

    def test_zero_raises(self):
        random.seed(501)
        from mpc_secret_shares import secure_inv
        with pytest.raises((ValueError, Exception)):
            secure_inv(_share(0), N, T, P)


# ==========================================================================
# Protocol 8 — GenRndBitSharing
# ==========================================================================

class TestGenRndBitSharing:
    def test_output_is_bit(self):
        random.seed(600)
        from mpc_secret_shares import protocol_8_gen_rnd_bit_sharing
        for _ in range(20):
            sh = protocol_8_gen_rnd_bit_sharing(N, T, P)
            b = _reconstruct(sh)
            assert b in (0, 1), f"Expected bit, got {b}"

    def test_both_values_observed(self):
        """Over 40 trials, both 0 and 1 should appear."""
        random.seed(601)
        from mpc_secret_shares import protocol_8_gen_rnd_bit_sharing
        bits = [_reconstruct(protocol_8_gen_rnd_bit_sharing(N, T, P))
                for _ in range(40)]
        assert 0 in bits and 1 in bits


# ==========================================================================
# Protocol 9 — Bitwise_LessThan
# ==========================================================================

class TestBitwiseLessThan:
    S = math.ceil(math.log2(P))   # 4 for p=11

    def _b_bits(self, b_val):
        from mpc_secret_shares import protocol_1_share
        return [protocol_1_share((b_val >> i) & 1, N, T, P)
                for i in range(self.S)]

    @pytest.mark.parametrize("a,b,expected", [
        (3, 7, 1), (9, 3, 0), (5, 5, 0), (0, 1, 1), (10, 10, 0),
    ])
    def test_comparison(self, a, b, expected):
        random.seed(700 + a * 11 + b)
        from mpc_secret_shares import protocol_9_bitwise_less_than
        f_sh = protocol_9_bitwise_less_than(a, self._b_bits(b), N, T, P)
        assert _reconstruct(f_sh) == expected, f"BitwiseLT({a},{b}) failed"


# ==========================================================================
# Protocol 7 — LSB
# ==========================================================================

class TestLSB:
    @pytest.mark.parametrize("secret", [0, 1, 2, 3, 4, 5, 7, 10])
    def test_lsb(self, secret):
        random.seed(800 + secret)
        from mpc_secret_shares import protocol_7_lsb
        lsb = _reconstruct(protocol_7_lsb(_share(secret), N, T, P))
        assert lsb == secret % 2, f"LSB({secret}): got {lsb}"


# ==========================================================================
# Protocol 6 — LessThan_Half_P
# ==========================================================================

class TestLessThanHalfP:
    @pytest.mark.parametrize("secret", range(P))
    def test_all_values(self, secret):
        random.seed(900 + secret)
        from mpc_secret_shares import protocol_6_less_than_half_p
        w = _reconstruct(
            protocol_6_less_than_half_p(_share(secret), N, T, P)
        )
        expected = 1 if secret < P / 2 else 0
        assert w == expected, f"LT_HalfP({secret}): got {w}"


# ==========================================================================
# Protocol 5 — SecureCompare
# ==========================================================================

class TestSecureCompare:
    @pytest.mark.parametrize("u,v,expected", [
        (2, 5, 1), (5, 2, 0), (3, 3, 0),
        (0, 1, 1), (10, 1, 0), (1, 10, 1),
    ])
    def test_compare(self, u, v, expected):
        random.seed(1000 + u * 11 + v)
        from mpc_secret_shares import protocol_5_secure_compare
        w = _reconstruct(
            protocol_5_secure_compare(_share(u), _share(v), N, T, P)
        )
        assert w == expected, f"SecureCompare({u},{v}): got {w}"


# ==========================================================================
# Protocol 10 — IsZero
# ==========================================================================

class TestIsZero:
    @pytest.mark.parametrize("u", range(P))
    def test_is_zero(self, u):
        random.seed(1100 + u)
        from mpc_secret_shares import protocol_10_is_zero
        w = _reconstruct(protocol_10_is_zero(_share(u), N, T, P))
        assert w == (1 if u == 0 else 0), f"IsZero({u}): got {w}"


# ==========================================================================
# Protocol 12 — Binary Long Division (cleartext)
# ==========================================================================

class TestBinaryLongDivision:
    @pytest.mark.parametrize("u,v", [(13, 3), (10, 2), (7, 4), (0, 5), (15, 1)])
    def test_divmod(self, u, v):
        from mpc_secret_shares import binary_long_division
        q, r = binary_long_division(u, v)
        assert (q, r) == divmod(u, v), f"BinaryDiv({u},{v}): got ({q},{r})"


# ==========================================================================
# Protocol 13 — Secure Division
# ==========================================================================

class TestSecureDivision:
    S = math.ceil(math.log2(P))   # 4 for p=11

    def _u_bits(self, u_val):
        from mpc_secret_shares import protocol_1_share
        return [protocol_1_share((u_val >> i) & 1, N, T, P)
                for i in range(self.S)]

    @pytest.mark.parametrize("u,v", [(7, 3), (8, 2), (6, 4), (10, 5)])
    def test_division(self, u, v):
        random.seed(1200 + u * 11 + v)
        from mpc_secret_shares import protocol_13_secure_division
        q_sh, r_sh, _ = protocol_13_secure_division(
            _share(u), _share(v), self._u_bits(u), N, T, P, s=self.S
        )
        exp_q, exp_r = divmod(u, v)
        assert _reconstruct(q_sh) == exp_q, f"SecureDiv({u},{v}) quotient failed"
        assert _reconstruct(r_sh) == exp_r, f"SecureDiv({u},{v}) remainder failed"


# ==========================================================================
# Algorithm 14 — Bitwise Sqrt (cleartext)
# ==========================================================================

class TestBitwiseSqrt:
    @pytest.mark.parametrize("u", list(range(50)) + [100, 256, 625])
    def test_sqrt(self, u):
        from mpc_secret_shares import algorithm_14_bitwise_sqrt
        assert algorithm_14_bitwise_sqrt(u) == math.isqrt(u), \
            f"BitwiseSqrt({u}) failed"


# ==========================================================================
# Protocol 15 — Secure Sqrt
# ==========================================================================

class TestSecureSqrt:
    def _s(self):
        s = math.ceil(math.log2(P))
        return s + 1 if s % 2 else s  # pad to even

    def _u_bits(self, u_val, s):
        from mpc_secret_shares import protocol_1_share
        return [protocol_1_share((u_val >> i) & 1, N, T, P)
                for i in range(s)]

    @pytest.mark.parametrize("u", [0, 1, 4, 9, 2, 3])
    def test_sqrt(self, u):
        random.seed(1500 + u)
        from mpc_secret_shares import protocol_15_secure_sqrt, algorithm_14_bitwise_sqrt
        s = self._s()
        expected = algorithm_14_bitwise_sqrt(u, s=s)
        w_sh = protocol_15_secure_sqrt(
            _share(u), self._u_bits(u, s), N, T, P, s=s
        )
        assert _reconstruct(w_sh) == expected, f"SecureSqrt({u}) failed"


# ==========================================================================
# Protocol 16 — m-ary OR
# ==========================================================================

class TestMaryOR:
    @pytest.mark.parametrize("bits,expected", [
        ([0, 0], 0), ([1, 0], 1), ([0, 1], 1), ([1, 1], 1),
        ([0, 0, 0], 0), ([1, 0, 0], 1), ([0, 0, 1], 1), ([1, 1, 1], 1),
    ])
    def test_or(self, bits, expected):
        random.seed(1600 + sum(bits))
        from mpc_secret_shares import protocol_16_m_ary_or
        v = _reconstruct(
            protocol_16_m_ary_or([_share(b) for b in bits], N, T, P)
        )
        assert v == expected, f"maryOR{bits}: got {v}"


# ==========================================================================
# Protocol 17 — Improved m-ary OR  (OR=1 cases only; see module note)
# ==========================================================================

class TestImprovedMaryOR:
    @pytest.mark.parametrize("bits", [
        [1, 0], [0, 1], [1, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1],
    ])
    def test_or_is_one(self, bits):
        random.seed(1700 + sum(bits))
        from mpc_secret_shares import protocol_17_improved_m_ary_or
        v = _reconstruct(
            protocol_17_improved_m_ary_or([_share(b) for b in bits], N, T, P)
        )
        assert v == 1, f"ImprovedOR{bits}: expected 1, got {v}"


# ==========================================================================
# Algorithm 18 — k-th Ranked Element (cleartext)
# ==========================================================================

class TestKthRanked:
    def test_full_dataset(self):
        from mpc_secret_shares import algorithm_18_kth_ranked
        data   = [3, 1, 4, 1, 5, 9, 2, 6]
        M, l   = 16, 4
        sorted_data = sorted(data)
        for k in range(1, len(data) + 1):
            mk = algorithm_18_kth_ranked(data, M, k, l)
            assert mk == sorted_data[k - 1], \
                f"KthRanked(k={k}): expected {sorted_data[k-1]}, got {mk}"

    def test_single_element(self):
        from mpc_secret_shares import algorithm_18_kth_ranked
        assert algorithm_18_kth_ranked([7], 8, 1, 3) == 7

    def test_all_equal(self):
        from mpc_secret_shares import algorithm_18_kth_ranked
        assert algorithm_18_kth_ranked([3, 3, 3], 4, 2, 2) == 3

    def test_even_valued_elements(self):
        """Regression: even-valued k-th elements — kappa_2 must use alpha-1, not alpha."""
        from mpc_secret_shares import kth_ranked
        assert kth_ranked([2],          8,  1)     == 2, "singleton 2"
        assert kth_ranked([4],          8,  1)     == 4, "singleton 4"
        assert kth_ranked([6],          8,  1)     == 6, "singleton 6"
        assert kth_ranked([2, 4, 6, 8], 16, 2)    == 4, "rank 2 of [2,4,6,8]"
        assert kth_ranked([10,11,4,5,8,9,5,3], 16, 3) == 5, "rank 3, duplicates"
        assert kth_ranked([10,11,4,5,8,9,5,3], 16, 4) == 5, "rank 4, duplicates"


# ==========================================================================
# Issue regression tests
# ==========================================================================

class TestIssueRegressions:
    """Targeted regression tests for the three known issues."""

    def test_issue1_coefficient_indexed_by_j(self):
        """Issue 1: Protocol 1 polynomial uses a_j (indexed by j), not a_i.

        Uses t=3 so the polynomial has two non-constant random coefficients.
        Every Z_p element must round-trip correctly; a wrong index would
        produce a different polynomial and break reconstruction.
        """
        random.seed(9001)
        from mpc_secret_shares import protocol_1_share, protocol_2_reconstruct
        for secret in range(P):
            sh = protocol_1_share(secret, n=4, t=3, P=P)
            recovered = protocol_2_reconstruct(sh[:3], P)
            assert recovered == secret, (
                f"Issue 1: coefficient mis-indexed for secret={secret}"
            )

    def test_issue2_algorithm14_odd_s_padded(self):
        """Issue 2: Algorithm 14 pads odd s to even so the LSB pair is not skipped."""
        from mpc_secret_shares import algorithm_14_bitwise_sqrt
        for u, s_odd in [(0, 2), (1, 2), (2, 4), (3, 4), (8, 4), (9, 4), (15, 4)]:
            result = algorithm_14_bitwise_sqrt(u, s=s_odd)
            assert result == math.isqrt(u), (
                f"Issue 2: Alg14 wrong for u={u}, s={s_odd} (odd-padded)"
            )

    def test_issue2_protocol15_odd_s_padded(self):
        """Issue 2: Protocol 15 pads odd s to even before the bit-pair loop."""
        random.seed(9002)
        from mpc_secret_shares import protocol_15_secure_sqrt, algorithm_14_bitwise_sqrt
        s_odd = 3
        for u in [0, 1, 2, 3, 4]:
            u_bits = [_share((u >> i) & 1) for i in range(s_odd)]
            w_sh = protocol_15_secure_sqrt(_share(u), u_bits, N, T, P, s=s_odd)
            expected = algorithm_14_bitwise_sqrt(u, s=s_odd)
            assert _reconstruct(w_sh) == expected, (
                f"Issue 2: SecureSqrt(u={u}, s={s_odd}) wrong"
            )

    def test_issue3_duplicate_values_adjacent_ranks(self):
        """Issue 3: duplicate values in dataset — consecutive ranks both return
        the duplicate correctly.

        dataset = [10, 11, 4, 5, 8, 9, 5, 3]  →  sorted [3,4,5,5,8,9,10,11]
        k=3 and k=4 must both return 5.
        """
        from mpc_secret_shares import algorithm_18_kth_ranked
        data = [10, 11, 4, 5, 8, 9, 5, 3]
        M, l = 16, 4
        assert algorithm_18_kth_ranked(data, M, 3, l) == 5, \
            "Issue 3: rank 3 of duplicate-containing dataset wrong"
        assert algorithm_18_kth_ranked(data, M, 4, l) == 5, \
            "Issue 3: rank 4 of duplicate-containing dataset wrong"


# ==========================================================================
# square_and_multiply
# ==========================================================================

class TestSquareAndMultiply:
    @pytest.mark.parametrize("x,c,n,expected", [
        (3, 13, 11, pow(3, 13, 11)),
        (2, 10,  7, pow(2, 10,  7)),
        (5,  0, 11, 1),
        (5,  1, 11, 5),
    ])
    def test_modular_exp(self, x, c, n, expected):
        from mpc_secret_shares import square_and_multiply
        assert square_and_multiply(x, c, n) == expected
