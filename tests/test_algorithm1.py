"""Tests for Algorithm 1 — Oblivious Candidate Permutation.

Reference: Algorithm 1 in "Fairness Without Exposure: Privacy-Preserving
Phragmén Voting", Section 5.1.

Test parameters: security_bits=128 (16-byte seeds), varied M and D.
"""

import hashlib
import pytest

from protocol.algorithm1_permutation import (
    _commitment,
    _fisher_yates,
    _verify_commitments,
    algorithm_1_oblivious_candidate_permutation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SECURITY_BITS = 128
_SEED_LEN = _SECURITY_BITS // 8


def _make_commitments(seeds: list[bytes]) -> list[bytes]:
    """Compute commitments for a list of seeds (zero-indexed parties)."""
    return [hashlib.sha256(s + d.to_bytes(4, "big")).digest()
            for d, s in enumerate(seeds)]


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

class TestOutputStructure:
    def test_m1_trivial(self):
        """M=1 → only possible permutation is [0]."""
        pi = algorithm_1_oblivious_candidate_permutation(
            _SECURITY_BITS, 1, 3
        )
        assert pi == [0]

    def test_output_is_valid_permutation(self):
        """Output contains every index in [0, M) exactly once."""
        pi = algorithm_1_oblivious_candidate_permutation(
            _SECURITY_BITS, 7, 3
        )
        assert sorted(pi) == list(range(7))

    def test_m2_valid_permutation(self):
        """M=2 → result is [0,1] or [1,0]."""
        pi = algorithm_1_oblivious_candidate_permutation(
            _SECURITY_BITS, 2, 3
        )
        assert sorted(pi) == [0, 1]
        assert len(pi) == 2

    def test_large_m(self):
        """M=100 → output is a valid permutation."""
        pi = algorithm_1_oblivious_candidate_permutation(
            _SECURITY_BITS, 100, 5
        )
        assert sorted(pi) == list(range(100))


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    _SEEDS = [bytes([i + 1]) * _SEED_LEN for i in range(3)]

    def test_same_seeds_same_permutation(self):
        """Given identical seeds both calls must return the same permutation."""
        pi1 = algorithm_1_oblivious_candidate_permutation(
            _SECURITY_BITS, 5, 3, seeds=self._SEEDS
        )
        pi2 = algorithm_1_oblivious_candidate_permutation(
            _SECURITY_BITS, 5, 3, seeds=self._SEEDS
        )
        assert pi1 == pi2

    def test_different_seeds_different_permutation(self):
        """Different seeds must (with overwhelming probability) yield a
        different permutation for large M."""
        seeds_a = [bytes([1]) * _SEED_LEN, bytes([2]) * _SEED_LEN,
                   bytes([3]) * _SEED_LEN]
        seeds_b = [bytes([4]) * _SEED_LEN, bytes([5]) * _SEED_LEN,
                   bytes([6]) * _SEED_LEN]
        pi_a = algorithm_1_oblivious_candidate_permutation(
            _SECURITY_BITS, 10, 3, seeds=seeds_a
        )
        pi_b = algorithm_1_oblivious_candidate_permutation(
            _SECURITY_BITS, 10, 3, seeds=seeds_b
        )
        # P(collision) = 1/10! ≈ 2.8e-7 — negligible
        assert pi_a != pi_b

    def test_xor_of_identical_seeds_valid(self):
        """All seeds equal → combined XOR = 0x00…; output still a valid permutation."""
        seed = bytes([0xAB]) * _SEED_LEN
        seeds = [seed] * 5
        pi = algorithm_1_oblivious_candidate_permutation(
            _SECURITY_BITS, 6, 5, seeds=seeds
        )
        assert sorted(pi) == list(range(6))


# ---------------------------------------------------------------------------
# Party-count edge cases
# ---------------------------------------------------------------------------

class TestPartyCount:
    def test_d1_single_party(self):
        """D=1 → no XOR fold; permutation derived from sole party's seed."""
        seeds = [bytes([0x42]) * _SEED_LEN]
        pi = algorithm_1_oblivious_candidate_permutation(
            _SECURITY_BITS, 4, 1, seeds=seeds
        )
        assert sorted(pi) == [0, 1, 2, 3]

    def test_d1_deterministic(self):
        """D=1, same seed → same permutation on two calls."""
        seeds = [bytes([0x99]) * _SEED_LEN]
        pi1 = algorithm_1_oblivious_candidate_permutation(
            _SECURITY_BITS, 5, 1, seeds=seeds
        )
        pi2 = algorithm_1_oblivious_candidate_permutation(
            _SECURITY_BITS, 5, 1, seeds=seeds
        )
        assert pi1 == pi2

    def test_d2_valid(self):
        """D=2 parties produce a valid permutation."""
        seeds = [bytes([0x11]) * _SEED_LEN, bytes([0x22]) * _SEED_LEN]
        pi = algorithm_1_oblivious_candidate_permutation(
            _SECURITY_BITS, 5, 2, seeds=seeds
        )
        assert sorted(pi) == list(range(5))


# ---------------------------------------------------------------------------
# Commitment verification
# ---------------------------------------------------------------------------

class TestCommitmentVerification:
    def test_tampered_seed_raises(self):
        """A seed that does not match its commitment must raise ValueError."""
        original = [bytes([0x01]) * _SEED_LEN,
                    bytes([0x02]) * _SEED_LEN,
                    bytes([0x03]) * _SEED_LEN]
        commitments = _make_commitments(original)
        tampered = [bytes([0xFF]) * _SEED_LEN,  # party 0 cheats
                    bytes([0x02]) * _SEED_LEN,
                    bytes([0x03]) * _SEED_LEN]
        with pytest.raises(ValueError, match="commitment mismatch"):
            _verify_commitments(tampered, commitments)

    def test_correct_seeds_no_raise(self):
        """Correct seeds must pass verification without error."""
        seeds = [bytes([d + 1]) * _SEED_LEN for d in range(4)]
        commitments = _make_commitments(seeds)
        _verify_commitments(seeds, commitments)  # must not raise

    def test_commitment_helper_matches_sha256(self):
        """_commitment must equal SHA-256(seed ‖ party_index_big_endian)."""
        seed = bytes([0xDE, 0xAD]) * (_SEED_LEN // 2)
        d = 7
        expected = hashlib.sha256(seed + d.to_bytes(4, "big")).digest()
        assert _commitment(seed, d) == expected


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_wrong_number_of_seeds_raises(self):
        """Passing 2 seeds when num_parties=3 must raise ValueError."""
        seeds = [bytes([0x01]) * _SEED_LEN, bytes([0x02]) * _SEED_LEN]
        with pytest.raises(ValueError):
            algorithm_1_oblivious_candidate_permutation(
                _SECURITY_BITS, 5, 3, seeds=seeds
            )

    def test_wrong_seed_length_raises(self):
        """Seed with wrong byte-length must raise ValueError."""
        seeds = [bytes([0x01]) * 8,           # too short
                 bytes([0x02]) * _SEED_LEN,
                 bytes([0x03]) * _SEED_LEN]
        with pytest.raises(ValueError):
            algorithm_1_oblivious_candidate_permutation(
                _SECURITY_BITS, 5, 3, seeds=seeds
            )


# ---------------------------------------------------------------------------
# Fisher-Yates helper
# ---------------------------------------------------------------------------

class TestFisherYates:
    def test_empty_candidates(self):
        """M=0 → empty permutation."""
        assert _fisher_yates(b"\x00" * _SEED_LEN, 0) == []

    def test_m1(self):
        """M=1 → only element is 0."""
        assert _fisher_yates(b"\xAB" * _SEED_LEN, 1) == [0]

    def test_deterministic(self):
        """Same combined seed → same permutation."""
        seed = bytes(range(_SEED_LEN))
        assert _fisher_yates(seed, 8) == _fisher_yates(seed, 8)

    def test_valid_permutation(self):
        """Output is always a valid permutation."""
        seed = bytes([0x55]) * _SEED_LEN
        pi = _fisher_yates(seed, 12)
        assert sorted(pi) == list(range(12))
