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
# Required audit tests
# ---------------------------------------------------------------------------

class TestValidPermutation:
    """Property 1: output is always a valid permutation of [0, M-1]."""

    @pytest.mark.parametrize("m", [2, 3, 5, 10, 20])
    def test_valid_permutation_for_required_sizes(self, m):
        """M ∈ {2, 3, 5, 10, 20} all produce a permutation of [0, M)."""
        pi = algorithm_1_oblivious_candidate_permutation(
            _SECURITY_BITS, m, 3
        )
        assert len(pi) == m
        assert sorted(pi) == list(range(m))

    @pytest.mark.parametrize("m", [2, 3, 5, 10, 20])
    def test_no_duplicates(self, m):
        """No candidate index appears more than once."""
        pi = algorithm_1_oblivious_candidate_permutation(
            _SECURITY_BITS, m, 3
        )
        assert len(set(pi)) == m

    @pytest.mark.parametrize("m", [2, 3, 5, 10, 20])
    def test_all_indices_present(self, m):
        """Every index in [0, M) is present exactly once."""
        pi = algorithm_1_oblivious_candidate_permutation(
            _SECURITY_BITS, m, 3
        )
        assert set(pi) == set(range(m))


class TestDifferentSeedsDifferentPermutations:
    """Property 2: different seed sets produce different permutations."""

    def test_two_independent_random_draws_differ(self):
        """Two independent fresh runs (seeds=None) almost certainly differ
        for large M.  P(collision for M=20) = 1/(20!) ≈ 4e-19."""
        pi_a = algorithm_1_oblivious_candidate_permutation(
            _SECURITY_BITS, 20, 3
        )
        pi_b = algorithm_1_oblivious_candidate_permutation(
            _SECURITY_BITS, 20, 3
        )
        assert pi_a != pi_b

    def test_explicit_different_seeds_differ(self):
        """Swapping every seed byte produces a different permutation."""
        seeds_a = [bytes([0x11]) * _SEED_LEN,
                   bytes([0x22]) * _SEED_LEN,
                   bytes([0x33]) * _SEED_LEN]
        seeds_b = [bytes([0xAA]) * _SEED_LEN,
                   bytes([0xBB]) * _SEED_LEN,
                   bytes([0xCC]) * _SEED_LEN]
        pi_a = algorithm_1_oblivious_candidate_permutation(
            _SECURITY_BITS, 10, 3, seeds=seeds_a
        )
        pi_b = algorithm_1_oblivious_candidate_permutation(
            _SECURITY_BITS, 10, 3, seeds=seeds_b
        )
        assert pi_a != pi_b


class TestSameSeedsSamePermutation:
    """Property 3: same seeds always reproduce the same permutation."""

    @pytest.mark.parametrize("m", [2, 3, 5, 10, 20])
    def test_deterministic_across_calls(self, m):
        """Three repeated calls with the same seeds all agree."""
        seeds = [bytes([d + 1]) * _SEED_LEN for d in range(3)]
        results = [
            algorithm_1_oblivious_candidate_permutation(
                _SECURITY_BITS, m, 3, seeds=seeds
            )
            for _ in range(3)
        ]
        assert results[0] == results[1] == results[2]


class TestSinglePartyCannotControl:
    """Property 4: changing any one party's seed changes the permutation,
    so no single party can predetermine the outcome.

    Each sub-test fixes all parties' seeds except one and shows that
    flipping that party's seed (almost certainly) changes π.
    """

    _BASE_SEEDS = [
        bytes([0x01]) * _SEED_LEN,
        bytes([0x02]) * _SEED_LEN,
        bytes([0x03]) * _SEED_LEN,
    ]

    def _run(self, seeds):
        return algorithm_1_oblivious_candidate_permutation(
            _SECURITY_BITS, 10, 3, seeds=seeds
        )

    def test_changing_party_0_seed_changes_permutation(self):
        """Flipping party 0's seed (parties 1, 2 fixed) changes π."""
        pi_orig = self._run(self._BASE_SEEDS)
        altered = [bytes([0xFF]) * _SEED_LEN,
                   self._BASE_SEEDS[1],
                   self._BASE_SEEDS[2]]
        pi_alt = self._run(altered)
        assert pi_orig != pi_alt

    def test_changing_party_1_seed_changes_permutation(self):
        """Flipping party 1's seed (parties 0, 2 fixed) changes π."""
        pi_orig = self._run(self._BASE_SEEDS)
        altered = [self._BASE_SEEDS[0],
                   bytes([0xFF]) * _SEED_LEN,
                   self._BASE_SEEDS[2]]
        pi_alt = self._run(altered)
        assert pi_orig != pi_alt

    def test_changing_party_2_seed_changes_permutation(self):
        """Flipping party 2's seed (parties 0, 1 fixed) changes π."""
        pi_orig = self._run(self._BASE_SEEDS)
        altered = [self._BASE_SEEDS[0],
                   self._BASE_SEEDS[1],
                   bytes([0xFF]) * _SEED_LEN]
        pi_alt = self._run(altered)
        assert pi_orig != pi_alt

    def test_xor_means_every_party_contributes(self):
        """The combined seed changes whenever any one party changes theirs:
        XOR(s1, s2, s3) ≠ XOR(s1', s2, s3) whenever s1 ≠ s1'."""
        s1 = bytes([0x01]) * _SEED_LEN
        s2 = bytes([0x02]) * _SEED_LEN
        s3 = bytes([0x03]) * _SEED_LEN
        s1_alt = bytes([0x04]) * _SEED_LEN   # differs in every bit from s1

        combined_orig = bytes(a ^ b ^ c for a, b, c in zip(s1, s2, s3))
        combined_alt  = bytes(a ^ b ^ c for a, b, c in zip(s1_alt, s2, s3))
        assert combined_orig != combined_alt


class TestWorksForRequiredSizes:
    """Property 5: algorithm works correctly for M in {2, 3, 5, 10, 20}."""

    @pytest.mark.parametrize("m", [2, 3, 5, 10, 20])
    def test_output_length(self, m):
        """Output list has exactly M elements."""
        pi = algorithm_1_oblivious_candidate_permutation(
            _SECURITY_BITS, m, 3
        )
        assert len(pi) == m

    @pytest.mark.parametrize("m", [2, 3, 5, 10, 20])
    def test_valid_permutation(self, m):
        """sorted(pi) == [0, 1, …, M-1] for every required M."""
        seeds = [bytes([m + d]) * _SEED_LEN for d in range(3)]
        pi = algorithm_1_oblivious_candidate_permutation(
            _SECURITY_BITS, m, 3, seeds=seeds
        )
        assert sorted(pi) == list(range(m))

    @pytest.mark.parametrize("m", [2, 3, 5, 10, 20])
    def test_reproducible(self, m):
        """Two calls with the same seeds agree for every required M."""
        seeds = [bytes([0xDE + d]) * _SEED_LEN for d in range(3)]
        pi1 = algorithm_1_oblivious_candidate_permutation(
            _SECURITY_BITS, m, 3, seeds=seeds
        )
        pi2 = algorithm_1_oblivious_candidate_permutation(
            _SECURITY_BITS, m, 3, seeds=seeds
        )
        assert pi1 == pi2


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
