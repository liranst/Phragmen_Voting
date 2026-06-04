"""Algorithm 1 — Oblivious Candidate Permutation.

Reference: Algorithm 1 in "Fairness Without Exposure: Privacy-Preserving
Phragmén Voting", Section 5.1 (page 8).  Run once at setup, before any
election period.  No MPC secure operations are required.

Each tallying party P_d (d ∈ [D]):
  1. Samples a secret random seed s_d ← {0,1}^λ.
  2. Broadcasts a commitment com_d = SHA-256(s_d ‖ d).
  3. Reveals s_d; all other parties verify the commitment.

The shared seed is s = s_1 ⊕ … ⊕ s_D, and the public permutation is
derived by running a Fisher-Yates shuffle seeded with s.
"""

import hashlib
import os
import random
from typing import List, Optional


# ---------------------------------------------------------------------------
# Internal helpers (exposed for unit-testing)
# ---------------------------------------------------------------------------

def _commitment(seed: bytes, party_index: int) -> bytes:
    """Return SHA-256(seed ‖ party_index) as defined in Algorithm 1, Line 3.

    Args:
        seed: The party's random seed bytes.
        party_index: Zero-based party index, encoded as a big-endian 4-byte
            unsigned integer before hashing.

    Returns:
        32-byte SHA-256 digest.
    """
    return hashlib.sha256(seed + party_index.to_bytes(4, "big")).digest()


def _verify_commitments(seeds: List[bytes], commitments: List[bytes]) -> None:
    """Verify every revealed seed against its published commitment.

    Implements the verification step in Algorithm 1, Lines 6–8.

    Args:
        seeds: Revealed seeds, one per party (zero-indexed).
        commitments: Published commitments in the same order.

    Raises:
        ValueError: If any seed does not reproduce its commitment.
    """
    for d, (seed, com) in enumerate(zip(seeds, commitments)):
        if _commitment(seed, d) != com:
            raise ValueError(
                f"commitment mismatch for party {d}: "
                "revealed seed does not match published commitment"
            )


def _fisher_yates(combined_seed: bytes, num_candidates: int) -> List[int]:
    """Produce a uniformly random permutation of [0, M) via Fisher-Yates.

    Implements Algorithm 1, Line 10.  The combined seed is converted to an
    integer and used to seed Python's Mersenne-Twister PRNG, which then
    drives `random.shuffle` (a Fisher-Yates implementation).

    Args:
        combined_seed: XOR of all party seeds (Algorithm 1, Line 9).
        num_candidates: M — size of the candidate set.

    Returns:
        A permutation of [0, M) as a list of length M.
    """
    seed_int = int.from_bytes(combined_seed, "big")
    rng = random.Random(seed_int)
    pi = list(range(num_candidates))
    rng.shuffle(pi)
    return pi


# ---------------------------------------------------------------------------
# Algorithm 1
# ---------------------------------------------------------------------------

def algorithm_1_oblivious_candidate_permutation(
    security_bits: int,
    num_candidates: int,
    num_parties: int,
    seeds: Optional[List[bytes]] = None,
) -> List[int]:
    """Generate a public uniformly random permutation of candidates.

    Algorithm 1 of "Fairness Without Exposure: Privacy-Preserving Phragmén
    Voting", Section 5.1.

    Each party commits to a random seed via SHA-256, then reveals it.  The
    combined (XOR) seed drives a Fisher-Yates shuffle that produces the
    oblivious permutation π used by Algorithm 6 for tie-breaking.

    Args:
        security_bits: λ — bit-length of each party's random seed.
            Must be a positive multiple of 8.
        num_candidates: M — number of candidates to permute.
        num_parties: D (= n in MPC notation) — number of tallying parties.
        seeds: Optional list of D byte-strings each of length
            ``security_bits // 8``.  When None every party samples a fresh
            random seed.  Supply explicit seeds only in tests to obtain
            deterministic output.

    Returns:
        A permutation π of [0, M) as a list of M distinct integers.

    Raises:
        ValueError: If ``seeds`` has wrong length or wrong element sizes.
        ValueError: If any revealed seed does not match its commitment
            (simulates an abort on cheating during the reveal phase).
    """
    seed_len = security_bits // 8

    # Lines 1-3: every party samples a seed and publishes a commitment.
    if seeds is None:
        seeds = [os.urandom(seed_len) for _ in range(num_parties)]
    else:
        if len(seeds) != num_parties:
            raise ValueError(
                f"expected {num_parties} seeds, got {len(seeds)}"
            )
        for d, s in enumerate(seeds):
            if len(s) != seed_len:
                raise ValueError(
                    f"seed for party {d} has length {len(s)}, "
                    f"expected {seed_len} (security_bits={security_bits})"
                )

    commitments = [_commitment(s, d) for d, s in enumerate(seeds)]  # Line 3

    # Lines 4-8: reveal phase — every party broadcasts its seed and all
    # others verify it against the stored commitment.
    _verify_commitments(seeds, commitments)  # Lines 6-8

    # Line 9: derive the shared seed by XOR-ing all party seeds.
    combined = seeds[0]
    for s in seeds[1:]:
        combined = bytes(a ^ b for a, b in zip(combined, s))

    # Lines 10-11: Fisher-Yates shuffle seeded with the combined seed.
    return _fisher_yates(combined, num_candidates)
