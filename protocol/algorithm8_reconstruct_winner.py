"""Algorithm 8 — Winner Reconstruction.

Reference: Algorithm 8 in "Fairness Without Exposure: Privacy-Preserving
Phragmén Voting", Section 5.8 (page 14).  Run every election period as the
final step, after Algorithm 7 has updated the wallets.

Reconstructs the one-hot indicator {[[χ_m]]}_{m∈[M]} produced by Algorithm 6
into a publicly revealed winner index m*.  This is the only step in the
per-period loop that reveals information: the identity of the elected
candidate.  All ballots, wallets, and intermediate scores remain secret.

The two consistency checks (Lines 3–6) guard against upstream protocol
failures; they abort rather than silently return a wrong winner.

No MPC secure operations are performed — Reconstruct is local Lagrange
interpolation on already-held shares.
"""

from typing import List

from mpc_secret_shares import protocol_2_reconstruct as reconstruct
from protocol.types import Shares


def algorithm_8_reconstruct_winner(
    chi: List[Shares],
    t: int,
    p: int,
) -> int:
    """Reconstruct and reveal the winning candidate index from the one-hot indicator.

    Algorithm 8 of "Fairness Without Exposure: Privacy-Preserving Phragmén
    Voting", Section 5.8.

    Each sharing [[χ_m]] is reconstructed publicly using t shares.  The
    results must form a valid one-hot vector: every entry in {0, 1} and
    exactly one entry equal to 1.  Any deviation indicates an upstream
    protocol failure and causes an immediate abort.

    Args:
        chi: List of M (t,n)-sharings of χ_m ∈ {0, 1}, produced by
            Algorithm 6.  ``chi[m]`` corresponds to candidate m (0-indexed).
        t: Reconstruction threshold — the number of shares used for
            Lagrange interpolation.
        p: Prime field modulus.

    Returns:
        The 0-indexed winner m* — the unique m ∈ [0, M) with χ_m = 1.

    Raises:
        ValueError: If any χ_m ∉ {0, 1} (Lines 3–4 of the paper).
        ValueError: If Σ χ_m ≠ 1 — either no winner or multiple winners
            (Lines 5–6 of the paper).
        ValueError: If ``chi`` is empty.
    """
    if not chi:
        raise ValueError("chi must be non-empty: at least one candidate required")

    # Lines 1-4: reconstruct every χ_m publicly and validate each value.
    chi_plain: List[int] = []
    for m, shares in enumerate(chi):                                 # Line 1
        val: int = reconstruct(shares[:t], p)             # Line 2
        if val not in (0, 1):                                        # Line 3
            raise ValueError(
                f"χ_{m} = {val} is not in {{0, 1}}: "
                "one-hot vector is malformed (upstream protocol failure)"
            )                                                        # Line 4
        chi_plain.append(val)

    # Lines 5-6: verify exactly one candidate received χ = 1.
    total = sum(chi_plain)
    if total != 1:                                                   # Line 5
        raise ValueError(
            f"Σ χ_m = {total} ≠ 1: expected exactly one winner "
            f"(got {total})"
        )                                                            # Line 6

    # Lines 7-8: return the unique winner index.
    return chi_plain.index(1)                                        # Lines 7-8
