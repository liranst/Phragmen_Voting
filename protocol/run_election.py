"""High-level runner for the PP-Phragmén protocol.

Orchestrates Algorithms 1–8 of "Fairness Without Exposure: Privacy-Preserving
Phragmén Voting" across one or more election periods.  Algorithm 1 (permutation)
and Algorithm 3 (initialization) run once before the first period; Algorithms
2 and 4–8 run every period.
"""

import random
from typing import List

from mpc_secret_shares import share

from protocol.algorithm1_permutation import algorithm_1_oblivious_candidate_permutation
from protocol.algorithm2_validation import algorithm_2_input_validation
from protocol.algorithm3_initialization import algorithm_3_initialization
from protocol.algorithm4_score import algorithm_4_score_computation
from protocol.algorithm5_find_min import algorithm_5_find_minimum_score
from protocol.algorithm6_one_hot import algorithm_6_one_hot_winner
from protocol.algorithm7_wallet_update import algorithm_7_wallet_update
from protocol.algorithm8_reconstruct_winner import algorithm_8_reconstruct_winner
from protocol.types import BallotMatrix


def run_election(
    ballot_periods: list[list[list[int]]],
    p: int,
    n_parties: int = 3,
    threshold: int = 2,
    security_bits: int = 128,
    seed: int = 42,
) -> list[int]:
    """Run PP-Phragmén for multiple election periods.

    Each period has its own ballot matrix — voters may change their votes
    between periods.

    Args:
        ballot_periods: List of ballot matrices.
            ``ballot_periods[r][i][m] = 1`` if in period r, voter i approves
            candidate m.  All matrices must have the same shape [N][M].
        p: Prime field modulus.
        n_parties: Number of MPC tallying parties.
        threshold: Reconstruction threshold (< n_parties).
        security_bits: Security parameter for the Algorithm 1 permutation.
        seed: Random seed for reproducibility.

    Returns:
        List of winner indices, one per period.

    Raises:
        ValueError: If ``ballot_periods`` is empty.
        ValueError: If the number of honest voters (n_valid) changes between
            periods, which would make the persisted wallet state inconsistent.

    Example:
        ballot_periods = [
            [[1, 0], [1, 0], [0, 1]],   # period 1
            [[0, 1], [1, 0], [0, 1]],   # period 2: voter 0 changed
        ]
        winners = run_election(ballot_periods, p=251)
        # → [0, 1]
    """
    if not ballot_periods:
        raise ValueError("ballot_periods must contain at least one period")

    random.seed(seed)

    n_candidates: int = len(ballot_periods[0][0])

    # Algorithm 1: generate the oblivious candidate permutation once.
    # Each party contributes one seed; their XOR determines π.
    seeds: List[bytes] = [random.randbytes(security_bits // 8) for _ in range(n_parties)]
    pi: List[int] = algorithm_1_oblivious_candidate_permutation(
        security_bits, n_candidates, n_parties, seeds=seeds
    )

    delta = None
    wallets = None
    n_valid: int = 0
    winners: List[int] = []

    for r, ballots in enumerate(ballot_periods):

        # Algorithm 2: validate fresh ballot entries every period.
        B_shares: BallotMatrix = _make_ballot_shares(ballots, n_parties, threshold, p)
        B_hat, n_valid_r = algorithm_2_input_validation(B_shares, n_parties, threshold, p)

        if r == 0:
            n_valid = n_valid_r
            # Algorithm 3: initialise Δ = 1 and all W̃ᵢ = 0 (first period only).
            delta, wallets = algorithm_3_initialization(n_valid, n_parties, threshold, p)
        elif n_valid_r != n_valid:
            raise ValueError(
                f"Period {r + 1}: honest voter count changed from {n_valid} to "
                f"{n_valid_r}; the honest voter set must remain stable across periods "
                "so that the persisted wallet state stays consistent."
            )

        # Algorithm 4: compute per-candidate scores as shared fractions.
        scores, A_shares, T_shares = algorithm_4_score_computation(
            B_hat, wallets, delta, n_valid, n_candidates, n_parties, threshold, p
        )

        # Algorithm 5: find the minimum (winning) score.
        winning_score = algorithm_5_find_minimum_score(scores, n_parties, threshold, p)

        # Algorithm 6: produce a shared one-hot winner indicator with π tie-breaking.
        chi = algorithm_6_one_hot_winner(
            scores, winning_score, pi, n_parties, threshold, p
        )

        # Algorithm 8: publicly reveal the winner index.
        winner: int = algorithm_8_reconstruct_winner(chi, threshold, p)
        winners.append(winner)

        # Algorithm 7: update wallets and Δ for the next period.
        wallets, delta = algorithm_7_wallet_update(
            B_hat, chi, wallets, delta, A_shares, T_shares,
            n_valid, n_candidates, n_parties, threshold, p
        )

    return winners


def _make_ballot_shares(
    ballots: List[List[int]],
    n_parties: int,
    threshold: int,
    p: int,
) -> BallotMatrix:
    """Secret-share every entry of a plaintext ballot matrix."""
    return [[share(v, n_parties, threshold, p) for v in row] for row in ballots]
