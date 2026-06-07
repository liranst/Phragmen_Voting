"""Verify that run_election() invokes all 8 algorithms the correct number
of times across a 2-period election.

Each algorithm is patched at the point where run_election imports it
(protocol.run_election.<name>) using wraps= so the real implementation
still executes and produces valid outputs for downstream algorithms.
"""

from unittest.mock import patch

import pytest

from protocol.algorithm1_permutation import (
    algorithm_1_oblivious_candidate_permutation,
)
from protocol.algorithm2_validation import algorithm_2_input_validation
from protocol.algorithm3_initialization import algorithm_3_initialization
from protocol.algorithm4_score import algorithm_4_score_computation
from protocol.algorithm5_find_min import algorithm_5_find_minimum_score
from protocol.algorithm6_one_hot import algorithm_6_one_hot_winner
from protocol.algorithm7_wallet_update import algorithm_7_wallet_update
from protocol.algorithm8_reconstruct_winner import algorithm_8_reconstruct_winner
from protocol.run_election import run_election

BALLOT_PERIODS = [
    [[1, 0], [1, 0], [0, 1]],   # period 1
    [[0, 1], [1, 0], [0, 1]],   # period 2: voter 0 switches
]
P = 251


class TestRunElectionCallsAllAlgorithms:
    """run_election() must delegate to every algorithm the expected number
    of times for a 2-period election."""

    def test_all_algorithms_called_correct_number_of_times(self):
        _base = "protocol.run_election"

        with (
            patch(f"{_base}.algorithm_1_oblivious_candidate_permutation",
                  wraps=algorithm_1_oblivious_candidate_permutation) as m1,
            patch(f"{_base}.algorithm_2_input_validation",
                  wraps=algorithm_2_input_validation) as m2,
            patch(f"{_base}.algorithm_3_initialization",
                  wraps=algorithm_3_initialization) as m3,
            patch(f"{_base}.algorithm_4_score_computation",
                  wraps=algorithm_4_score_computation) as m4,
            patch(f"{_base}.algorithm_5_find_minimum_score",
                  wraps=algorithm_5_find_minimum_score) as m5,
            patch(f"{_base}.algorithm_6_one_hot_winner",
                  wraps=algorithm_6_one_hot_winner) as m6,
            patch(f"{_base}.algorithm_7_wallet_update",
                  wraps=algorithm_7_wallet_update) as m7,
            patch(f"{_base}.algorithm_8_reconstruct_winner",
                  wraps=algorithm_8_reconstruct_winner) as m8,
        ):
            winners = run_election(BALLOT_PERIODS, p=P)

        # Sanity-check the election produced two winners
        assert len(winners) == 2

        # Algorithm 1: permutation generated once at setup
        assert m1.call_count == 1, (
            f"Algorithm 1 expected 1 call, got {m1.call_count}"
        )
        # Algorithm 2: ballot validation runs every period
        assert m2.call_count == 2, (
            f"Algorithm 2 expected 2 calls, got {m2.call_count}"
        )
        # Algorithm 3: initialisation runs once (first period only)
        assert m3.call_count == 1, (
            f"Algorithm 3 expected 1 call, got {m3.call_count}"
        )
        # Algorithms 4–8: per-period loop, 2 periods → 2 calls each
        assert m4.call_count == 2, (
            f"Algorithm 4 expected 2 calls, got {m4.call_count}"
        )
        assert m5.call_count == 2, (
            f"Algorithm 5 expected 2 calls, got {m5.call_count}"
        )
        assert m6.call_count == 2, (
            f"Algorithm 6 expected 2 calls, got {m6.call_count}"
        )
        assert m7.call_count == 2, (
            f"Algorithm 7 expected 2 calls, got {m7.call_count}"
        )
        assert m8.call_count == 2, (
            f"Algorithm 8 expected 2 calls, got {m8.call_count}"
        )
