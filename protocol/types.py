"""Shared type aliases used across all PP-Phragmén protocol modules.

Reference: Section 5 of "Fairness Without Exposure: Privacy-Preserving
Phragmén Voting."
"""

from typing import List, Tuple

Shares = List[Tuple[int, int]]
"""A (t,n)-Shamir sharing: [(party_id, share_value), ...] of length n."""

BallotMatrix = List[List[Shares]]
"""N_valid × M matrix; BallotMatrix[i][m] is the (t,n)-sharing of B̂_{i,m}."""

ScorePair = Tuple[Shares, Shares]
"""Candidate score as (Score^num, Score^den) — both (t,n)-sharings."""
