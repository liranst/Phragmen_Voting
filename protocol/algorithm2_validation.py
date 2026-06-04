"""Algorithm 2 — Input Validation, Sanitization & Compaction.

Reference: Algorithm 2 in "Fairness Without Exposure: Privacy-Preserving
Phragmén Voting", Section 5.2 (page 9).  Run once at setup (or per period
if ballots are re-submitted).

For each ballot entry B_{n,m} the protocol verifies B_{n,m} ∈ {0, 1} by
checking that B_{n,m}·(B_{n,m} − 1) = 0 using a single SecureMult followed
by a public Reconstruct.  Voters whose ballot fails this check are flagged as
cheaters and removed; the remaining honest-voter rows are compacted into a
fresh matrix B̂ of size N_valid × M.
"""

from typing import List, Tuple

from mpc_primitives.mpc_project.mpc_secret_shares import (
    protocol_2_reconstruct,
    protocol_4_secure_mult,
    shares_one,
    shares_sub,
)
from protocol.types import BallotMatrix, Shares


def algorithm_2_input_validation(
    B_shares: BallotMatrix,
    n: int,
    t: int,
    p: int,
) -> Tuple[BallotMatrix, int]:
    """Validate ballot entries, remove cheating voters, and compact the matrix.

    Algorithm 2 of "Fairness Without Exposure: Privacy-Preserving Phragmén
    Voting", Section 5.2.

    For each entry B_{n,m} the protocol computes
    ``[[u]] = SecureMult([[B]], [[B]] − [[1]])``
    and reconstructs u publicly.  Because u = B·(B−1) equals 0 if and only
    if B ∈ {0, 1}, a non-zero u identifies voter n as a cheater.  The first
    invalid entry short-circuits the inner loop (Lines 7–8 of the paper).

    Args:
        B_shares: An N × M matrix of (t,n)-sharings, where
            ``B_shares[voter][candidate]`` is the sharing of B_{n,m}.
            Voters are zero-indexed; the paper uses 1-indexed notation.
        n: Number of MPC parties (talliers).
        t: Reconstruction threshold.
        p: Prime field modulus.

    Returns:
        A tuple ``(B_hat, n_valid)`` where:

        * ``B_hat`` is an N_valid × M matrix of (t,n)-sharings containing
          only the rows of honest voters, in their original relative order
          and re-indexed from 0.
        * ``n_valid`` is the count of honest voters (public after this step).
    """
    num_voters = len(B_shares)
    num_candidates = len(B_shares[0]) if num_voters > 0 else 0

    cheaters: set[int] = set()
    one_shares: Shares = shares_one(n)  # [[1]] — shared public constant

    # Lines 2-8: for each voter, check every ballot entry until a violation
    # is found.  The inner loop breaks on the first detected cheat (Line 8).
    for voter in range(num_voters):                                  # Line 2
        for cand in range(num_candidates):                           # Line 3
            b_nm: Shares = B_shares[voter][cand]

            # Line 4: [[u]] = SecureMult([[B]], [[B]] − [[1]])
            # [[B]] − [[1]] is a local operation (affine subtraction).
            b_minus_one: Shares = shares_sub(b_nm, one_shares, p)
            u_shares: Shares = protocol_4_secure_mult(
                b_nm, b_minus_one, n, t, p
            )

            # Line 5: u ← Reconstruct([[u]])  — publicly revealed.
            u_val: int = protocol_2_reconstruct(u_shares[:t], p)

            # Lines 6-8: non-zero u means B ∉ {0, 1} → voter is a cheater.
            if u_val != 0:                                           # Line 6
                cheaters.add(voter)                                  # Line 7
                break                                                # Line 8

    # Line 9: N_valid = N − |Cheaters|.
    n_valid: int = num_voters - len(cheaters)

    # Lines 10-15: compact honest voters into B_hat, preserving row order.
    B_hat: BallotMatrix = [
        B_shares[voter]
        for voter in range(num_voters)
        if voter not in cheaters
    ]

    return B_hat, n_valid                                            # Line 16
