"""State persistence between election periods for tallying parties.

Reference: "Fairness Without Exposure: Privacy-Preserving Phragmén Voting",
Section 5 — the protocol operates *periodically*; talliers retain the shared
wallet state from one period to the next.

Architecture note
-----------------
The entities that hold persistent state are the **talliers** P_1 … P_D, not
voters and not candidates.  Each tallier P_d stores only its own share index
(the d-th component) from every (t,n)-sharing:

    delta_share_d  = delta_shares[d-1]           — one Tuple[int, int]
    wallet_d[i]    = wallet_shares[i][d-1]        — one Tuple[int, int] per voter

Voters submit **fresh ballots** each period and may vote for different
candidates.  Only the scaled-wallet state carries over.

Workflow
--------
End of period r:
    for d in 1..n:
        save_tallier_state(d, delta_r, wallets_r, path_d)

Start of period r+1:
    states = [load_tallier_state(d, path_d) for d in 1..n]
    delta_r, wallets_r = combine_tallier_states(states)
    # then run Algorithm 2 on the *new* ballots, Algorithms 4-8 as usual
"""

import json
from typing import List, Tuple

from protocol.types import Shares

# One party's contribution to a single sharing: (party_id, share_value).
SingleShare = Tuple[int, int]


def save_tallier_state(
    tallier_id: int,
    delta_shares: Shares,
    wallet_shares: List[Shares],
    path: str,
) -> None:
    """Persist tallier d's individual shares to a JSON file.

    Extracts the d-th entry from each (t,n)-sharing and writes it to disk.
    No other party's data is included; the file contains only what P_d
    would hold in a real distributed deployment.

    Args:
        tallier_id: 1-based party index d ∈ {1, …, n}.
        delta_shares: Full (t,n)-sharing of the global denominator Δ.
        wallet_shares: List of N_valid full (t,n)-sharings of W̃_i.
        path: Destination file path.  Parent directory must exist.
    """
    idx = tallier_id - 1   # 0-based index into the Shares list

    state = {
        "tallier_id": tallier_id,
        "n_valid": len(wallet_shares),
        "delta_share": list(delta_shares[idx]),
        "wallet_shares": [list(ws[idx]) for ws in wallet_shares],
    }

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(state, fh)


def load_tallier_state(
    tallier_id: int,
    path: str,
) -> Tuple[SingleShare, List[SingleShare]]:
    """Load tallier d's individual shares from a JSON file.

    Args:
        tallier_id: 1-based party index d — must match the value stored in
            the file, otherwise a ValueError is raised.
        path: Source file path.

    Returns:
        A tuple ``(delta_share_d, wallet_shares_d)`` where:

        * ``delta_share_d``   is ``(d, Δ_d)``   — tallier d's share of Δ.
        * ``wallet_shares_d`` is ``[(d, W̃_i_d) for i in range(n_valid)]``
          — tallier d's share of each voter's wallet.

    Raises:
        ValueError: If the stored tallier_id does not match the argument.
        FileNotFoundError: If the path does not exist.
    """
    with open(path, "r", encoding="utf-8") as fh:
        state = json.load(fh)

    stored_id = state["tallier_id"]
    if stored_id != tallier_id:
        raise ValueError(
            f"File {path!r} contains state for tallier {stored_id}, "
            f"but tallier_id={tallier_id} was requested"
        )

    delta_share: SingleShare = tuple(state["delta_share"])           # type: ignore[assignment]
    wallet_shares_d: List[SingleShare] = [
        tuple(ws) for ws in state["wallet_shares"]                   # type: ignore[misc]
    ]
    return delta_share, wallet_shares_d


def combine_tallier_states(
    tallier_states: List[Tuple[SingleShare, List[SingleShare]]],
) -> Tuple[Shares, List[Shares]]:
    """Reconstruct full (t,n)-Shares objects from all talliers' individual shares.

    In a real deployment each tallier sends its share to an agreed
    reconstruction point; in the simulation all shares are available locally.

    Args:
        tallier_states: List of ``(delta_share_d, wallet_shares_d)`` tuples,
            one per tallier, ordered by tallier index 1 … n
            (``tallier_states[0]`` = P_1's state, etc.).

    Returns:
        ``(delta_shares, wallet_shares)`` — full (t,n)-sharings ready for
        use by Algorithms 4–8 in the next period.
    """
    n = len(tallier_states)

    # delta_shares = [(1, Δ_1), (2, Δ_2), …, (n, Δ_n)]
    delta_shares: Shares = [tallier_states[d][0] for d in range(n)]

    # wallet_shares[i] = [(1, W̃_i_1), …, (n, W̃_i_n)]
    n_valid = len(tallier_states[0][1])
    wallet_shares: List[Shares] = [
        [tallier_states[d][1][i] for d in range(n)]
        for i in range(n_valid)
    ]

    return delta_shares, wallet_shares
