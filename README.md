# PP-Phragmén: Privacy-Preserving Phragmén Voting

PP-Phragmén is a cryptographic implementation of the Phragmén proportional representation voting method in which all ballots and intermediate tallies remain secret throughout the computation. The protocol runs as a Secure Multi-Party Computation (MPC) over Shamir secret shares: each party holds only an encrypted fragment of every vote and score, and the winning committee is determined by jointly executing a sequence of secure comparison, multiplication, and division sub-protocols without any party ever learning another party's inputs or any intermediate plaintext. This implementation follows Algorithms 1–8 of the accompanying Artificial Intelligence journal submission and is built on top of the MPC primitive library [`mpc-secret-shares`](https://pypi.org/project/mpc-secret-shares/).

## Installation

```
pip install pp-phragmen
```

## Usage

The package exposes Algorithms 1–8 from the paper as individual functions in the `protocol` module. The example below runs one complete election period for 3 voters and 2 candidates using 3 MPC tallying parties.

```python
import random
from mpc_secret_shares import share, reconstruct
from protocol.algorithm1_permutation import algorithm_1_oblivious_candidate_permutation
from protocol.algorithm2_validation import algorithm_2_input_validation
from protocol.algorithm3_initialization import algorithm_3_initialization
from protocol.algorithm4_score import algorithm_4_score_computation
from protocol.algorithm5_find_min import algorithm_5_find_minimum_score
from protocol.algorithm6_one_hot import algorithm_6_one_hot_winner
from protocol.algorithm7_wallet_update import algorithm_7_wallet_update
from protocol.algorithm8_reconstruct_winner import algorithm_8_reconstruct_winner

# ── MPC parameters ───────────────────────────────────────────────────────────
P = 251       # prime field modulus  (must satisfy p > N · Δ² — see paper §7)
N = 3         # number of tallying parties
T = 2         # reconstruction threshold  (T < N)

# ── Election parameters ──────────────────────────────────────────────────────
N_VOTERS      = 3
N_CANDIDATES  = 2
SECURITY_BITS = 128

random.seed(42)

# ── One-time setup (before the first period) ─────────────────────────────────

# Algorithm 1: each party contributes a random seed; their XOR fixes the
# oblivious candidate permutation used for tie-breaking throughout all periods.
seeds = [random.randbytes(SECURITY_BITS // 8) for _ in range(N)]
pi = algorithm_1_oblivious_candidate_permutation(
    SECURITY_BITS, N_CANDIDATES, N, seeds=seeds
)

# Secret-share the ballot matrix: B[i][m] = 1 if voter i approves candidate m.
# Voter 0 → {C0},  voter 1 → {C0},  voter 2 → {C1}.
ballots = [[1, 0], [1, 0], [0, 1]]
B = [[share(v, N, T, P) for v in row] for row in ballots]

# Algorithm 2: validate every ballot entry (B ∈ {0,1}) and remove cheaters.
B_hat, n_valid = algorithm_2_input_validation(B, N, T, P)

# Algorithm 3: initialise the global denominator Δ = 1 and all wallets W̃ᵢ = 0.
#              Run once, before the first period only.
delta, wallets = algorithm_3_initialization(n_valid, N, T, P)

# ── Per-period loop ───────────────────────────────────────────────────────────

# Algorithm 4: compute per-candidate scores as shared fractions (num / den).
scores, A_shares, T_shares = algorithm_4_score_computation(
    B_hat, wallets, delta, n_valid, N_CANDIDATES, N, T, P
)

# Algorithm 5: find the minimum (winning) score.
winning_score = algorithm_5_find_minimum_score(scores, N, T, P)

# Algorithm 6: produce a shared one-hot winner indicator with π tie-breaking.
chi = algorithm_6_one_hot_winner(scores, winning_score, pi, N, T, P)

# Algorithm 8: publicly reconstruct the winner index from the one-hot indicator.
winner = algorithm_8_reconstruct_winner(chi, T, P)
print(f"Period 1 winner: candidate {winner}")   # → candidate 0

# Algorithm 7: update voter wallets and Δ ready for the next period.
wallets, delta = algorithm_7_wallet_update(
    B_hat, chi, wallets, delta, A_shares, T_shares,
    n_valid, N_CANDIDATES, N, T, P
)
```

A multi-period run and the tallier state-persistence API (`save_tallier_state`, `load_tallier_state`, `combine_tallier_states`) are demonstrated in `tests/test_integration.py`.
