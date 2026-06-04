"""
Algorithm 18 — Computing the k-th Ranked Element (Cleartext)
Determine the k-th smallest element from a dataset using a binary-search
strategy over the range [0, M).

This is a *cleartext* algorithm.  It operates on ordinary Python integers
(not secret shares) and serves as a reference / oracle for the secure
counterpart that would use SecureCompare for the counting steps.

Pseudocode reference (line numbers match the paper):
---------------------------------------------------------------------------
Input:  m          — size of dataset
        {u_j}_{j ∈ [m]}  — dataset of integers
        M = 2^l    — upper bound (all u_j ∈ [0, M))
        k ∈ [m]   — the rank to select (1-indexed: k=1 is the minimum)

Line 1:  α ← 2^{l-1}
Line 2:  for all i = l-1 down to 1 do
Line 3:    κ_1 ← Σ_{j ∈ [m]} 1_{u_j ≤ α−1}
Line 4:    if κ_1 < k then
Line 5:      α ← α + 2^{i-1}
Line 6:    else
Line 7:      α ← α − 2^{i-1}
Line 8:  κ_2 ← Σ_{j ∈ [m]} 1_{u_j ≤ α}
Line 9:  if κ_2 > k then         (note: paper says ≥; using > is standard)
Line 10:   α ← α − 1
Line 11: M_k ← α
Output: The k-th ranked element M_k
---------------------------------------------------------------------------

Intuition (binary search on the value range)
--------------------------------------------
The algorithm performs a binary search over the sorted output range [0, M),
maintaining a candidate α that converges to the k-th smallest value.

At each step it counts how many dataset elements are ≤ α−1:
  • If count < k:  the k-th element is ≥ α, so move α right (add 2^{i-1}).
  • If count ≥ k:  the k-th element is < α, so move α left (subtract 2^{i-1}).

After l−1 iterations, α is a candidate.  The final adjustment (Lines 8-10)
handles ties — if more than k elements equal α, step back by 1.

Complexity: O(l · m) comparisons, l = log₂(M).
"""

import math
from typing import List


def algorithm_18_kth_ranked(
    dataset: List[int],
    M: int,
    k: int,
    l: int = None,
    print_: bool = False,
) -> int:
    """Return the k-th smallest element from dataset (1-indexed rank).

    Parameters
    ----------
    dataset : List[int]
        A list of m integers, all in [0, M).
    M       : int
        Upper bound on values; must be a power of 2 (M = 2^l).
    k       : int
        The rank to select.  k=1 returns the minimum, k=m the maximum.
    l       : int, optional
        Exponent such that M = 2^l.  Computed from M if omitted.
    print_  : bool
        Verbose debug output.

    Returns
    -------
    int
        The k-th smallest element M_k.

    Raises
    ------
    ValueError
        If k is out of range or M is not a power of 2.
    """
    m = len(dataset)
    if not (1 <= k <= m):
        raise ValueError(f"k={k} must be in [1, {m}]")
    if M <= 0 or (M & (M - 1)) != 0:
        raise ValueError(f"M={M} must be a positive power of 2")

    if l is None:
        l = int(math.log2(M))

    # ------------------------------------------------------------------
    # Line 1: α ← 2^{l-1}  (start at the midpoint of [0, M))
    # ------------------------------------------------------------------
    alpha = 1 << (l - 1)                                            # Line 1
    if print_:
        print(f"[Algorithm18] M=2^{l}={M}, k={k}, initial α={alpha}")

    # ------------------------------------------------------------------
    # Lines 2-7: Binary search to narrow α to the k-th ranked value
    # ------------------------------------------------------------------
    for i in range(l - 1, 0, -1):                                   # Line 2

        # Line 3: κ_1 = |{u_j : u_j ≤ α − 1}|  = |{u_j : u_j < α}|
        kappa_1 = sum(1 for u in dataset if u <= alpha - 1)         # Line 3
        if print_:
            print(f"  i={i}: α={alpha}, κ_1={kappa_1}")

        if kappa_1 < k:                                              # Line 4
            alpha = alpha + (1 << (i - 1))                          # Line 5
        else:
            alpha = alpha - (1 << (i - 1))                          # Line 7

    # ------------------------------------------------------------------
    # Lines 8-10: Final tie-break adjustment
    # ------------------------------------------------------------------
    kappa_2 = sum(1 for u in dataset if u <= alpha)                 # Line 8
    if print_:
        print(f"[Algorithm18] Final α={alpha}, κ_2={kappa_2}")

    if kappa_2 > k:                                                  # Line 9
        # More than k elements are ≤ α, meaning the k-th element is < α
        alpha = alpha - 1                                            # Line 10

    if print_:
        print(f"[Algorithm18] M_{k} = {alpha}")

    return alpha                                                     # Line 11


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Algorithm 18 Self-Test ===\n")

    # Example 1: sorted dataset — every rank matches the sorted list
    data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
    M    = 16   # 2^4, all values in [0, 16)
    l    = 4
    sorted_data = sorted(data)
    print(f"Dataset: {data}")
    print(f"Sorted:  {sorted_data}")
    all_ok = True
    for k in range(1, len(data) + 1):
        mk = algorithm_18_kth_ranked(data, M, k, l)
        # sorted_data[k-1] is the k-th smallest
        expected = sorted_data[k - 1]
        status = "✓" if mk == expected else "✗"
        print(f"  k={k}: M_{k} = {mk}  (expected {expected}) {status}")
        all_ok = all_ok and (mk == expected)

    print()

    # Example 2: edge cases
    data2 = [7]
    mk = algorithm_18_kth_ranked(data2, 8, 1, 3)
    print(f"Single element [7], k=1: M_1 = {mk}  (expected 7) {'✓' if mk == 7 else '✗'}")
    all_ok = all_ok and (mk == 7)

    data3 = [0, 0, 0, 0]
    mk = algorithm_18_kth_ranked(data3, 4, 2, 2)
    print(f"All-zeros [0,0,0,0], k=2: M_2 = {mk}  (expected 0) {'✓' if mk == 0 else '✗'}")
    all_ok = all_ok and (mk == 0)

    print()
    print("Algorithm 18 self-test", "PASSED ✓" if all_ok else "FAILED ✗")
