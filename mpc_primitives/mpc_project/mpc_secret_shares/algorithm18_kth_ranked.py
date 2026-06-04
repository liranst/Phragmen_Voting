"""
Algorithm 18 — Computing the k-th Ranked Element (Cleartext)

Determine the k-th smallest element from a dataset using a binary-search
strategy over the value range [0, M).

This cleartext algorithm serves as both a standalone utility and a reference
for the secure analogue that would replace the counting steps with
SecureCompare calls.

Reference: "SoK: Secure Computation over Secret Shares", Algorithm 18.
"""

import math
from typing import List, Optional


def algorithm_18_kth_ranked(
    dataset: List[int],
    M: int,
    k: int,
    l: Optional[int] = None,
    print_: bool = False,
) -> int:
    """Return the k-th smallest element from *dataset* (1-indexed rank).

    Parameters
    ----------
    dataset : list of int
        Dataset of *m* integers, all in ``[0, M)``.
    M       : int
        Upper bound; must be a power of 2 (``M = 2^l``).
    k       : int
        Rank to select.  ``k=1`` → minimum, ``k=len(dataset)`` → maximum.
    l       : int, optional
        Exponent such that ``M = 2^l``.  Computed from *M* when omitted.
    print_  : bool
        Verbose debug output.

    Returns
    -------
    int
        The k-th smallest element ``M_k``.

    Raises
    ------
    ValueError
        If *k* is out of range or *M* is not a positive power of 2.
    """
    m = len(dataset)
    if not (1 <= k <= m):
        raise ValueError(f"k={k} must be in [1, {m}]")
    if M <= 0 or (M & (M - 1)) != 0:
        raise ValueError(f"M={M} must be a positive power of 2")

    if l is None:
        l = int(math.log2(M))

    # ------------------------------------------------------------------
    # Line 1
    # ------------------------------------------------------------------
    alpha = 1 << (l - 1)                                    # Line 1

    if print_:
        print(f"[Algorithm18] M=2^{l}={M}, k={k}, initial α={alpha}")

    # ------------------------------------------------------------------
    # Lines 2-7: binary search over the value range
    # ------------------------------------------------------------------
    for i in range(l - 1, 0, -1):                           # Line 2
        kappa_1 = sum(1 for u in dataset if u <= alpha - 1) # Line 3
        if print_:
            print(f"  i={i}: α={alpha}, κ_1={kappa_1}")
        if kappa_1 < k:                                      # Line 4
            alpha = alpha + (1 << (i - 1))                  # Line 5
        else:
            alpha = alpha - (1 << (i - 1))                  # Line 7

    # ------------------------------------------------------------------
    # Lines 8-10: tie-break adjustment
    # ------------------------------------------------------------------
    kappa_2 = sum(1 for u in dataset if u <= alpha)         # Line 8
    if print_:
        print(f"[Algorithm18] Final α={alpha}, κ_2={kappa_2}")
    kappa_prev = sum(1 for u in dataset if u <= alpha - 1)  # Line 9
    if kappa_prev >= k:                                      # Line 9
        alpha = alpha - 1                                    # Line 10

    return alpha                                             # Line 11
