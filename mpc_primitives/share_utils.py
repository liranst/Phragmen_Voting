"""
share_utils.py — Local share arithmetic helpers.

A (t,n)-Shamir sharing is stored as List[Tuple[int, int]]:
    [(1, y_1), (2, y_2), ..., (n, y_n)]

All operations here are *local* (no communication), exploiting the linearity
of Shamir's scheme:  if shares_u encodes secret u and shares_v encodes v,
then any linear combination of their share-values encodes the same linear
combination of the secrets.
"""

from typing import List, Tuple

Shares = List[Tuple[int, int]]  # type alias used throughout the library


# ---------------------------------------------------------------------------
# Constant sharings
# ---------------------------------------------------------------------------

def shares_zero(n: int) -> Shares:
    """Return a valid (t,n)-sharing of the constant 0 for any threshold t.

    The underlying polynomial is f(x) = 0, so every evaluation is 0.
    """
    return [(i, 0) for i in range(1, n + 1)]


def shares_one(n: int) -> Shares:
    """Return a valid (t,n)-sharing of the constant 1 for any threshold t.

    The underlying polynomial is f(x) = 1 (degree-0), so every evaluation is 1.
    """
    return [(i, 1) for i in range(1, n + 1)]


def shares_const(c: int, n: int, p: int) -> Shares:
    """Return a valid (t,n)-sharing of the public constant c (mod p)."""
    val = c % p
    return [(i, val) for i in range(1, n + 1)]


# ---------------------------------------------------------------------------
# Local arithmetic (no interaction needed)
# ---------------------------------------------------------------------------

def shares_add(a: Shares, b: Shares, p: int) -> Shares:
    """[[a]] + [[b]]  (local, element-wise mod p).

    Encodes the secret a + b.
    """
    return [(xa, (ya + yb) % p) for (xa, ya), (_, yb) in zip(a, b)]


def shares_sub(a: Shares, b: Shares, p: int) -> Shares:
    """[[a]] - [[b]]  (local, element-wise mod p).

    Encodes the secret a - b.
    """
    return [(xa, (ya - yb) % p) for (xa, ya), (_, yb) in zip(a, b)]


def shares_scalar(c: int, a: Shares, p: int) -> Shares:
    """c * [[a]]  (local scalar multiplication mod p).

    Encodes the secret c * a.
    """
    c_mod = c % p
    return [(x, (c_mod * y) % p) for x, y in a]


def shares_add_const(a: Shares, c: int, p: int) -> Shares:
    """[[a]] + c  (local, add public constant c to all share values mod p).

    Encodes the secret a + c.
    """
    c_mod = c % p
    return [(x, (y + c_mod) % p) for x, y in a]


def shares_negate(a: Shares, p: int) -> Shares:
    """- [[a]]  (local negation mod p).

    Encodes the secret -a.
    """
    return [(x, (-y) % p) for x, y in a]


def shares_sum(share_list: List[Shares], n: int, p: int) -> Shares:
    """Sum a list of sharings locally.

    Returns [[sum_i s_i]] where each s_i is a sharing in share_list.
    """
    result = shares_zero(n)
    for s in share_list:
        result = shares_add(result, s, p)
    return result
