"""
share_utils — Local share arithmetic helpers.

A (t,n)-Shamir sharing is represented as ``List[Tuple[int, int]]``:
    [(1, y_1), (2, y_2), ..., (n, y_n)]

All operations here are *local* (no inter-party communication).
They exploit the linearity of Shamir's scheme: any linear combination of
share-values encodes the same linear combination of the underlying secrets.
"""

from typing import List, Tuple

Shares = List[Tuple[int, int]]


# ---------------------------------------------------------------------------
# Constant sharings
# ---------------------------------------------------------------------------

def shares_zero(n: int) -> Shares:
    """Sharing of the constant 0 (valid for any threshold t)."""
    return [(i, 0) for i in range(1, n + 1)]


def shares_one(n: int) -> Shares:
    """Sharing of the constant 1 (constant polynomial f(x)=1)."""
    return [(i, 1) for i in range(1, n + 1)]


def shares_const(c: int, n: int, p: int) -> Shares:
    """Sharing of the public constant *c* (mod *p*)."""
    val = c % p
    return [(i, val) for i in range(1, n + 1)]


# ---------------------------------------------------------------------------
# Local arithmetic
# ---------------------------------------------------------------------------

def shares_add(a: Shares, b: Shares, p: int) -> Shares:
    """[[a]] + [[b]]  encodes secret  a + b  (mod p)."""
    return [(xa, (ya + yb) % p) for (xa, ya), (_, yb) in zip(a, b)]


def shares_sub(a: Shares, b: Shares, p: int) -> Shares:
    """[[a]] - [[b]]  encodes secret  a - b  (mod p)."""
    return [(xa, (ya - yb) % p) for (xa, ya), (_, yb) in zip(a, b)]


def shares_scalar(c: int, a: Shares, p: int) -> Shares:
    """c * [[a]]  encodes secret  c * a  (mod p)."""
    c_mod = c % p
    return [(x, (c_mod * y) % p) for x, y in a]


def shares_add_const(a: Shares, c: int, p: int) -> Shares:
    """[[a]] + c  encodes secret  a + c  (mod p)."""
    c_mod = c % p
    return [(x, (y + c_mod) % p) for x, y in a]


def shares_negate(a: Shares, p: int) -> Shares:
    """-[[a]]  encodes secret  -a  (mod p)."""
    return [(x, (-y) % p) for x, y in a]


def shares_sum(share_list: List[Shares], n: int, p: int) -> Shares:
    """Sum a list of sharings locally: [[Σ_i s_i]]."""
    result = shares_zero(n)
    for s in share_list:
        result = shares_add(result, s, p)
    return result
