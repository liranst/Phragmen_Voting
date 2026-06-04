"""
Protocol 12 — Binary Long Division (Cleartext)

Reference cleartext implementation of binary long division.
Used as a test oracle for Protocol 13 (the secure version).

Reference: "SoK: Secure Computation over Secret Shares", Algorithm 12.
"""

from typing import Tuple


def binary_long_division(u: int, v: int, print_: bool = False) -> Tuple[int, int]:
    """Return ``(quotient, remainder)`` of ``u ÷ v`` via binary long division.

    Parameters
    ----------
    u : int
        Numerator (non-negative integer).
    v : int
        Divisor (positive integer).
    print_ : bool
        Verbose step-by-step debug output.

    Returns
    -------
    tuple[int, int]
        ``(q, r)`` such that ``u = q * v + r`` and ``0 <= r < v``.
    """
    u_bits = [int(bit) for bit in format(u, 'b')]  # MSB first
    s = len(u_bits)

    # Lines 1-2: initialise quotient and remainder
    q = 0
    r = 0

    if print_:
        print(f"Binary long division: {u} ÷ {v}")
        print(f"u in binary: {u_bits}")
        print("=" * 40)

    # Line 3: process each bit of u from MSB to LSB
    for j, u_j in enumerate(u_bits):
        # Line 4: shift remainder and append next bit
        r = 2 * r + u_j                              # Line 4

        if r < v:                                    # Line 5
            q_j = 0                                  # Line 6
        else:
            q_j = 1                                  # Line 8
            r = r - v                                # Line 9

        q = 2 * q + q_j                              # Line 10

        if print_:
            print(f"  step {j+1}: u_j={u_j}, r={r}, q_j={q_j}, q={q}")

    return q, r
