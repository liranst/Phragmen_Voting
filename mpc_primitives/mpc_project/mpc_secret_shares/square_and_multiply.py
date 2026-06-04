"""
square_and_multiply — Fast modular exponentiation.

Computes x^c mod n in O(log c) multiplications using the binary
square-and-multiply method.
"""


def square_and_multiply(x: int, c: int, n: int) -> int:
    """Return ``x**c mod n`` using binary exponentiation.

    Parameters
    ----------
    x : int
        Base.
    c : int
        Exponent (non-negative integer).
    n : int
        Modulus.
    """
    binary_c = bin(c)[2:]   # e.g. 13 → '1101'

    z = 1
    for bit in binary_c:
        z = (z * z) % n          # square
        if bit == '1':
            z = (z * x) % n      # multiply
    return z
