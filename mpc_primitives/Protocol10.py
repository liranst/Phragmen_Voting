from MPC_over_secret_shares.Protocol1 import protocol_1_share
from MPC_over_secret_shares.Protocol2 import protocol_2_reconstruct
from MPC_over_secret_shares.Protocol4 import protocol_4_secure_mult


def protocol_10_is_zero(shares_u, n, t, p, print_=False):
    """
    Protocol 10: IsZero - Equality to zero
    מטרת הפונקציה היא לחשב באופן מאובטח שיתוף של הביט w = 1_{u=0}
    מבלי לחשוף את הסוד u עצמו.

    Optimization — skip the redundant MSB iteration:
    -------------------------------------------------
    The exponent p-1 always has its most significant bit equal to 1.
    In the original square-and-multiply loop, the very first iteration
    (i = s-1) would unconditionally compute:
        [[z]] <- SecureMult([[1]], [[1]]) = [[1]]   (square, always)
        [[z]] <- SecureMult([[1]], [[u]]) = [[u]]   (multiply, because MSB = 1)
    These two SecureMult calls are entirely redundant — they merely set
    [[z]] to [[u]].

    The fix is:
        Line 1: [[z]] <- [[u]]          (initialize directly to u)
        Line 2: loop i = s-2 down to 0  (skip the MSB at i = s-1)
        Line 3:   [[z]] <- SecureMult([[z]], [[z]])
        Line 4:   if b_i == 1:
        Line 5:     [[z]] <- SecureMult([[z]], [[u]])
        Line 6: [[w]] <- [[1]] - [[z]]  (local affine operation)

    This saves exactly 2 SecureMult calls (and 2 communication rounds)
    on every invocation, regardless of the field size p.
    """
    if print_:
        print("\n" + "=" * 50)
        print("=== התחלת פרוטוקול 10: IsZero (Equality to Zero) ===")
        print("=" * 50)

    # שלב הכנה: ייצוג בינארי של p-1 (המעריך בחזקה של פרמה)
    p_minus_1 = p - 1
    # פונקציית bin מחזירה מחרוזת בסגנון '0b1010', לכן חותכים את ה-'0b'
    binary_b = bin(p_minus_1)[2:]
    if print_:
        print(f"המעריך לחישוב הוא p-1 = {p_minus_1}")
        print(f"הייצוג הבינארי שלו: {binary_b}")

    # Line 1: [[z]] <- [[u]]
    # האתחול הוא ישירות ל-[[u]] ולא ל-[[1]], כך שדילוג על הסיבית
    # המובילה (שהיא תמיד 1) חוסך שתי פעולות SecureMult יקרות.
    shares_z = list(shares_u)
    if print_:
        print(f"אתחול מותאם: shares_z <- shares_u = ")
        print(f"shares_z = {shares_z}")

    # Lines 2-5: לולאת Square-and-Multiply — מתחילים מ-s-2 (דולגים על ה-MSB)
    # binary_b[1:] מדלג על הסיבית המובילה (אינדקס 0 = MSB = תמיד '1')
    for idx, bit in enumerate(binary_b[1:]):
        if print_:
            print(f"\n--- איטרציה {idx + 1} | סיבית נוכחית: {bit} (i = s-{idx + 2}) ---")

        # Line 3: העלאה בריבוע (Square) — מתבצע תמיד
        shares_z = protocol_4_secure_mult(shares_z, shares_z, n, t, p)
        if print_:
            print(f"Square: shares_z = {shares_z}")

        # Lines 4-5: הכפלה ב-u (Multiply) — מתבצע רק אם הסיבית היא 1
        if bit == '1':
            if print_:
                print("הסיבית היא 1. מבצע פעולת Multiply: [[z]] <- SecureMult([[z]], [[u]])")
            shares_z = protocol_4_secure_mult(shares_z, shares_u, n, t, p)
            if print_:
                print(f"Multiply: shares_z = {shares_z}")

    # בשלב זה, shares_z מכיל את המניות של u^(p-1)

    # Line 6: חישוב מקומי של [[w]] = [[1]] - [[z]]
    # פעולה אפינית מקומית: אין צורך בתקשורת.
    # אם u = 0:  z = 0^(p-1) = 0  (mod p)  →  w = 1 - 0 = 1  ✓
    # אם u ≠ 0:  z = u^(p-1) = 1  (פרמה)   →  w = 1 - 1 = 0  ✓
    if print_:
        print("\n--- שלב סופי: חישוב מקומי של [[w]] = [[1]] - [[z]] ---")
    shares_w = []

    for i in range(1, n + 1):
        # מחלצים את ערך ה-y של המניה של המשתתף ה-i
        z_i = shares_z[i - 1][1]

        # חיסור מודולו p: w_i = (1 - z_i) mod p
        w_i = (1 - z_i) % p
        shares_w.append((i, w_i))
        if print_:
            print(f"P{i} מחשב: [[w]]_{i} = (1 - {z_i}) mod {p} = {w_i}")
    if print_:
        print("=" * 50)

    # Output: מחזירים את המניות של התשובה הסופית
    return shares_w

if __name__ == "__main__":
    S = 0

    P_FIELD = 11
    N_PARTIES = 3
    T_THRESHOLD = 2
    shares_u = protocol_1_share(S, P_FIELD, N_PARTIES, T_THRESHOLD, P_FIELD)

    # shares_u = [(1, 6), (2, 5), (3, 4)]
    shares_w_zero = protocol_10_is_zero(shares_u, N_PARTIES, T_THRESHOLD, P_FIELD)
    print(f"{shares_w_zero=}")
    recovered_secret = protocol_2_reconstruct(shares_w_zero, P_FIELD)
    print(f"{recovered_secret=}")