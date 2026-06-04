# protocol1.py
import random


def protocol_1_share(S, n, t, P, print_=False):
    """
    Input: S (הסוד), n (מספר המשתתפים), t (סף השחזור), P (המספר הראשוני התוחם את השדה)
    הפונקציה מדמה את התהליך שעושה ה"דילר" (P_i) כדי לחלק את הסוד למניות.
    """
    # print("--- שלב 1: בניית הפולינום האקראי ---")

    # ניצור רשימה שתשמור את המקדמים האקראיים של הפולינום
    coefficients = []

    # מחרוזת עזר כדי להדפיס את הפולינום בצורה יפה
    poly_str = f"{S}"

    # מגרילים t-1 מקדמים אקראיים לפולינום
    for j in range(1, t):
        a_j = random.randint(0, P - 1)
        coefficients.append(a_j)
        poly_str += f" + {a_j}x^{j}"
    if print_:
        print(f"======="*3)
        print(f"הסוד שלנו (S) הוא: {S}")
        print(f"הפולינום שהוגרל הוא: f(x) = {poly_str} (mod {P})")
        print(f"=======" * 3)
    def f(x):
        # במקום להדפיס רק את האיבר האחרון, נבנה את התרגיל המלא
        result = S
        calc_str = f"{S}"

        for j in range(1, t):
            a_j = coefficients[j - 1]
            term = a_j * (x ** j)

            # מוסיפים למחרוזת ההדפסה את האיבר הנוכחי
            calc_str += f" + {a_j}*({x}^{j})"
            result += term

        # מפעילים את המודולו בסוף לקבלת התוצאה הסופית בשדה
        final_result = result % P
        if print_:
            #             הדפסה מפורטת של החישוב
            print(f"f({x}) = {calc_str} = {result} mod {P} = {final_result}")
            print(f"f({x}) = {final_result}")
        return final_result

    # נכין רשימה ריקה שתשמור את כל ה"מניות" (Shares) שנחלק למשתתפים
    shares = []

    # print("--- שלב 2: חישוב המניות (Shares) לכל משתתף ---")

    # הלולאה עוברת על כל אחד מ-n המשתתפים (מ-1 עד n)
    for j in range(1, n + 1):
        # print(f"P{j}: ", end="")
        # print(f"---- rund {j}-----")
        share_value = f(j)

        shares.append((j, share_value))
        # print("-" * 40)

    return shares


if __name__ == "__main__":
    # שימוש בזרע (seed) אקראי קבוע רק לצורך הדגמה זהה לפלט שלך (אופציונלי)
    random.seed(41)

    gold = protocol_1_share(S=1, n=3, t=2, P=11, print_=False)

    print("\n===== פלט סופי: חלוקת המניות (Shamir's Secret Sharing) =====")
    print(f"shares: {gold}")

    """
    gold = protocol_1_share(S=7, n=3, t=2, P=11)
    gold: [(1, 6), (2, 5), (3, 4)]
    """