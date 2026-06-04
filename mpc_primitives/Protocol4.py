# Protocol4
import random
import os

from MPC_over_secret_shares.Protocol1 import protocol_1_share
from MPC_over_secret_shares.Protocol2 import protocol_2_reconstruct

# בהנחה שהפונקציות protocol_1_share ו-protocol_2_reconstruct מוגדרות כאן
# או מיובאות מקובץ אחר (למשל: from protocols import protocol_1_share, protocol_2_reconstruct)

def protocol_4_secure_mult(shares_u, shares_v, n, t, P,print_=False):
    """
    Input:
        shares_u - חלקי הסוד של u
        shares_v - חלקי הסוד של v
        n - מספר המשתתפים
        t - סף השחזור
        P - המספר הראשוני (שדה)
    Output: חלקי הסוד של המכפלה w = u * v (עם סף t)
    """
    # print("\n" + "=" * 50)
    # print("=== התחלת פרוטוקול 4: Secure Multiplication ===")
    # print("=" * 50)

    # רשימות לשמירת חלקי הסוד שהמשתתפים מחלקים
    # כל איבר ברשימה ייצג רשימת מניות שנוצרה ע"י משתתף ספציפי
    all_shares_r_t = []  # מניות עבור t
    all_shares_R_2t1 = []  # מניות עבור 2t-1
    if print_:
        print("\n--- שורות 1-5: הגרלת ערכים אקראיים וחלוקה (Share) ---")
    for i in range(1, n + 1):
        # שורה 2: כל משתתף מגריל r_i
        r_i = random.randint(0, P - 1)
        # שורה 3: (R_i מוגדר כ-r_i באופן מובלע)
        if print_:
            print(f"\n[P{i}] מגריל ערך סודי r_{i} = {r_i}")

            # שורה 4: חלוקה עם סף t
            print(f">>>> חלוקה עם סף t <<<<")
        # print(f"> P{i} מריץ Share(r_i) עם סף t={t}:")
        shares_r = protocol_1_share(r_i, n, t, P)
        all_shares_r_t.append(shares_r)
        if print_:
            print(f"all_shares_r_t = {all_shares_r_t[-1]}")
            # שורה 5: חלוקה עם סף 2t-1
            print(f">>> חלוקה עם סף 2t-1 <<<")
            # print(f"> P{i} מריץ Share(R_i) עם סף 2t-1={2 * t - 1}:")
        shares_R = protocol_1_share(r_i, n, 2 * t - 1, P)
        all_shares_R_2t1.append(shares_R)
        if print_:

            print(f"all_shares_R_2t1 = {all_shares_R_2t1[-1]}")
    if print_:
        print(f"סיכום")
        print(f"all_shares_r_t = {all_shares_r_t}")
        print(f"all_shares_R_2t1 = {all_shares_R_2t1}")
        print("\n--- שורות 6-8: סכימה מקומית של הערך האקראי המשותף ---")
        print(f"Pj: [[r]]j = Σ[[ri]]j")
    shares_r_final = []  # יכיל את [[r]]
    shares_R_final = []  # יכיל את [[R]]

    for j in range(1, n + 1):
        sum_r = 0
        sum_R = 0

        # המשתתף ה-j סוכם את כל המניות המיועדות אליו (נמצאות באינדקס j-1 בכל רשימה של משתתף i)
        for i in range(n):
            sum_r = (sum_r + all_shares_r_t[i][j - 1][1]) % P
            sum_R = (sum_R + all_shares_R_2t1[i][j - 1][1]) % P

        shares_r_final.append((j, sum_r))
        shares_R_final.append((j, sum_R))
        # print(f"P{j} מחשב: [[r]]_{j} = {sum_r}, [[R]]_{j} = {sum_R}")
    if print_:
        print(f"sum shares_r_final = {shares_r_final}")
        print(f"sum shares_R_final = {shares_R_final}")
        print()
        print("\n--- שורות 9-10: יצירת הפולינום המוסווה z ---")
        print(f"Pi [[z]]i = [[u]]i*[[v]]i+[[R]]i")
        print()
        print(f"חישוב: z_i = (u_i * v_i + R_i) mod P","\n")
    shares_z = []
    for i in range(1, n + 1):
        u_i = shares_u[i - 1][1]
        v_i = shares_v[i - 1][1]
        R_i = shares_R_final[i - 1][1]

        # חישוב: z_i = (u_i * v_i + R_i) mod P
        if print_:
            print(f">>>> round {i} <<<<<<")
        z_i = (u_i * v_i + R_i) % P
        if print_:
            print(f"z_{i} = ({u_i} * {v_i} + {R_i}) % {P} = {z_i}")

        shares_z.append((i, z_i))
        # print(f"P{i} מחשב: [[z]]_{i} = ({u_i} * {v_i} + {R_i}) mod {P} = {z_i}")

    # print("\n--- שורות 11-12: שחזור z ושידור פומבי ---")
    # אנו צריכים 2t-1 חלקים כדי לשחזר את z במלואו (נחתוך את הרשימה כדי להעביר בדיוק את הכמות הנדרשת לפונקציה שלך)
    required_shares_for_z = 2 * t - 1
    if print_:
        print(f"required_shares_for_z = {required_shares_for_z}")
        print(f"size shares_z = {len(shares_z)}, shares_z = {shares_z}")

    shares_for_reconstruction = shares_z[:required_shares_for_z]
    if print_:
        print(shares_z)
        print(shares_z[:66])

    # שימוש בפרוטוקול 2 לשחזור הערך (הפונקציה שלך משתמשת באות p קטנה, לכן העברנו P)
    z_reconstructed = protocol_2_reconstruct(shares_for_reconstruction, P)
    # print(f"\nהערך הפומבי ששוחזר ומשודר לכולם (P1 broadcasts z): z = {z_reconstructed}")
    if print_:
        print(f"z_reconstructed = {z_reconstructed}")
    # print("\n--- שורות 13-14: חילוץ התוצאה הסופית w ---")
    shares_w = []
    for i in range(1, n + 1):
        r_i = shares_r_final[i - 1][1]

        # הסרת ההסוואה על ידי חיסור הערך האקראי הקטן מתוך z הגלוי
        w_i = (z_reconstructed - r_i) % P
        shares_w.append((i, w_i))
        # print(f"P{i} מקבל מניה סופית: [[w]]_{i} = ({z_reconstructed} - {r_i}) mod {P} = {w_i}")

    # print("=" * 50)
    return shares_w


if __name__ == "__main__":
    # הגדרת seed לשם עקביות בתוצאות בזמן פיתוח
    random.seed(42)

    # פרמטרים התחלתיים
    P_field = 11
    n_parties = 3
    t_threshold = 2

    # הסודות שנרצה להכפיל: 4 * 5 = 20 (שזה 9 מודולו 11)
    secret_u = 4
    secret_v = 5
    print(f"סודות מקוריים לחישוב: u={secret_u}, v={secret_v}")
    print(f"תוצאה מצופה סופית: (4 * 5) mod 11 = 9\n")
    print(f"הכנה לפי הפעלת הפרוטוקול 4")
    # שלב מקדים: יצירת המניות של u ו-v בעזרת פרוטוקול 1 (המייצג את מצב הפתיחה)
    print(">>> מחלקים את הסוד u <<<")
    shares_u = protocol_1_share(secret_u, n_parties, t_threshold, P_field)
    print(f"shares_u = {shares_u}")
    print("\n>>> מחלקים את הסוד v <<<")

    shares_v = protocol_1_share(secret_v, n_parties, t_threshold, P_field)
    print(f"shares_v = {shares_v}")
    print()

    #os.system('cls' if os.name == 'nt' else 'clear')
    # print("קלט")
    # print(f"[[u]] = {shares_u}\n[[v]] = {shares_v}")
    print(f"מפה הפעלת פרוטוקול 4")
    # הפעלת פרוטוקול 4
    shares_w = protocol_4_secure_mult(shares_u, shares_v, n_parties, t_threshold, P_field)

    # וידוא נכונות: נשחזר את התוצאה בעזרת פרוטוקול 2 כדי לראות אם קיבלנו 9
    print("\n>>> מוודאים את התוצאה ע\"י שחזור הסוד w מתוך המניות החדשות <<<")
    recovered_w = protocol_2_reconstruct(shares_w, P_field)

    print(f"\nהתוצאה ששוחזרה: {recovered_w}")
    if recovered_w == (secret_u * secret_v) % P_field:
        print("הפרוטוקול עבד בהצלחה! תוצאת הכפל נשמרה בסוד ומומשה כהלכה.")
    else:
        print("ישנה שגיאה בחישוב התוצאה.")