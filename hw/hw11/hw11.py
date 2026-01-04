import numpy as np
from collections import Counter

def solve_ode_general(coefficients):
    """
    solve_ode_general(coefficients)

    求解「常係數齊次線性常微分方程」的一般解（以字串形式輸出）。

    假設方程為：
        a0 y^(n) + a1 y^(n-1) + ... + a(n-1) y' + an y = 0
    輸入 coefficients = [a0, a1, ..., an]  (a0 != 0)

    方法：
    1) 建立特徵多項式 a0 λ^n + a1 λ^(n-1) + ... + an = 0
    2) 用 np.roots 求根（數值法）
    3) 以容忍誤差 clustering 合併「幾乎相同」的根，避免浮點誤差導致：
       - 重根被拆成多個近似根
       - 共軛根重複計數
    4) 依根型態產生基底解：
       - 實根 a，重數 m：x^k e^(a x), k=0..m-1
       - 複根 a±bi，重數 m：x^k e^(a x)cos(bx), x^k e^(a x)sin(bx), k=0..m-1

    回傳：
        "y(x) = ..."  的字串
    """
    coeffs = np.array(coefficients, dtype=float)
    if coeffs.ndim != 1 or len(coeffs) < 2:
        raise ValueError("coefficients 必須是一維且長度 >= 2")
    if abs(coeffs[0]) < 1e-15:
        raise ValueError("最高階係數 a0 不可為 0")

    # --- 1) 求特徵根 ---
    roots = np.roots(coeffs)

    # --- 2) 將近似相同的根做 clustering（避免浮點誤差拆根/亂出虛部） ---
    # 容忍誤差：同時考慮絕對與相對（根大時用相對誤差，根小時用絕對誤差）
    abs_tol = 1e-7
    rel_tol = 1e-7

    clusters = []  # 每個元素: {"rep": complex, "members": [complex,...]}
    for r in roots:
        placed = False
        for c in clusters:
            rep = c["rep"]
            tol = abs_tol + rel_tol * max(1.0, abs(rep))
            if abs(r - rep) <= tol:
                c["members"].append(r)
                # 更新代表值（取平均更穩）
                c["rep"] = sum(c["members"]) / len(c["members"])
                placed = True
                break
        if not placed:
            clusters.append({"rep": r, "members": [r]})

    # cluster -> (representative root, multiplicity)
    uniq = [(c["rep"], len(c["members"])) for c in clusters]

    # --- 3) 將「幾乎為實數」的根強制視為實根（例如 2+1e-12 i） ---
    imag_tol = 1e-7
    cleaned = []
    for r, m in uniq:
        if abs(r.imag) <= imag_tol:
            cleaned.append((complex(r.real, 0.0), m))
        else:
            cleaned.append((r, m))
    uniq = cleaned

    # --- 4) 組裝一般解基底 ---
    # 先把根依型態分開
    real_roots = []      # (a, m)
    complex_roots = []   # (a+bi, m) with b != 0

    for r, m in uniq:
        if abs(r.imag) <= 0.0:
            real_roots.append((r.real, m))
        else:
            complex_roots.append((r, m))

    # 將實根排序（可讀性：大到小）
    real_roots.sort(key=lambda t: t[0], reverse=True)

    # 複根：配對共軛（只用 b>0 產生 cos/sin）
    # 用「近似 key」來配對，避免 0.99999999 與 1.00000001 分裂
    def key_ab(z):
        a = z.real
        b = abs(z.imag)
        # 量化到容忍誤差尺度（避免浮點噪音）
        qa = round(a / abs_tol) * abs_tol
        qb = round(b / abs_tol) * abs_tol
        return (qa, qb)

    # 收集複根 multiplicity（分別記正/負虛部）
    pos = Counter()
    neg = Counter()
    rep_pos = {}  # key -> representative (a+bi, b>0)
    for z, m in complex_roots:
        k = key_ab(z)
        if z.imag > 0:
            pos[k] += m
            # 取一個代表值（用平均方向修正：確保 imag > 0）
            rep_pos[k] = complex(z.real, abs(z.imag))
        else:
            neg[k] += m
            rep_pos.setdefault(k, complex(z.real, abs(z.imag)))

    # 以 min(pos,neg) 作為共軛對重數（對於實係數，多半相等；用 min 更穩健）
    complex_pairs = []
    for k in set(list(pos.keys()) + list(neg.keys())):
        mpair = min(pos[k], neg[k])
        if mpair > 0:
            zrep = rep_pos[k]
            complex_pairs.append((zrep.real, abs(zrep.imag), mpair))

    # 排序：先比 alpha=a，再比 beta=b
    complex_pairs.sort(key=lambda t: (t[0], t[1]), reverse=False)

    # --- 5) 產生項目字串 ---
    def fmt_num(x):
        # 避免 -0.0
        if abs(x) < 5e-13:
            x = 0.0
        # 用一般浮點輸出（跟你的測試輸出風格接近）
        return str(float(x))

    terms = []
    C_idx = 1

    # 實根項：x^k e^(a x)
    for a, m in real_roots:
        a_s = fmt_num(a)
        for k in range(m):
            if k == 0:
                terms.append(f"C_{C_idx}e^({a_s}x)")
            elif k == 1:
                terms.append(f"C_{C_idx}xe^({a_s}x)")
            else:
                terms.append(f"C_{C_idx}x^{k}e^({a_s}x)")
            C_idx += 1

    # 複根項：x^k e^(a x)cos(bx) 與 x^k e^(a x)sin(bx)
    for a, b, m in complex_pairs:
        a_s = fmt_num(a)
        b_s = fmt_num(b)
        for k in range(m):
            # 前綴 x^k
            if k == 0:
                prefix = ""
            elif k == 1:
                prefix = "x"
            else:
                prefix = f"x^{k}"

            # e^(ax) 可能 a=0
            exp_part = f"e^({a_s}x)"

            # 組合：prefix * exp * trig
            if prefix == "":
                terms.append(f"C_{C_idx}{exp_part}cos({b_s}x)")
            else:
                terms.append(f"C_{C_idx}{prefix}{exp_part}cos({b_s}x)")
            C_idx += 1

            if prefix == "":
                terms.append(f"C_{C_idx}{exp_part}sin({b_s}x)")
            else:
                terms.append(f"C_{C_idx}{prefix}{exp_part}sin({b_s}x)")
            C_idx += 1

    if not terms:
        return "y(x) = 0"

    return "y(x) = " + " + ".join(terms)


# -----------------------------
# 以下是測試主程式（照你的題目）
# -----------------------------

# 範例測試 (1): 實數單根: y'' - 3y' + 2y = 0
print("--- 實數單根範例 ---")
coeffs1 = [1, -3, 2]
print(f"方程係數: {coeffs1}")
print(solve_ode_general(coeffs1))

# 範例測試 (2): 實數重根: y'' - 4y' + 4y = 0
print("\n--- 實數重根範例 ---")
coeffs2 = [1, -4, 4]
print(f"方程係數: {coeffs2}")
print(solve_ode_general(coeffs2))

# 範例測試 (3): 複數共軛根: y'' + 4y = 0
print("\n--- 複數共軛根範例 ---")
coeffs3 = [1, 0, 4]
print(f"方程係數: {coeffs3}")
print(solve_ode_general(coeffs3))

# 範例測試 (4): 複數重根 (二重): (D^2 + 1)^2 y = 0
print("\n--- 複數重根範例 ---")
coeffs4 = [1, 0, 2, 0, 1]
print(f"方程係數: {coeffs4}")
print(solve_ode_general(coeffs4))

# 範例測試 (5): 高階重根: (lambda - 2)^3
print("\n--- 高階重根範例 ---")
coeffs5 = [1, -6, 12, -8]
print(f"方程係數: {coeffs5}")
print(solve_ode_general(coeffs5))
