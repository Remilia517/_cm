import random
import cmath

# ========= 核心：多項式與導數（Horner 同時計算） =========
def poly_and_derivative(c, z):
    """
    c: [a_n, ..., a_0] (高次到常數)，係數可為 real/complex
    回傳 f(z), f'(z)（Horner 同時計算）
    """
    f = 0j
    df = 0j
    for a in c:
        df = df * z + f
        f = f * z + a
    return f, df

def poly_eval(c, z):
    """只算 f(z)，c: [a_n,...,a_0]"""
    f = 0j
    for a in c:
        f = f * z + a
    return f

def cauchy_bound(c):
    """所有根都落在 |z| <= 1 + max(|a_i|)/|a_n| 內（Cauchy bound）"""
    a_n = c[0]
    if a_n == 0:
        raise ValueError("最高次係數不能為 0")
    mx = 0.0
    for a in c[1:]:
        mx = max(mx, abs(a))
    return 1.0 + mx / (abs(a_n) + 1e-15)

def deflate(c, r):
    """
    用 (x - r) 做一次合成除法，把 n 次降成 n-1 次。
    回傳新的係數列表（長度少 1）
    """
    n = len(c) - 1
    b = [0j] * n
    b[0] = c[0]
    for i in range(1, n):
        b[i] = c[i] + b[i-1] * r
    return b

# ========= 用 GD 找一個根（複平面） =========
def find_one_root_by_gd(c, lr=1e-3, max_iter=200000, tol=1e-12, trials=50, seed=None):
    """
    在複平面用梯度下降最小化 0.5*|f(z)|^2 來找一個根
    """
    if seed is not None:
        random.seed(seed)

    R = cauchy_bound(c)
    best_z = None
    best_absf = float("inf")

    for _ in range(trials):
        # 隨機初始點：半徑用 sqrt 分佈讓面積均勻
        rho = R * (random.random() ** 0.5)
        theta = 2.0 * cmath.pi * random.random()
        z = rho * cmath.exp(1j * theta)

        for _ in range(max_iter):
            f, df = poly_and_derivative(c, z)
            af = abs(f)
            if af < tol:
                return z

            # 梯度方向（Wirtinger 形式常用寫法）：f(z)*conj(f'(z))
            g = f * df.conjugate()

            # 若導數幾乎 0，容易卡住：微擾動跳開
            if abs(df) < 1e-14 or abs(g) < 1e-18:
                z += (random.uniform(-1, 1) + 1j * random.uniform(-1, 1)) * (R * 1e-3)
                continue

            # 防爆：裁切步長（避免一下跳太遠）
            step = lr * g
            m = abs(step)
            if m > 1.0:
                step *= (1.0 / m)

            z = z - step

        # 記錄這次最好的 z
        af = abs(poly_eval(c, z))
        if af < best_absf:
            best_absf = af
            best_z = z

    return best_z  # 沒到 tol 就回傳目前最接近的

def polish_by_newton(c, z, iters=30):
    """用幾步牛頓法把根修漂亮（強烈建議，deflation 更穩）"""
    for _ in range(iters):
        f, df = poly_and_derivative(c, z)
        if abs(df) < 1e-14:
            break
        z2 = z - f / df
        if abs(z2 - z) < 1e-14:
            z = z2
            break
        z = z2
    return z

def all_roots(c, lr=1e-3, tol=1e-10, trials=80, seed=None, use_newton_polish=True):
    """
    回傳所有 n 個根（複數），順序不固定
    c: [a_n,...,a_0]
    """
    c_cur = [complex(a) for a in c]
    n = len(c_cur) - 1
    if n < 1:
        return []

    roots = []
    for _ in range(n):
        z = find_one_root_by_gd(c_cur, lr=lr, tol=tol*1e-2, trials=trials, seed=seed)
        if z is None:
            break

        if use_newton_polish:
            z = polish_by_newton(c_cur, z, iters=50)

        # 若還不準，再加強一次
        if abs(poly_eval(c_cur, z)) > tol * 100:
            z2 = find_one_root_by_gd(c_cur, lr=lr, tol=tol*1e-3, trials=trials*2, seed=seed)
            if use_newton_polish:
                z2 = polish_by_newton(c_cur, z2, iters=80)
            z = z2

        roots.append(z)
        c_cur = deflate(c_cur, z)

    return roots

# ========= 測試工具 =========
def pretty_roots(rs, nd=10, imag_eps=1e-10):
    """把根印得好看：接近實數就只印實部"""
    out = []
    for z in rs:
        if abs(z.imag) < imag_eps:
            out.append(round(z.real, nd))
        else:
            out.append(complex(round(z.real, nd), round(z.imag, nd)))
    return out

def check_roots(c, rs):
    """驗證：印出每個根代回去的 |f(z)|"""
    print("Verify |f(z)|:")
    for z in rs:
        val = abs(poly_eval([complex(a) for a in c], z))
        print(f"  z={z}  |f(z)|={val:.3e}")

if __name__ == "__main__":
    c1 = [1, 0, -5, 0, 4]   # x^4 - 5x^2 + 4
    rs1 = all_roots(c1, lr=1e-3, tol=1e-12, trials=120, seed=0)

    rs1_sorted = sorted(rs1, key=lambda z: z.real)
    print("Poly: x^4 - 5x^2 + 4")
    print("Roots:", pretty_roots(rs1_sorted))

