import cmath   # 專門處理複數的數學函式庫

def root2(a, b, c):
    """回傳二次多項式 ax^2 + bx + c 的兩個根（可能是複數）"""
    if a == 0:
        raise ValueError("a 不能為 0，否則不是二次多項式")

    # 判別式
    d = b*b - 4*a*c

    # 用 cmath.sqrt，d<0 時會自動給複數平方根
    sqrt_d = cmath.sqrt(d)

    # 根的公式
    x1 = (-b + sqrt_d) / (2*a)
    x2 = (-b - sqrt_d) / (2*a)
    return x1, x2


def f(x, a, b, c):
    """原來的多項式 f(x) = ax^2 + bx + c"""
    return a*x*x + b*x + c


def check_roots(a, b, c):
    """驗證 root2 算出的根代回去是否接近 0"""
    r1, r2 = root2(a, b, c)

    ok1 = cmath.isclose(f(r1, a, b, c), 0, rel_tol=1e-9, abs_tol=1e-9)
    ok2 = cmath.isclose(f(r2, a, b, c), 0, rel_tol=1e-9, abs_tol=1e-9)

    print("root1 =", r1, "f(root1) =", f(r1, a, b, c), "=>", ok1)
    print("root2 =", r2, "f(root2) =", f(r2, a, b, c), "=>", ok2)


# 範例：x^2 + 2x + 5 = 0，會有複數根
check_roots(1, 2, 5)
