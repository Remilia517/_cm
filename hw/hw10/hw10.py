import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# 原始函數 f(x)
# -------------------------
def f(x):
    return np.exp(-x**2)   # 高斯函數（傅立葉轉換後仍是高斯）

# -------------------------
# DFT（連續傅立葉正轉換）
# F(ω) = ∫ f(x) e^{-iωx} dx
# -------------------------
def dft(f, x_vals, omega):
    dx = x_vals[1] - x_vals[0]
    integrand = f(x_vals) * np.exp(-1j * omega * x_vals)
    return np.sum(integrand) * dx

# -------------------------
# IDFT（連續傅立葉逆轉換）
# f(x) = (1 / 2π) ∫ F(ω) e^{iωx} dω
# -------------------------
def idft(F_vals, omega_vals, x):
    domega = omega_vals[1] - omega_vals[0]
    integrand = F_vals * np.exp(1j * omega_vals * x)
    return (1 / (2 * np.pi)) * np.sum(integrand) * domega

# -------------------------
# 數值設定
# -------------------------
L = 5          # x 範圍 [-L, L]
W = 10         # ω 範圍 [-W, W]
N = 400        # 取樣點數

x_vals = np.linspace(-L, L, N)
omega_vals = np.linspace(-W, W, N)

# -------------------------
# 計算正轉換 F(ω)
# -------------------------
F_vals = np.array([dft(f, x_vals, w) for w in omega_vals])

# -------------------------
# 計算逆轉換 f_hat(x)
# -------------------------
f_reconstructed = np.array([idft(F_vals, omega_vals, x) for x in x_vals])

# -------------------------
# 繪圖驗證
# -------------------------
plt.figure(figsize=(8, 5))
plt.plot(x_vals, f(x_vals), label="Original f(x)")
plt.plot(x_vals, np.real(f_reconstructed), "--", label="Reconstructed f(x)")
plt.legend()
plt.xlabel("x")
plt.ylabel("Value")
plt.title("DFT → IDFT Reconstruction")
plt.grid()
plt.show()
