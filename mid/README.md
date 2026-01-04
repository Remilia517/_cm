本報告使用chatgpt完成[程式碼](https://chatgpt.com/share/695a430a-9420-8013-ba44-0bde28f19443)和[說明](https://chatgpt.com/share/695a4325-1418-8013-934c-b23879ff2829)。
# Monte Carlo Demo (Basic but Complete)

本專案是一份**不依賴任何第三方套件**的 Python Monte Carlo（蒙地卡羅）示範程式，完整展示：

1. 使用隨機模擬估計圓周率 π
2. 使用 Monte Carlo 方法進行數值積分，並計算標準誤與 95% 信賴區間
3. 透過重複實驗與不同樣本數，觀察估計值的收斂性與穩定性

此專案適合作為：
- 機率與統計 / 數值方法課程範例
- Monte Carlo 方法入門教學
- 驗證標準誤（SE）與信賴區間合理性的實驗模板

---

## 專案設計重點

- **零外部依賴**：僅使用 Python 標準函式庫
- **估計值 + 不確定度**：所有 Monte Carlo 結果皆回傳估計值、SE 與 95% CI
- **可重現性（Reproducibility）**：明確使用 `random.Random(seed)` 控制隨機性
- **結構清楚、可擴充**：適合加入進階 Monte Carlo 技術（variance reduction 等）

---

## 程式結構總覽

```
mc_demo.py
├─ MCResult              # Monte Carlo 結果資料結構
├─ estimate_pi           # π 的 Monte Carlo 估計
├─ mc_integrate_uniform  # 一維 Monte Carlo 積分
├─ run_convergence       # 收斂實驗（多次重複）
├─ summarize_runs        # 收斂結果摘要
└─ main                  # 示範主程式
```

---

## 資料結構：MCResult

```python
@dataclass
class MCResult:
    estimate: float
    se: float
    ci95: tuple[float, float]
    n: int
```

| 欄位 | 說明 |
|----|----|
| `estimate` | Monte Carlo 估計值（point estimate） |
| `se` | 標準誤（standard error） |
| `ci95` | 近似 95% 信賴區間 |
| `n` | 樣本數 |

> 設計理念：**估計值與不確定度永遠一起回傳**，方便後續分析與比較。

---

## 一、圓周率 π 的 Monte Carlo 估計

### 數學原理

在正方形 \([-1,1] \times [-1,1]\) 中均勻隨機抽樣：

- 正方形面積 = 4
- 單位圓面積 = π
- 點落在圓內的機率：

\[
p = \frac{\pi}{4}
\]

因此：
\[
\hat{\pi} = 4 \cdot \hat{p}
\]

### 標準誤與信賴區間

- \( \hat{p} \) 為 Bernoulli 比例估計
- 近似變異數：

\[
\mathrm{Var}(\hat{\pi}) \approx 16 \cdot \frac{\hat{p}(1-\hat{p})}{n}
\]

- 95% CI（常態近似）：

\[
\hat{\pi} \pm 1.96 \cdot SE
\]

### 對應函式

```python
estimate_pi(n: int, *, rng: random.Random) -> MCResult
```

---

## 二、Monte Carlo 數值積分

### 數學原理

令 \( U \sim \mathrm{Uniform}(a,b) \)，則：

\[
\int_a^b f(x) dx = (b-a) \cdot \mathbb{E}[f(U)]
\]

Monte Carlo 估計：

\[
\hat{I} = (b-a) \cdot \frac{1}{n} \sum_{i=1}^n f(U_i)
\]

### 標準誤

\[
SE(\hat{I}) \approx (b-a) \cdot \frac{s}{\sqrt{n}}
\]

其中 \( s \) 為樣本標準差（ddof=1）。

### 對應函式

```python
mc_integrate_uniform(
    f: Callable[[float], float],
    a: float,
    b: float,
    n: int,
    *,
    rng: random.Random
) -> MCResult
```

---

## 三、收斂實驗（Convergence Experiment）

### 設計目的

- 檢查估計值是否隨樣本數增加而趨於穩定
- 驗證理論標準誤是否與實際重複實驗的變異一致

### 函式說明

```python
run_convergence(
    fn: Callable[[int, random.Random], MCResult],
    ns: list[int],
    repeats: int,
    *,
    seed: int = 0
) -> dict[int, list[MCResult]]
```

特點：
- 每次重複試驗使用**獨立 RNG seed**
- 整體仍可透過 `seed` 完全重現

### 統計摘要

```python
summarize_runs(runs: dict[int, list[MCResult]])
```

輸出指標：

| 指標 | 意義 |
|----|----|
| mean(est) | 多次試驗估計值平均 |
| std(est) | 試驗間估計值的實際波動 |
| mean(SE) | 每次試驗內部估計的平均標準誤 |

理想情況下：

```
std(est) ≈ mean(SE)
```

且皆隨 \( n \) 呈 \( 1/\sqrt{n} \) 衰減。
