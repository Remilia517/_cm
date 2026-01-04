"""
Basic but complete Monte Carlo demo
- Estimate pi by dart throwing
- Monte Carlo integration with confidence interval
- Convergence experiment across sample sizes
No external dependencies needed.
"""

from __future__ import annotations
import math
import random
import statistics
from dataclasses import dataclass
from typing import Callable, Iterable


# ----------------------------
# Utilities
# ----------------------------

@dataclass
class MCResult:
    estimate: float
    se: float                 # standard error (approx)
    ci95: tuple[float, float] # 95% confidence interval (approx)
    n: int


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    if not xs:
        raise ValueError("empty data")
    return sum(xs) / len(xs)


def _sample_std(xs: list[float]) -> float:
    # sample standard deviation (unbiased, ddof=1)
    if len(xs) < 2:
        return float("nan")
    return statistics.stdev(xs)


# ----------------------------
# 1) Estimate pi
# ----------------------------

def estimate_pi(n: int, *, rng: random.Random) -> MCResult:
    if n <= 0:
        raise ValueError("n must be positive")

    inside = 0
    # Generate points uniformly in [-1, 1] x [-1, 1]
    for _ in range(n):
        x = rng.uniform(-1.0, 1.0)
        y = rng.uniform(-1.0, 1.0)
        if x * x + y * y <= 1.0:
            inside += 1

    p_hat = inside / n
    pi_hat = 4.0 * p_hat

    # For a Bernoulli proportion p_hat, Var(p_hat) ≈ p_hat(1-p_hat)/n
    # So Var(pi_hat) ≈ 16 * p_hat(1-p_hat)/n
    var = 16.0 * p_hat * (1.0 - p_hat) / n
    se = math.sqrt(var)
    ci95 = (pi_hat - 1.96 * se, pi_hat + 1.96 * se)

    return MCResult(estimate=pi_hat, se=se, ci95=ci95, n=n)


# ----------------------------
# 2) Monte Carlo integration
# ----------------------------

def mc_integrate_uniform(
    f: Callable[[float], float],
    a: float,
    b: float,
    n: int,
    *,
    rng: random.Random,
) -> MCResult:
    """
    Estimate integral_a^b f(x) dx using uniform sampling.
      I ≈ (b-a) * mean(f(U_i)), where U_i ~ Uniform(a,b)

    Also returns an approximate 95% CI using sample standard deviation.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not (b > a):
        raise ValueError("require b > a")

    vals = []
    for _ in range(n):
        x = rng.uniform(a, b)
        vals.append(f(x))

    m = _mean(vals)
    s = _sample_std(vals)  # std of f(x) under uniform samples
    scale = (b - a)

    est = scale * m
    se = scale * (s / math.sqrt(n))  # standard error of the mean times scale
    ci95 = (est - 1.96 * se, est + 1.96 * se)

    return MCResult(estimate=est, se=se, ci95=ci95, n=n)


# ----------------------------
# 3) Convergence experiment
# ----------------------------

def run_convergence(
    fn: Callable[[int, random.Random], MCResult],
    ns: list[int],
    repeats: int,
    *,
    seed: int = 0
) -> dict[int, list[MCResult]]:
    """
    Run repeated MC experiments for each n to see stability / convergence.
    """
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    out: dict[int, list[MCResult]] = {}
    base_rng = random.Random(seed)

    for n in ns:
        results: list[MCResult] = []
        for _ in range(repeats):
            # Use independent seeds per run for fair repeats
            run_seed = base_rng.randrange(1, 10**9)
            rng = random.Random(run_seed)
            results.append(fn(n, rng))
        out[n] = results
    return out


def summarize_runs(runs: dict[int, list[MCResult]]) -> None:
    """
    Print summary stats for each n: mean estimate, std of estimates, mean SE.
    """
    print("\nConvergence summary")
    print("-" * 72)
    print(f"{'n':>10} | {'mean(est)':>12} | {'std(est)':>12} | {'mean(SE)':>12}")
    print("-" * 72)
    for n in sorted(runs):
        ests = [r.estimate for r in runs[n]]
        ses = [r.se for r in runs[n]]
        mean_est = _mean(ests)
        std_est = statistics.pstdev(ests) if len(ests) > 1 else 0.0
        mean_se = _mean(ses)
        print(f"{n:>10} | {mean_est:>12.6f} | {std_est:>12.6f} | {mean_se:>12.6f}")
    print("-" * 72)


# ----------------------------
# Main demo
# ----------------------------

def main() -> None:
    seed = 42
    rng = random.Random(seed)

    print("=== Monte Carlo Demo (basic but complete) ===\n")

    # (A) Estimate pi
    n_pi = 200_000
    r_pi = estimate_pi(n_pi, rng=rng)
    print("[1] Estimate pi by dart throwing")
    print(f"n={r_pi.n:,}  pi_hat={r_pi.estimate:.6f}  true_pi={math.pi:.6f}")
    print(f"SE≈{r_pi.se:.6f}  95% CI≈[{r_pi.ci95[0]:.6f}, {r_pi.ci95[1]:.6f}]\n")

    # (B) Monte Carlo integration example: integral_0^1 x^2 dx = 1/3
    print("[2] Monte Carlo integration")
    f = lambda x: x * x
    true = 1.0 / 3.0
    n_int = 100_000
    r_int = mc_integrate_uniform(f, 0.0, 1.0, n_int, rng=rng)
    print("Integral of x^2 on [0,1]")
    print(f"n={r_int.n:,}  est={r_int.estimate:.6f}  true={true:.6f}")
    print(f"SE≈{r_int.se:.6f}  95% CI≈[{r_int.ci95[0]:.6f}, {r_int.ci95[1]:.6f}]\n")

    # (C) Convergence study (repeat runs)
    ns = [200, 1_000, 5_000, 20_000]
    repeats = 30

    print("[3] Convergence experiment (repeats to show stability)\n")

    pi_runs = run_convergence(
        fn=lambda n, rng: estimate_pi(n, rng=rng),
        ns=ns,
        repeats=repeats,
        seed=1
    )
    print("Pi estimation:")
    summarize_runs(pi_runs)

    int_runs = run_convergence(
        fn=lambda n, rng: mc_integrate_uniform(f, 0.0, 1.0, n, rng=rng),
        ns=ns,
        repeats=repeats,
        seed=2
    )
    print("\nIntegration estimation:")
    summarize_runs(int_runs)

    print("\nDone.")


if __name__ == "__main__":
    main()
