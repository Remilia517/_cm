import math
from typing import List, Tuple

def _log(x: float, base: float) -> float:
    """log with custom base; base=math.e gives ln."""
    if x <= 0:
        raise ValueError("log input must be > 0")
    if base == math.e:
        return math.log(x)
    return math.log(x) / math.log(base)

def _normalize(p: List[float]) -> List[float]:
    s = sum(p)
    if s <= 0:
        raise ValueError("Sum of probabilities must be > 0")
    return [x / s for x in p]

def _check_prob_vec(p: List[float], eps: float = 1e-12) -> None:
    if any(x < -eps for x in p):
        raise ValueError("Probabilities must be non-negative")
    s = sum(p)
    if abs(s - 1.0) > 1e-9:
        raise ValueError(f"Probabilities must sum to 1. Got sum={s}")

def entropy(p: List[float], base: float = 2.0) -> float:
    """
    H(P) = - sum_i p_i log(p_i)
    0*log0 treated as 0.
    """
    _check_prob_vec(p)
    h = 0.0
    for pi in p:
        if pi > 0:
            h -= pi * _log(pi, base)
    return h

def cross_entropy(p: List[float], q: List[float], base: float = 2.0) -> float:
    """
    H(P, Q) = - sum_i p_i log(q_i)
    Requires: if p_i > 0 then q_i > 0, else infinite.
    """
    _check_prob_vec(p)
    _check_prob_vec(q)
    ce = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            if qi <= 0:
                return float("inf")
            ce -= pi * _log(qi, base)
    return ce

def kl_divergence(p: List[float], q: List[float], base: float = 2.0) -> float:
    """
    D_KL(P||Q) = sum_i p_i log(p_i / q_i)
    Requires: if p_i > 0 then q_i > 0, else infinite.
    """
    _check_prob_vec(p)
    _check_prob_vec(q)
    kl = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            if qi <= 0:
                return float("inf")
            kl += pi * _log(pi / qi, base)
    return kl

def mutual_information(joint_xy: List[List[float]], base: float = 2.0) -> float:
    """
    I(X;Y) = sum_{x,y} p(x,y) log( p(x,y) / (p(x)p(y)) )
    joint_xy: matrix p[x][y] sums to 1.
    """
    # flatten sum check
    total = sum(sum(row) for row in joint_xy)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Joint distribution must sum to 1. Got sum={total}")
    if any(pxy < -1e-12 for row in joint_xy for pxy in row):
        raise ValueError("Joint probabilities must be non-negative")

    px = [sum(row) for row in joint_xy]  # p(x)
    py = [sum(joint_xy[x][y] for x in range(len(joint_xy)))
          for y in range(len(joint_xy[0]))]  # p(y)

    mi = 0.0
    for x in range(len(joint_xy)):
        for y in range(len(joint_xy[0])):
            pxy = joint_xy[x][y]
            if pxy > 0:
                denom = px[x] * py[y]
                # denom should be > 0 if pxy > 0
                mi += pxy * _log(pxy / denom, base)
    return mi

if __name__ == "__main__":
    # ===== Example 1: Entropy / Cross-Entropy / KL =====
    # True distribution P and model distribution Q (discrete)
    P = [0.1, 0.4, 0.5]
    Q = [0.2, 0.3, 0.5]

    print("=== Using base-2 (bits) ===")
    print("H(P) =", entropy(P, base=2))
    print("H(P,Q) =", cross_entropy(P, Q, base=2))
    print("KL(P||Q) =", kl_divergence(P, Q, base=2))
    # identity: H(P,Q) = H(P) + KL(P||Q)
    print("Check: H(P)+KL =", entropy(P, 2) + kl_divergence(P, Q, 2))

    # ===== Example 2: Mutual Information I(X;Y) =====
    # Joint distribution p(x,y) for X in {0,1}, Y in {0,1,2}
    joint = [
        [0.10, 0.10, 0.20],  # x=0
        [0.05, 0.25, 0.30],  # x=1
    ]
    print("\nI(X;Y) =", mutual_information(joint, base=2), "bits")
