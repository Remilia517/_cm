import math
import random

def cross_entropy(p, q, base=math.e):
    """H(p, q) = - sum p_i log q_i"""
    ce = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            if qi <= 0:
                return float("inf")
            ce -= pi * math.log(qi, base)
    return ce

def normalize(v):
    s = sum(v)
    return [x / s for x in v]

# 固定一個真實分佈 p
p = [0.1, 0.4, 0.5]

# 產生一個不同於 p 的 q
q = normalize([random.random() for _ in range(len(p))])

# 確保 q != p
assert any(abs(pi - qi) > 1e-6 for pi, qi in zip(p, q))

H_pp = cross_entropy(p, p)
H_pq = cross_entropy(p, q)

print("p =", p)
print("q =", q)
print("H(p,p) =", H_pp)
print("H(p,q) =", H_pq)
print("H(p,q) > H(p,p) ?", H_pq > H_pp)
