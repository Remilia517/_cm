import math

p = 0.5
log_prob = 10000 * math.log10(p)

print(f"log10(P) = {log_prob}")
print(f"P ≈ 10^{log_prob}")
