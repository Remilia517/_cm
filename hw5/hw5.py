"""
Finite Field GF(p) with Element objects + group/field axiom checkers

- Elements are objects supporting: +, -, *, /, **, ==, int()
- Field object provides:
    - F(x) -> Element in this field
    - F.zero, F.one
    - F.add_group  : (F, +)  commutative group
    - F.mul_group  : (F\\{0}, ×) commutative group
    - F.mul_monoid : (F, ×)  (includes 0) for distributivity checks

- Includes teacher-style randomized axiom checkers:
    - group axioms: closure, associativity, identity, inverse, commutativity
    - distributivity
    - check_field_axioms

Run this file directly to see a demo.
"""

from __future__ import annotations
from dataclasses import dataclass
import random

# =========================
# Utilities
# =========================

NUM_TEST_CASES = 100

def is_prime(n: int) -> bool:
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


# =========================
# Element object: GF(p)
# =========================

@dataclass(frozen=True)
class FpElement:
    """
    Finite field element in GF(p).
    value is always normalized to 0..p-1.
    """
    field: "GFpField"
    value: int

    def __post_init__(self):
        if not isinstance(self.value, int):
            raise TypeError("FpElement value must be int")
        object.__setattr__(self, "value", self.value % self.field.p)

    def __repr__(self) -> str:
        return f"{self.value} (mod {self.field.p})"

    def __int__(self) -> int:
        return self.value

    def is_zero(self) -> bool:
        return self.value == 0

    def _coerce(self, other) -> "FpElement":
        """
        Allow mixing with int. Disallow mixing different fields.
        """
        if isinstance(other, FpElement):
            if other.field is not self.field:
                raise TypeError("Cannot operate on elements from different fields")
            return other
        if isinstance(other, int):
            return self.field(other)
        return NotImplemented

    # ----- addition -----
    def __add__(self, other):
        o = self._coerce(other)
        if o is NotImplemented:
            return NotImplemented
        return self.field(self.value + o.value)

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        return self.field(-self.value)

    def __sub__(self, other):
        o = self._coerce(other)
        if o is NotImplemented:
            return NotImplemented
        return self.field(self.value - o.value)

    def __rsub__(self, other):
        o = self._coerce(other)
        if o is NotImplemented:
            return NotImplemented
        return o.__sub__(self)

    # ----- multiplication -----
    def __mul__(self, other):
        o = self._coerce(other)
        if o is NotImplemented:
            return NotImplemented
        return self.field(self.value * o.value)

    def __rmul__(self, other):
        return self.__mul__(other)

    # ----- inverse / division -----
    def inv(self) -> "FpElement":
        if self.value == 0:
            raise ZeroDivisionError("0 has no multiplicative inverse in GF(p)")
        # Fermat: a^(p-2) ≡ a^{-1} mod p
        return self.field(pow(self.value, self.field.p - 2, self.field.p))

    def __truediv__(self, other):
        o = self._coerce(other)
        if o is NotImplemented:
            return NotImplemented
        return self * o.inv()

    def __rtruediv__(self, other):
        o = self._coerce(other)
        if o is NotImplemented:
            return NotImplemented
        return o / self

    # ----- power -----
    def __pow__(self, n: int):
        if not isinstance(n, int):
            return NotImplemented
        if n < 0:
            return (self.inv()) ** (-n)
        return self.field(pow(self.value, n, self.field.p))

    # ----- equality -----
    def __eq__(self, other):
        o = self._coerce(other)
        if o is NotImplemented:
            return False
        return self.value == o.value and self.field is o.field


# =========================
# Field object
# =========================

class GFpField:
    """
    GF(p) field factory.
    Usage:
        F = GFpField(7)
        a = F(3)
        b = F(10)  # == 3 (mod 7)
        print(a + b, a * b, a / b)
    """
    def __init__(self, p: int):
        if not isinstance(p, int):
            raise TypeError("p must be int")
        if not is_prime(p):
            raise ValueError("GF(p) requires p to be a prime")
        self.p = p
        self.zero = FpElement(self, 0)
        self.one = FpElement(self, 1)

        # group-like adapters for your teacher-style tests
        self.add_group = AddGroupAdapter(self)
        self.mul_group = MulGroupNonZeroAdapter(self)
        self.mul_monoid = MulMonoidAdapter(self)

    def __call__(self, x: int) -> FpElement:
        if not isinstance(x, int):
            raise TypeError("GF(p) elements are created from ints")
        return FpElement(self, x)


# =========================
# Group/Monoid adapters (teacher interface)
# identity / operation / inverse / include / random_generate
# =========================

class AddGroupAdapter:
    """(F, +)"""
    def __init__(self, field: GFpField):
        self.F = field
        self._identity = field.zero

    @property
    def identity(self):
        return self._identity

    def include(self, x):
        return isinstance(x, FpElement) and x.field is self.F

    def operation(self, a, b):
        if not (self.include(a) and self.include(b)):
            raise TypeError("AddGroup: operands must be elements of the same field")
        return a + b

    def inverse(self, a):
        if not self.include(a):
            raise TypeError("AddGroup: operand must be an element of this field")
        return -a

    def random_generate(self):
        return self.F(random.randrange(self.F.p))


class MulMonoidAdapter:
    """(F, ×) includes 0; not a group (0 has no inverse). Used for distributivity tests."""
    def __init__(self, field: GFpField):
        self.F = field
        self._identity = field.one

    @property
    def identity(self):
        return self._identity

    def include(self, x):
        return isinstance(x, FpElement) and x.field is self.F

    def operation(self, a, b):
        if not (self.include(a) and self.include(b)):
            raise TypeError("MulMonoid: operands must be elements of the same field")
        return a * b

    def random_generate(self):
        return self.F(random.randrange(self.F.p))


class MulGroupNonZeroAdapter:
    """(F\\{0}, ×)"""
    def __init__(self, field: GFpField):
        self.F = field
        self._identity = field.one

    @property
    def identity(self):
        return self._identity

    def include(self, x):
        return isinstance(x, FpElement) and x.field is self.F and not x.is_zero()

    def operation(self, a, b):
        if not (self.include(a) and self.include(b)):
            raise TypeError("MulGroup: operands must be nonzero elements of the same field")
        return a * b

    def inverse(self, a):
        if not self.include(a):
            raise TypeError("MulGroup: operand must be a nonzero element of this field")
        return a.inv()

    def random_generate(self):
        # random nonzero: 1..p-1
        return self.F(random.randint(1, self.F.p - 1))


# =========================
# Teacher-style randomized axiom checkers
# =========================

# 1. 封閉性 (Closure)
def check_closure(g):
    for _ in range(NUM_TEST_CASES):
        a = g.random_generate()
        b = g.random_generate()
        result = g.operation(a, b)
        assert g.include(result), f"Closure failed: {a} op {b} = {result} is not in G"

# 2. 結合性 (Associativity)
def check_associativity(g):
    for _ in range(NUM_TEST_CASES):
        a = g.random_generate()
        b = g.random_generate()
        c = g.random_generate()
        assert g.operation(g.operation(a, b), c) == g.operation(a, g.operation(b, c)), \
            f"Associativity failed: ({a} op {b}) op {c} != {a} op ({b} op {c})"

# 3. 單位元素 (Identity Element)
def check_identity_element(g):
    for _ in range(NUM_TEST_CASES):
        a = g.random_generate()
        assert g.operation(a, g.identity) == a, \
            f"Left identity failed: {a} op {g.identity} != {a}"
        assert g.operation(g.identity, a) == a, \
            f"Right identity failed: {g.identity} op {a} != {a}"

# 4. 反元素 (Inverse Element)
def check_inverse_element(g):
    for _ in range(NUM_TEST_CASES):
        a = g.random_generate()
        a_inverse = g.inverse(a)
        assert g.include(a_inverse), f"Inverse {a_inverse} for {a} is not in G"
        assert g.operation(a, a_inverse) == g.identity, \
            f"Left inverse failed: {a} op {a_inverse} != {g.identity}"
        assert g.operation(a_inverse, a) == g.identity, \
            f"Right inverse failed: {a_inverse} op {a} != {g.identity}"

# 5. 交換性 (Commutativity)
def check_commutativity(g):
    for _ in range(NUM_TEST_CASES):
        a = g.random_generate()
        b = g.random_generate()
        assert g.operation(a, b) == g.operation(b, a), \
            f"Commutativity failed: {a} op {b} != {b} op {a}"

def check_group_axioms(g):
    check_closure(g)
    check_associativity(g)
    check_identity_element(g)
    check_inverse_element(g)
    print("All group axioms passed!")

def check_commutative_group(g):
    check_closure(g)
    check_associativity(g)
    check_identity_element(g)
    check_inverse_element(g)
    check_commutativity(g)
    print("交換群公理全部通過！")


# 6. 分配律 (Distributivity)
def check_distributivity(f: GFpField):
    """檢驗乘法對加法的分配律（在整個體上，含 0）"""
    print("--- 檢驗分配律 ---")
    for _ in range(NUM_TEST_CASES):
        a = f.add_group.random_generate()
        b = f.add_group.random_generate()
        c = f.add_group.random_generate()

        # 左分配律: a * (b + c) = (a * b) + (a * c)
        lhs = f.mul_monoid.operation(a, f.add_group.operation(b, c))
        rhs = f.add_group.operation(
            f.mul_monoid.operation(a, b),
            f.mul_monoid.operation(a, c)
        )
        assert lhs == rhs, \
            f"Left distributivity failed: {a} * ({b} + {c}) != ({a} * {b}) + ({a} * {c})"

        # 右分配律: (a + b) * c = (a * c) + (b * c)
        lhs = f.mul_monoid.operation(f.add_group.operation(a, b), c)
        rhs = f.add_group.operation(
            f.mul_monoid.operation(a, c),
            f.mul_monoid.operation(b, c)
        )
        assert lhs == rhs, \
            f"Right distributivity failed: ({a} + {b}) * {c} != ({a} * {c}) + ({b} * {c})"
    print("分配律通過！")


def check_field_axioms(f: GFpField):
    """
    檢驗有限體 GF(p) 的完整公理（以老師的隨機測試風格）
    """
    check_commutative_group(f.add_group)
    print("-" * 30)
    check_commutative_group(f.mul_group)
    print("-" * 30)
    check_distributivity(f)
    print("\n恭喜！所有有限體公理檢驗成功！")


# =========================
# Demo
# =========================

if __name__ == "__main__":
    F = GFpField(7)

    # Like ints/floats:
    a = F(3)
    b = F(4)
    c = F(10)  # == 3 mod 7

    print("Field:", f"GF({F.p})")
    print("a, b, c =", a, b, c)
    print("a + b =", a + b)
    print("a - b =", a - b)
    print("a * b =", a * b)
    print("a / b =", a / b)
    print("a^5 =", a ** 5)
    print("2 + a =", 2 + a)
    print("a * 100 =", a * 100)

    print("\n=== Run teacher-style axiom checks ===")
    check_field_axioms(F)
