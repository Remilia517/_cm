from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Optional, Tuple, List

EPS = 1e-9


def is_close(a: float, b: float, eps: float = EPS) -> bool:
    return abs(a - b) <= eps


@dataclass(frozen=True)
class Point:
    """點：2D 平面上的座標 (x, y)"""
    x: float
    y: float

    def __add__(self, other: "Point") -> "Point":
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Point") -> "Point":
        return Point(self.x - other.x, self.y - other.y)

    def scale(self, s: float, center: "Point" = None) -> "Point":
        """以 center 為中心縮放（同心縮放）"""
        if center is None:
            center = Point(0.0, 0.0)
        v = self - center
        return Point(center.x + v.x * s, center.y + v.y * s)

    def rotate(self, theta_rad: float, center: "Point" = None) -> "Point":
        """以 center 為中心旋轉 theta（弧度），逆時針為正"""
        if center is None:
            center = Point(0.0, 0.0)
        v = self - center
        c, s = math.cos(theta_rad), math.sin(theta_rad)
        return Point(center.x + v.x * c - v.y * s,
                     center.y + v.x * s + v.y * c)

    def dist2(self, other: "Point") -> float:
        dx, dy = self.x - other.x, self.y - other.y
        return dx * dx + dy * dy

    def dist(self, other: "Point") -> float:
        return math.sqrt(self.dist2(other))


@dataclass(frozen=True)
class Line:
    """
    線：用一般式表示 ax + by + c = 0
    (a, b) 不可同時為 0
    """
    a: float
    b: float
    c: float

    def __post_init__(self):
        if is_close(self.a, 0.0) and is_close(self.b, 0.0):
            raise ValueError("Line requires (a, b) not both 0.")

    @staticmethod
    def from_points(p1: Point, p2: Point) -> "Line":
        """由兩點決定一直線"""
        if p1.dist2(p2) <= EPS:
            raise ValueError("Two distinct points are required to define a line.")
        # (y - y1) = m (x - x1) → 轉一般式
        # 令 a = y1 - y2, b = x2 - x1, c = x1*y2 - x2*y1
        a = p1.y - p2.y
        b = p2.x - p1.x
        c = p1.x * p2.y - p2.x * p1.y
        return Line(a, b, c)

    def direction(self) -> Point:
        """線的方向向量 (dx, dy) 可取 (b, -a)"""
        return Point(self.b, -self.a)

    def normal(self) -> Point:
        """法向量 (a, b)"""
        return Point(self.a, self.b)

    def intersect_line(self, other: "Line") -> Optional[Point]:
        """兩直線交點（平行或重合回傳 None）"""
        d = self.a * other.b - other.a * self.b
        if is_close(d, 0.0):
            return None
        x = (self.b * other.c - other.b * self.c) / d
        y = (other.a * self.c - self.a * other.c) / d
        return Point(x, y)

    def project_point(self, p: Point) -> Point:
        """點 p 到直線的垂足（正交投影）"""
        # 直線 ax+by+c=0，垂足公式：
        # x' = x - a*(ax+by+c)/(a^2+b^2)
        # y' = y - b*(ax+by+c)/(a^2+b^2)
        denom = self.a * self.a + self.b * self.b
        t = (self.a * p.x + self.b * p.y + self.c) / denom
        return Point(p.x - self.a * t, p.y - self.b * t)

    def perpendicular_through(self, p: Point) -> "Line":
        """過點 p 作此直線的垂線"""
        # 若原線法向量 n=(a,b)，則原線方向向量 d=(b,-a)
        # 垂線的法向量可取原線方向向量 (b, -a)
        A = self.b
        B = -self.a
        C = -(A * p.x + B * p.y)
        return Line(A, B, C)

    def translate(self, dx: float, dy: float) -> "Line":
        """
        平移直線：把平面所有點 (x, y) → (x+dx, y+dy)
        代入 ax+by+c=0 得到新 c
        """
        # a(x-dx)+b(y-dy)+c=0 → ax+by+(c - a*dx - b*dy)=0
        return Line(self.a, self.b, self.c - self.a * dx - self.b * dy)

    def rotate(self, theta_rad: float, center: Point = None) -> "Line":
        """
        旋轉直線：用兩個點在直線上取樣，旋轉後再重建
        """
        if center is None:
            center = Point(0.0, 0.0)

        # 找兩個在線上的點：優先用截距
        pts = []
        if not is_close(self.b, 0.0):
            # x=0 → y = -c/b
            pts.append(Point(0.0, -self.c / self.b))
        if not is_close(self.a, 0.0):
            # y=0 → x = -c/a
            pts.append(Point(-self.c / self.a, 0.0))
        if len(pts) < 2:
            # 非常特殊情況：用方向向量補點
            p0 = Point(0.0, 0.0)
            # 讓 p0 在直線上：找任一點在直線上
            if not is_close(self.b, 0.0):
                p0 = Point(0.0, -self.c / self.b)
            else:
                p0 = Point(-self.c / self.a, 0.0)
            d = self.direction()
            pts = [p0, Point(p0.x + d.x, p0.y + d.y)]

        p1r = pts[0].rotate(theta_rad, center)
        p2r = pts[1].rotate(theta_rad, center)
        return Line.from_points(p1r, p2r)

    def scale(self, s: float, center: Point = None) -> "Line":
        """
        縮放直線：用兩點縮放後重建（同心縮放）
        """
        if center is None:
            center = Point(0.0, 0.0)

        # 取兩個點同 rotate 的策略
        pts = []
        if not is_close(self.b, 0.0):
            pts.append(Point(0.0, -self.c / self.b))
        if not is_close(self.a, 0.0):
            pts.append(Point(-self.c / self.a, 0.0))
        if len(pts) < 2:
            p0 = Point(0.0, 0.0)
            if not is_close(self.b, 0.0):
                p0 = Point(0.0, -self.c / self.b)
            else:
                p0 = Point(-self.c / self.a, 0.0)
            d = self.direction()
            pts = [p0, Point(p0.x + d.x, p0.y + d.y)]

        p1s = pts[0].scale(s, center)
        p2s = pts[1].scale(s, center)
        return Line.from_points(p1s, p2s)


@dataclass(frozen=True)
class Circle:
    """圓：圓心 center、半徑 r"""
    center: Point
    r: float

    def __post_init__(self):
        if self.r < 0:
            raise ValueError("Radius must be non-negative.")

    def intersect_circle(self, other: "Circle") -> List[Point]:
        """兩圓交點（0/1/2個）"""
        c1, c2 = self.center, other.center
        r1, r2 = self.r, other.r
        d = c1.dist(c2)

        # 無交點：太遠、包含且不交、同心不同半徑
        if d > r1 + r2 + EPS:
            return []
        if d < abs(r1 - r2) - EPS:
            return []
        if is_close(d, 0.0) and not is_close(r1, r2):
            return []
        # 無限多交點：同圓（這裡回傳空，視作「交點不唯一」）
        if is_close(d, 0.0) and is_close(r1, r2):
            return []

        # 幾何推導：沿著 c1->c2 的方向找交點
        # a = (r1^2 - r2^2 + d^2) / (2d)
        a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d)
        h2 = r1 * r1 - a * a
        if h2 < -EPS:
            return []
        h = math.sqrt(max(0.0, h2))

        # 基準點 p = c1 + a * (c2-c1)/d
        vx = (c2.x - c1.x) / d
        vy = (c2.y - c1.y) / d
        px = c1.x + a * vx
        py = c1.y + a * vy

        # 垂直方向 ( -vy, vx )
        rx = -vy * h
        ry = vx * h

        p1 = Point(px + rx, py + ry)
        p2 = Point(px - rx, py - ry)

        if p1.dist2(p2) <= EPS:
            return [p1]  # 相切
        return [p1, p2]

    def intersect_line(self, line: Line) -> List[Point]:
        """直線與圓交點（0/1/2個）"""
        # 作法：把座標平移到圓心，直線 ax+by+c=0
        # 圓心 (cx,cy)，令 x = X+cx, y=Y+cy
        # 代入：aX + bY + (a*cx+b*cy+c)=0
        cx, cy = self.center.x, self.center.y
        a, b, c = line.a, line.b, line.c + a * cx + b * cy
        r = self.r

        # 原點到直線距離：|c|/sqrt(a^2+b^2)
        denom = a * a + b * b
        dist = abs(c) / math.sqrt(denom)
        if dist > r + EPS:
            return []
        # 投影點（在平移座標下）
        # (X0, Y0) = -c/(a^2+b^2) * (a, b)
        X0 = -a * c / denom
        Y0 = -b * c / denom

        # 若相切
        if is_close(dist, r):
            return [Point(X0 + cx, Y0 + cy)]

        # 方向向量（沿直線）：(b, -a)，單位化
        inv = 1.0 / math.sqrt(denom)
        dx = b * inv
        dy = -a * inv

        # 從投影點沿直線走 t，滿足 X^2+Y^2=r^2
        # t = sqrt(r^2 - dist^2)
        t = math.sqrt(max(0.0, r * r - dist * dist))
        p1 = Point(X0 + dx * t + cx, Y0 + dy * t + cy)
        p2 = Point(X0 - dx * t + cx, Y0 - dy * t + cy)
        return [p1, p2]

    def translate(self, dx: float, dy: float) -> "Circle":
        return Circle(Point(self.center.x + dx, self.center.y + dy), self.r)

    def rotate(self, theta_rad: float, center: Point = None) -> "Circle":
        return Circle(self.center.rotate(theta_rad, center), self.r)

    def scale(self, s: float, center: Point = None) -> "Circle":
        # 圓心縮放、半徑也乘 |s|
        return Circle(self.center.scale(s, center), abs(self.r * s))


@dataclass(frozen=True)
class Triangle:
    """三角形：三個頂點 A, B, C"""
    A: Point
    B: Point
    C: Point

    def translate(self, dx: float, dy: float) -> "Triangle":
        return Triangle(Point(self.A.x + dx, self.A.y + dy),
                        Point(self.B.x + dx, self.B.y + dy),
                        Point(self.C.x + dx, self.C.y + dy))

    def rotate(self, theta_rad: float, center: Point = None) -> "Triangle":
        return Triangle(self.A.rotate(theta_rad, center),
                        self.B.rotate(theta_rad, center),
                        self.C.rotate(theta_rad, center))

    def scale(self, s: float, center: Point = None) -> "Triangle":
        return Triangle(self.A.scale(s, center),
                        self.B.scale(s, center),
                        self.C.scale(s, center))

    def side_lengths(self) -> Tuple[float, float, float]:
        ab = self.A.dist(self.B)
        bc = self.B.dist(self.C)
        ca = self.C.dist(self.A)
        return ab, bc, ca


def pythagorean_verify(line: Line, p: Point, q_on_line: Point) -> Tuple[bool, Triangle, Point]:
    """
    根據(直線上的點 q, 線外點 p)做垂足 h，形成直角三角形 p-q-h
    驗證 |PQ|^2 ≈ |PH|^2 + |QH|^2
    回傳: (是否成立, 三角形, 垂足H)
    """
    h = line.project_point(p)  # 垂足
    tri = Triangle(p, q_on_line, h)

    PQ2 = p.dist2(q_on_line)
    PH2 = p.dist2(h)
    QH2 = q_on_line.dist2(h)
    ok = is_close(PQ2, PH2 + QH2, eps=1e-7)  # 誤差可放寬一點
    return ok, tri, h


# ------------------ Demo ------------------
if __name__ == "__main__":
    # 1) 定義點、線、圓
    p1 = Point(0, 0)
    p2 = Point(4, 4)
    L1 = Line.from_points(p1, p2)        # y = x
    L2 = Line.from_points(Point(0, 4), Point(4, 0))  # y = -x + 4

    C1 = Circle(Point(2, 1), 3)
    C2 = Circle(Point(5, 1), 2)

    # 2) 兩直線交點、兩圓交點、線圓交點
    inter_LL = L1.intersect_line(L2)
    print("Line-Line intersection:", inter_LL)

    inter_CC = C1.intersect_circle(C2)
    print("Circle-Circle intersections:", inter_CC)

    inter_LC = C1.intersect_line(L2)
    print("Line-Circle intersections:", inter_LC)

    # 3) 給定直線與線外點，作垂直線
    P = Point(3, 0)  # 線外點（對 L1: y=x）
    perp = L1.perpendicular_through(P)
    H = L1.project_point(P)
    print("Perpendicular line through P:", perp)
    print("Foot of perpendicular H:", H)

    # 4) 用(直線上的點 Q)、(線外點 P)、(垂足 H) 驗證畢氏
    Q = Point(2, 2)  # 在 L1 上
    ok, tri, H2 = pythagorean_verify(L1, P, Q)
    print("Pythagorean holds?", ok)
    print("Triangle vertices (P, Q, H):", tri)

    # 5) 定義三角形物件（已完成：Triangle）
    T = Triangle(Point(0, 0), Point(2, 0), Point(0, 1))
    print("Triangle side lengths:", T.side_lengths())

    # 6) 幾何物件平移/縮放/旋轉
    print("Point rotate 90deg:", Point(1, 0).rotate(math.pi/2))
    print("Line translate dx=1,dy=2:", L1.translate(1, 2))
    print("Circle scale by 2:", C1.scale(2))
    print("Triangle rotate 30deg:", T.rotate(math.pi/6))
