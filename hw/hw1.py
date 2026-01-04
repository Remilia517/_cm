h = 0.00001

def df(f, x):
    return (f(x + h) - f(x)) / h

def integral(f, a, b):
    area = 0
    x = a
    while x < b:
        area += f(x) * h
        x += h
    return area

def theorem1(f, x):
    F = lambda u: integral(f, 0, u)
    return df(F, x) == f(x)
