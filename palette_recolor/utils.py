import math


def LABtoXYZ(LAB):
    def f(n):
        return n ** 3 if n > 6 / 29 else 3 * ((6 / 29) ** 2) * (n - 4 / 29)

    assert (ValidLAB(LAB))

    L, a, b = LAB[0], LAB[1], LAB[2]
    X = 95.047 * f((L + 16) / 116 + a / 500)
    Y = 100.000 * f((L + 16) / 116)
    Z = 108.883 * f((L + 16) / 116 - b / 200)
    return (X, Y, Z)


def XYZtoRGB(XYZ):
    def f(n):
        return n * 12.92 if n <= 0.0031308 else (n ** (1 / 2.4)) * 1.055 - 0.055

    X, Y, Z = [x / 100 for x in XYZ]
    R = f(3.2406 * X + -1.5372 * Y + -0.4986 * Z) * 255
    G = f(-0.9689 * X + 1.8758 * Y + 0.0415 * Z) * 255
    B = f(0.0557 * X + -0.2040 * Y + 1.0570 * Z) * 255
    return (R, G, B)


def LABtoRGB(LAB):
    return XYZtoRGB(LABtoXYZ(LAB))


def RGBtoXYZ(RGB):
    def f(n):
        return n / 12.92 if n <= 0.04045 else ((n + 0.055) / 1.055) ** 2.4

    assert (ValidRGB(RGB))

    R, G, B = [f(x / 255) for x in RGB]
    X = (0.4124 * R + 0.3576 * G + 0.1805 * B) * 100
    Y = (0.2126 * R + 0.7152 * G + 0.0722 * B) * 100
    Z = (0.0193 * R + 0.1192 * G + 0.9505 * B) * 100
    return (X, Y, Z)


def XYZtoLAB(XYZ):
    def f(n):
        return n ** (1 / 3) if n > (6 / 29) ** 3 else (n / (3 * ((6 / 29) ** 2))) + (4 / 29)

    X, Y, Z = XYZ[0], XYZ[1], XYZ[2]
    X /= 95.047
    Y /= 100.000
    Z /= 108.883

    L = 116 * f(Y) - 16
    a = 500 * (f(X) - f(Y))
    b = 200 * (f(Y) - f(Z))
    return (L, a, b)


def RGBtoLAB(RGB):
    return XYZtoLAB(RGBtoXYZ(RGB))


def ValidRGB(RGB):
    return False not in [0 <= x <= 255 for x in RGB]


def ValidLAB(LAB):
    L, a, b = LAB
    return 0 <= L <= 100 and -128 <= a <= 127 and -128 <= b <= 127
