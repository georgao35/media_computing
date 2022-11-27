import copy
import math
import numpy as np

from PIL import ImageCms
from skimage import color


def rgb_img_to_lab(image):
    img = np.array(image)
    return color.rgb2lab(img)


def lab_img_to_rgb(image):
    # return color.lab2rgb(image)
    return (color.lab2rgb(image) * 255).astype(np.int8)


def LABtoXYZ(LAB):
    def f(n):
        return n ** 3 if n > 6 / 29 else 3 * ((6 / 29) ** 2) * (n - 4 / 29)

    assert (ValidLAB(LAB))

    L, a, b = LAB[0], LAB[1], LAB[2]
    X = 95.047 * f((L + 16) / 116 + a / 500)
    Y = 100.000 * f((L + 16) / 116)
    Z = 108.883 * f((L + 16) / 116 - b / 200)
    return X, Y, Z


def XYZtoRGB(XYZ, clip=True):
    def f(n):
        return n * 12.92 if n <= 0.0031308 else (n ** (1 / 2.4)) * 1.055 - 0.055

    X, Y, Z = [x / 100 for x in XYZ]
    R = f(3.2406 * X + -1.5372 * Y + -0.4986 * Z) * 255
    G = f(-0.9689 * X + 1.8758 * Y + 0.0415 * Z) * 255
    B = f(0.0557 * X + -0.2040 * Y + 1.0570 * Z) * 255
    return tuple(np.clip((R, G, B), a_min=0, a_max=255)) if clip else (R, G, B)


def LABtoRGB(LAB, clip=True):
    return XYZtoRGB(LABtoXYZ(LAB), clip)


def RGBtoXYZ(RGB):
    def f(n):
        return n / 12.92 if n <= 0.04045 else ((n + 0.055) / 1.055) ** 2.4

    assert (ValidRGB(RGB))

    R, G, B = [f(x / 255) for x in RGB]
    X = (0.4124 * R + 0.3576 * G + 0.1805 * B) * 100
    Y = (0.2126 * R + 0.7152 * G + 0.0722 * B) * 100
    Z = (0.0193 * R + 0.1192 * G + 0.9505 * B) * 100
    return X, Y, Z


def XYZtoLAB(XYZ, clip=True):
    def f(n):
        return n ** (1 / 3) if n > (6 / 29) ** 3 else (n / (3 * ((6 / 29) ** 2))) + (4 / 29)

    X, Y, Z = XYZ[0], XYZ[1], XYZ[2]
    X /= 95.047
    Y /= 100.000
    Z /= 108.883

    L = 116 * f(Y) - 16
    a = 500 * (f(X) - f(Y))
    b = 200 * (f(Y) - f(Z))
    return tuple(np.clip((L, a, b), a_min=[0, -128, -128], a_max=[100, 128, 128])) if clip else (L, a, b)


def RGBtoLAB(RGB, clip=True):
    return XYZtoLAB(RGBtoXYZ(RGB), clip)


def ValidRGB(RGB):
    return False not in [0 <= x <= 255 for x in RGB]


def ValidLAB(LAB):
    L, a, b = LAB[0], LAB[1], LAB[2]
    return 0 <= L <= 100 and -128 <= a <= 127 and -128 <= b <= 127


def InGamut(LAB):
    return ValidLAB(LAB) and ValidRGB(LABtoRGB(LAB, False))


def GetIntersect(o, n):
    p1 = copy.deepcopy(o)
    p2 = copy.deepcopy(n)
    offset = p2 - p1
    while InGamut(p2):
        p1 += offset
        p2 += offset
    return GetBoundary(p1, p2)


def GetBoundary(o, n, iter_n=20):
    start = copy.deepcopy(o)
    end = copy.deepcopy(n)
    for i in range(iter_n):
        if np.linalg.norm(start - end) < 0.001:
            break
        mid = (start + end) / 2
        if InGamut(mid):
            start = mid
        else:
            end = mid
    return (start + end) / 2
