import itertools

import numpy as np
from PIL import Image

from utils import LABtoRGB, RGBtoLAB, rgb_img_to_lab, lab_img_to_rgb, GetBoundary, ValidRGB, ValidLAB


def modify_lumin(palette_orig, mod_palette_idx, mod_palette_color):
    # change the lumin* after modify. parameters are in rgb
    palettes_lab = [RGBtoLAB(color) for color in palette_orig]
    palettes_lab[mod_palette_idx] = RGBtoLAB(mod_palette_color)
    # calculate new lumin based on equation
    for i in range(mod_palette_idx + 1, len(palette_orig)):
        palettes_lab[i] = (min(palettes_lab[i][0], palettes_lab[i - 1][0]), *palettes_lab[i][1:])
    for i in range(mod_palette_idx - 1, -1, -1):
        palettes_lab[i] = (max(palettes_lab[i][0], palettes_lab[i + 1][0]), *palettes_lab[i][1:])
    return [LABtoRGB(color) for color in palettes_lab]


def interp_L(l, orig_palette, new_palette):
    # calculate the k such that origin lumin is in [l_k, l_{k+1}]
    k = np.sum([l > cl for cl in orig_palette[:, 0]], axis=0) - 1
    l1, l2 = orig_palette[k, 0], orig_palette[k + 1, 0]
    # calculate the interpolation coef t, and get new lumin
    t = np.nan_to_num((l - l1) / (l2 - l1), nan=1.)
    L1, L2 = new_palette[k, 0], new_palette[k + 1, 0]
    return (L2 - L1) * t + L1


def modify_AB(x, ab_orig, ab_new, l):
    ab_orig[0] = l
    ab_new[0] = l
    # calculate the boundaries
    offset = ab_new - ab_orig
    if np.linalg.norm(offset) < 0.001:
        return x
    c_b = GetBoundary(ab_orig, offset, 1, 255)
    x_0 = x + offset  # calculate x_0
    if ValidLAB(x_0) and ValidRGB(LABtoRGB(x_0)):
        x_b = GetBoundary(x, x_0 - x, 1, 255)
    else:
        x_b = GetBoundary(ab_new, x_0 - ab_new, 0, 1)

    # calculate the modified color
    if np.linalg.norm(x_b - x) == 0:
        return x
    coef = np.min((1. / np.linalg.norm(x_b - x), 1. / np.linalg.norm(c_b - ab_orig))) * np.linalg.norm(offset)
    return x + (x_b - x) * coef


def image_recolor(img, palette_orig, palette_new, change_id):
    img_lab_np = rgb_img_to_lab(img)
    h, w, c = img_lab_np.shape
    palettes_lab_orig = np.array([(100, 0, 0)] + [RGBtoLAB(color) for color in palette_orig] + [(0, 0, 0)])
    palettes_lab_new = np.array([(100, 0, 0)] + [RGBtoLAB(color) for color in palette_new] + [(0, 0, 0)])
    L = interp_L(img_lab_np[:, :, 0], palettes_lab_orig[::-1], palettes_lab_new[::-1])
    # for i, j in itertools.product(range(h), range(w)):
    #     v = modify_AB(img_lab_np[i, j], palettes_lab_orig[change_id+1], palettes_lab_new[change_id+1], L[i, j])
    #     img_lab_np[i, j] = v
    img_lab_np[:, :, 0] = L
    return Image.fromarray(lab_img_to_rgb(img_lab_np), 'RGB')
