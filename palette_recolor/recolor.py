from utils import LABtoRGB, RGBtoLAB, rgb_img_to_lab, lab_img_to_rgb


def modify_lumin(palette_orig, mod_palette_idx, mod_palette_color):
    # change the lumin* after modify. parameters are in rgb
    palettes_lab = [RGBtoLAB(color) for color in palette_orig]
    palettes_lab[mod_palette_idx] = RGBtoLAB(mod_palette_color)
    # calculate new lumin based on equation
    for i in range(mod_palette_idx + 1, len(palette_orig)):
        palettes_lab[i] = (min(palettes_lab[i][0], palettes_lab[i-1][0]), *palettes_lab[i][1:])
    for i in range(mod_palette_idx-1, -1, -1):
        palettes_lab[i] = (max(palettes_lab[i][0], palettes_lab[i+1][0]), *palettes_lab[i][1:])
    return [LABtoRGB(color) for color in palettes_lab]


def interp_L():
    pass


def modify_AB():
    pass


def image_recolor(img, palette_orig, palette_new):
    img_lab = rgb_img_to_lab(img)
