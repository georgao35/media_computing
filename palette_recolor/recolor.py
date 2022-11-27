import copy
import itertools
import tqdm
from tqdm import contrib
from PIL import Image
from skimage import io, color

from utils import *

from IPython import embed


def modify_lumin(palettes_orig, mod_palette_idx, mod_palette_color):
    # change the lumin* after modify. parameters are in rgb
    palettes_lab = copy.deepcopy(palettes_orig)
    palettes_lab[mod_palette_idx] = RGBtoLAB(mod_palette_color)
    # calculate new lumin based on equation
    for i in range(mod_palette_idx + 1, len(palettes_lab)):
        palettes_lab[i] = (min(palettes_lab[i][0], palettes_lab[i - 1][0]), *palettes_lab[i][1:])
    for i in range(mod_palette_idx - 1, -1, -1):
        palettes_lab[i] = (max(palettes_lab[i][0], palettes_lab[i + 1][0]), *palettes_lab[i][1:])
    print(palettes_lab)
    return palettes_lab


def interp_L(l, orig_palette, new_palette):
    # calculate the k such that origin lumin is in [l_k, l_{k+1}]
    k = np.sum([l > cl for cl in orig_palette[:, 0]], axis=0) - 1
    l1, l2 = orig_palette[k, 0], orig_palette[k + 1, 0]
    # calculate the interpolation coef t, and get new lumin
    t = np.nan_to_num((l - l1) / (l2 - l1), nan=1.)
    L1, L2 = new_palette[k, 0], new_palette[k + 1, 0]
    return (L2 - L1) * t + L1


def omega(cs1, lbd, Lab, sigma):
    res = 0.
    for i in range(cs1.shape[0]):
        res += lbd[i] * phi(np.linalg.norm(cs1[i] - Lab), sigma)
    return res


def get_lambda(palettes, sigma):
    k = palettes.shape[0]
    s = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            s[i, j] = phi(np.linalg.norm(palettes[i] - palettes[j]), sigma)
    # res = np.clip(np.linalg.inv(s), a_min=0., a_max=np.inf)
    # row_sums = res.sum(axis=0, keepdims=True)
    # return res / row_sums
    return np.linalg.inv(s)


def phi(r, sigma):
    return np.exp(-r * r / (2 * sigma * sigma))


def get_sigma(palettes):
    res = 0
    k = palettes.shape[0]
    for i, j in itertools.product(range(k), repeat=2):
        if i == j:
            continue
        res += np.linalg.norm(palettes[i] - palettes[j])
    return res / (k * k + k)


def modify_AB(pixel_color, ab1, ab2):
    def distance(p1, p2):
        return np.linalg.norm(p1 - p2)
    # set the initial values. use copy to prevent modify original values
    x = pixel_color
    C, C_prime = copy.deepcopy(ab1), copy.deepcopy(ab2)
    C[0] = x[0]
    C_prime[0] = x[0]
    # calculate the boundaries
    offset = C_prime - C
    if np.linalg.norm(offset) < 0.001:
        return x
    c_b = GetIntersect(C, C_prime)
    x_0 = x + offset  # calculate x_0
    # get x_b based on situation of x_0
    if InGamut(x_0):
        x_b = GetIntersect(x, x_0)
    else:
        x_b = GetBoundary(C_prime, x_0)

    # calculate the modified color
    if distance(x_b, x) == 0:
        return x
    if distance(c_b, C) == 0:
        ratio = 1
    else:
        ratio = np.min((1, (distance(x_b, x) / distance(c_b, C))))
    ratio *= distance(C_prime, C)
    res = x + (x_b - x) / distance(x_b, x) * ratio
    # print(res - x)
    return res


def image_recolor(img, palette_orig_lab, palette_new_lab, change_id):
    img_lab_np = rgb_img_to_lab(img)
    h, w, c = img_lab_np.shape
    k = len(palette_orig_lab)
    palettes_lab_orig = np.array([(100, 0, 0)] + palette_orig_lab + [(0, 0, 0)])
    palettes_lab_new = np.array([(100, 0, 0)] + palette_new_lab + [(0, 0, 0)])
    # get linear transformation of L
    L = interp_L(img_lab_np[:, :, 0], palettes_lab_orig[::-1], palettes_lab_new[::-1])

    # get transformation of AB
    sigma = get_sigma(palettes_lab_orig[:-1])
    lmbda = get_lambda(palettes_lab_orig[1:-1], sigma)

    for i, j in tqdm.tqdm(itertools.product(range(h), range(w))):
        res = np.zeros(3)
        for p in range(k):
            v = modify_AB(img_lab_np[i, j], copy.deepcopy(palettes_lab_orig[p + 1]),
                          copy.deepcopy(palettes_lab_new[p + 1]))
            v[0] = L[i, j]
            wei = omega(palettes_lab_orig[1:-1], lmbda[:, p], v, sigma)
            res += wei * v
        img_lab_np[i, j] = res

    img_lab_np[:, :, 0] = L
    return Image.fromarray(lab_img_to_rgb(img_lab_np), 'RGB')
