import copy
import itertools
import tqdm
from PIL import Image
from multiprocessing import Pool, cpu_count
from utils import *


def modify_lumin(palettes_orig, mod_palette_idx, mod_palette_color):
    # change the lumin* after modify. parameters are in rgb
    palettes_lab = copy.deepcopy(palettes_orig)
    palettes_lab[mod_palette_idx] = RGBtoLAB(mod_palette_color)
    # calculate new lumin based on equation
    for i in range(mod_palette_idx + 1, len(palettes_lab)):
        palettes_lab[i] = (min(palettes_lab[i][0], palettes_lab[i - 1][0]), *palettes_lab[i][1:])
    for i in range(mod_palette_idx - 1, -1, -1):
        palettes_lab[i] = (max(palettes_lab[i][0], palettes_lab[i + 1][0]), *palettes_lab[i][1:])
    return palettes_lab


def interp_L(l, orig_palette, new_palette):
    # calculate the k such that origin lumin is in [l_k, l_{k+1}]
    k = np.sum([l >= cl for cl in orig_palette[:, 0]], axis=0) - 1
    print(l)
    k[k == 6] = 5
    l1, l2 = orig_palette[k, 0], orig_palette[k + 1, 0]
    # calculate the interpolation coef t, and get new lumin
    t = np.nan_to_num((l - l1) / (l2 - l1), nan=1.)
    L1, L2 = new_palette[k, 0], new_palette[k + 1, 0]
    return (L2 - L1) * t + L1


def omega(cs1, lmbda, Lab, sigma):
    res = 0.
    for i in range(cs1.shape[0]):
        res += lmbda[i] * phi(np.linalg.norm(cs1[i] - Lab), sigma)
    return res


def get_lambda(palettes, sigma):
    k = palettes.shape[0]
    s = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            s[i, j] = phi(np.linalg.norm(palettes[i] - palettes[j]), sigma)
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
    return res / (k * k - k)


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
    return x + (x_b - x) / distance(x_b, x) * ratio


def get_sample_colors(size=16):
    # vertex of the grids
    levels = [i * (255/(size-1)) for i in range(size)]
    colors = []
    lab_colors = []
    for r, g, b in itertools.product(levels, repeat=3):
        colors.append((r, g, b))
        lab_colors.append(RGBtoLAB((r, g, b)))

    return np.array(colors), np.array(lab_colors)


def get_nearest_vertices(c, step, step_range):
    # for each color, calculate the vertexes of the grid it belongs
    vertices = []
    for c_i in c:
        index = c_i / step
        vertices.append((step_range[math.floor(index)], step_range[math.ceil(index)]))
    return vertices


def trilin_interp(target, vertex, sample_colors):
    # calculate tri linear interpolation, first calculate the weights
    xyz_dist = []
    for i in range(3):
        dist = (target[i] - vertex[i][0]) / (vertex[i][1] - vertex[i][0]) if vertex[i][1] != vertex[i][0] else 0
        xyz_dist.append((1 - dist, dist))
    # get the value of all vertices
    eight_vertices_val = [np.array(sample_colors[c]) for c in itertools.product(*vertex)]
    x_interp, y_interp, res = [], [], 0
    for i in range(4):
        x_interp.append(eight_vertices_val[i] * xyz_dist[0][0] + eight_vertices_val[i + 4] * xyz_dist[0][1])

    for i in range(2):
        y_interp.append(x_interp[i] * xyz_dist[1][0] + x_interp[i + 2] * xyz_dist[1][1])

    res = y_interp[0] * xyz_dist[2][0] + y_interp[1] * xyz_dist[2][1]

    return res


def image_recolor(img: Image, palette_orig_lab, palette_new_lab, grid_n=16):
    k = len(palette_orig_lab)
    palettes_lab_orig = np.array([(100, 0, 0)] + palette_orig_lab + [(0, 0, 0)])
    palettes_lab_new = np.array([(100, 0, 0)] + palette_new_lab + [(0, 0, 0)])
    # get the weights of transformation of AB
    sigma = get_sigma(palettes_lab_orig[1:-1])
    lmbda = get_lambda(palettes_lab_orig[1:-1], sigma)

    # get the color of the vertex of grids
    sample_colors, sample_colors_lab = get_sample_colors(grid_n)
    # calculate linear transformation of L for all sampled colors
    L_sample = interp_L(sample_colors_lab[:, 0], palettes_lab_orig[::-1], palettes_lab_new[::-1])
    sample_colors_map = {}
    # calculate the AB transformation, and get the transformed color of all sampled colors
    for i, c_rgb in tqdm.tqdm(enumerate(sample_colors), total=len(sample_colors)):
        res = np.zeros((k, 3))
        wei = np.zeros((k, 1))
        c_lab = np.array(RGBtoLAB(c_rgb))
        for p in range(k):
            v = modify_AB(c_lab, copy.deepcopy(palettes_lab_orig[p + 1]),
                          copy.deepcopy(palettes_lab_new[p + 1]))
            wei[p] = omega(palettes_lab_orig[1:-1], lmbda[p, :], c_lab, sigma)
            res[p] = v
        # normalize weights
        wei[wei < 0] = 0
        if wei.sum() == 0:
            wei[:] = 1.
        wei /= wei.sum()
        res = np.sum(res * wei, axis=0)
        res[0] = L_sample[i]
        sample_colors_map[tuple(c_rgb)] = res

    # calculate the mapping for all colors in pic based on tri-interpolation of grid vertices
    colors = img.getcolors(img.height * img.width)
    step = 255 / (grid_n - 1)
    step_range = [round(i * (255 / (grid_n - 1)), 5) for i in range(grid_n)]

    args = []
    color_map = {}
    for cnt, c in colors:
        nearest_corners = get_nearest_vertices(c, step, step_range)
        args.append((c, nearest_corners, sample_colors_map))
    with Pool(cpu_count() - 1) as pool:
        interp_res = pool.starmap(trilin_interp, args)
    for i in range(len(colors)):
        color_map[colors[i][1]] = tuple([int(x) for x in interp_res[i]])

    # based on the color mapping calculated above, changed all color in the original picture
    result = Image.new('LAB', img.size)
    result_pixels = result.load()
    img_pixels = img.load()
    for i, j in tqdm.tqdm(itertools.product(range(img.width), range(img.height)), total=img.height * img.width):
        result_pixels[i, j] = ByteLAB(color_map[img_pixels[i, j]])

    return lab_img_to_rgb(result)
