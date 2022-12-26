import numpy as np
import jittor as jt
import jsparse.nn.functional as F
from tqdm import tqdm

from IPython import embed


def simple_test(orig, match, segment_mask, left_top_pos):
    pass


def adjacency(x, y):
    return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]


def in_mask(idx, mask):
    return mask[idx] != 0


def on_edge(idx, mask):
    if not in_mask(idx, mask):
        return False
    w, h = mask.shape
    for adjacent in adjacency(*idx):
        if not (0 <= adjacent[0] < w and 0 <= adjacent[1] < h):
            continue
        if not in_mask(adjacent, mask):
            return True
    return False


def poisson_matrix(points, mask):
    n = len(points)
    w, h = mask.shape[0], mask.shape[1]
    indices = []
    vals = []
    point_map_rev = {}
    for i, point in enumerate(points):
        point_map_rev[point] = i

    for i, point in tqdm(enumerate(points), total=n):
        # indices.append((i, i))
        # vals.append(4)
        for adjacent in adjacency(*point):
            if 0 <= adjacent[0] < w and 0 <= adjacent[1] < h and in_mask(adjacent, mask):
                j = point_map_rev[adjacent]
                indices.append((i, j))
                vals.append(1)

    return {'indices': np.array(indices), 'vals': np.array(vals)}


def get_b(points, src, target, mask, left_top_pos):
    n = len(points)
    src = src.copy().astype(np.float32)
    target = np.copy(target).astype(np.float32)
    w, h = src.shape
    # w, h, c = src.shape
    # b = np.zeros((n, c))
    b = np.zeros(n)
    for idx, point in tqdm(enumerate(points), total=n):
        i, j = point
        res = 4 * src[i, j]
        for adjacent in adjacency(i, j):
            if 0 <= adjacent[0] < w and 0 <= adjacent[1] < h:
                res -= src[adjacent]
        # if on edge, make constraint to be the target value
        if on_edge(point, mask):
            for adjacent in adjacency(*point):
                if 0 <= adjacent[0] < w and 0 <= adjacent[1] < h and not in_mask(adjacent, mask):
                    point_in_target = (adjacent[0] + left_top_pos[0], adjacent[1] + left_top_pos[1])
                    res += target[point_in_target]
        b[idx] = res
    return b[..., np.newaxis]


def sparse_matmul(rows, cols, values, vec):
    size = len(vec)
    return F.spmm(rows=rows,
                  cols=cols,
                  vals=values,
                  size=(size, size),
                  mat=vec, cuda_spmm_alg=1)


def laplace_solver(A, b, x0, iter_num=6000):
    indices, vals = A['indices'], A['vals']
    with jt.no_grad():
        rows = jt.int32(indices[:, 0])
        cols = jt.int32(indices[:, 1])
        vals = jt.float32(vals)
        x = jt.float32(x0)
        b = jt.array(b)
        # while True:
        for i in tqdm(range(iter_num)):
            res = sparse_matmul(rows, cols, vals, x)
            x_prime = (res + b) / 4
            delta = x_prime.data - x.data
            if np.linalg.norm(delta) < 0.1:
                break
            x = x_prime
            # print(np.linalg.norm(delta))
    x = x.numpy()
    x[x > 255] = 255
    x[x < 0] = 0
    return x.astype(np.uint8)


def poisson_blending(orig: np.ndarray, match: np.ndarray, segment_mask: np.ndarray, left_top_pos):
    # get div of match picture
    roi_w, roi_h, c = match.shape
    img_w, img_h, img_c = orig.shape

    segmented_indices = np.nonzero(segment_mask)
    x0 = match[segmented_indices]
    segmented_indices = list(zip(segmented_indices[0], segmented_indices[1]))
    num_pixels = len(segmented_indices)
    # get laplace matrix A
    A = poisson_matrix(segmented_indices, segment_mask)
    # get b
    bs = [get_b(segmented_indices, match[:, :, i], orig[:, :, i], segment_mask, left_top_pos) for i in range(c)]
    # b = get_b(segmented_indices, match, orig, segment_mask, left_top_pos)
    # solve x=A\b
    # x = np.concatenate([laplace_solver(A, b, x0[:, i:i+1]) for i, b in enumerate(bs)], axis=1)  # by channel
    x = laplace_solver(A, np.concatenate(bs, axis=1), x0)
    composite = orig.copy().astype(np.uint8)
    for i, point in enumerate(segmented_indices):
        abs_point = (point[0]+left_top_pos[0], point[1]+left_top_pos[1])
        composite[abs_point] = x[i]
    return composite
