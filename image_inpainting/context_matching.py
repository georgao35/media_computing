import math
import cv2
import numpy as np
import jittor as jt
import maxflow


def conv2d(x, w):
    N, H, W, C = x.shape
    Kh, Kw, _C, Kc = w.shape
    assert C == _C
    xx = x.reindex([N, H - Kh + 1, W - Kw + 1, Kh, Kw, C, Kc], [
        'i0',  # Nid
        'i1+i3',  # Hid+Khid
        'i2+i4',  # Wid+KWid
        'i5',  # Cid|
    ])
    ww = w.broadcast_var(xx)
    yy = xx * ww
    y = yy.sum([3, 4, 5])  # Kh, Kw, c
    return y


def to_lab(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2Lab)


def get_best_match(orig, fillup, mask):
    # get best match. Use FFT to quickly get ssd.
    #
    fillup_lab = to_lab(fillup).astype(np.float32)
    orig_lab = to_lab(orig).astype(np.float32)
    M = np.concatenate([mask[..., np.newaxis], mask[..., np.newaxis], mask[..., np.newaxis]], axis=2)
    BM = orig_lab * M
    B2M = np.sum(np.square(orig_lab), axis=2, keepdims=True) * M[:, :, :1]
    kern = jt.ones_like(B2M[..., np.newaxis])
    B2M_res = conv2d(jt.float32(B2M[np.newaxis, ...]), kern)

    A2 = np.square(fillup_lab)
    ABM_res = conv2d(jt.float32(fillup_lab[np.newaxis, ...]), jt.float32(BM[::-1, ::-1, :, np.newaxis]))
    A2M_res = conv2d(jt.float32(A2[np.newaxis, ...]), jt.float32(M[::-1, ::-1, :, np.newaxis]))
    # calculate result
    res = A2M_res + B2M_res - 2 * ABM_res
    # get min index
    _, res_w, res_h, _ = res.shape
    scene_w, scene_h, _ = orig.shape
    idx_min = np.argmin(res.data.squeeze())
    idx_min = idx_min // res_h, idx_min % res_h
    return idx_min


def get_best_match_prim(orig_scene, match):
    """
    naive implementation of finding best match
    :param orig_scene:
    :param match:
    :return:
    """
    def ssd(img1, img2):
        return np.sum(np.square(img1[img2 > 0].astype("float") - img2[img2 > 0].astype("float")))

    image_to_compare = orig_scene.copy()
    r, c, _ = match.shape
    ir, ic, _ = image_to_compare.shape
    best_x, best_y, best_sample, min_ssd = 0, 0, None, math.inf
    # iterate through all possible position
    for x in range(r - ir):
        for y in range(c - ic):
            A = match[x:x + ir, y:y + ic, :]
            # calculate ssd with mask
            current_ssd = ssd(A, image_to_compare)
            if current_ssd is None:
                pass
            elif min_ssd > current_ssd:
                min_ssd = current_ssd
                best_sample = A
                best_x = x
                best_y = y
    return best_x, best_y


def graph_cut(orig_scene, fillup_scene, mask_scene):
    diff = np.absolute(orig_scene - fillup_scene).sum(axis=2)
    w, h = mask_scene.shape
    n_nodes = (mask_scene > 0).sum()
    n_edges = n_nodes * 4

    # init graph with maxflow
    graph = maxflow.Graph[float](n_nodes, n_edges)
    nodes = graph.add_nodes(n_nodes)
    nodemap, nodemap_inv = [], {}
    # record the mapping between node index and pixel position
    for y in range(w):
        for x in range(h):
            if mask_scene[y, x] > 0:
                nodemap.append((y, x))
                nodemap_inv[(y, x)] = len(nodemap_inv)

    # construct graph based on the part each pixel belongs
    # the value of edge is based on the difference of pixel value
    for y in range(w):
        for x in range(h):
            if mask_scene[y, x] == 0:
                continue
            if y + 1 < w and mask_scene[y+1, x]:
                value = diff[y, x] + diff[y+1, x]
                graph.add_edge(nodes[nodemap_inv[(y, x)]], nodes[nodemap_inv[(y+1, x)]], value, value)
            if x + 1 < h and mask_scene[y, x+1]:
                value = diff[y, x] + diff[y, x+1]
                graph.add_edge(nodes[nodemap_inv[(y, x)]], nodes[nodemap_inv[(y, x+1)]], value, value)
            if mask_scene[y, x] == 1:
                graph.add_tedge(nodes[nodemap_inv[(y, x)]], 1e10, 0)
            if mask_scene[y, x] == 2:
                graph.add_tedge(nodes[nodemap_inv[(y, x)]], 0, 1e10)

    flow = graph.maxflow()
    segmentation_result = 1 + np.array(list(map(graph.get_segment, nodes)))
    seg_pic = np.zeros_like(mask_scene)
    for i, result in enumerate(segmentation_result):
        seg_pic[nodemap[i][0], nodemap[i][1]] = result
    return seg_pic, flow
