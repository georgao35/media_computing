import cv2
import numpy as np
import jittor as jt
# import maxflow


def ssd(img1, img2):
    return np.sum(np.square(img1[img2 > 0].astype("float") - img2[img2 > 0].astype("float")))


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
    # scale = [0.81, 0.9, 1.]
    # get best match. Use FFT to quickly get ssd
    fillup_lab = to_lab(fillup).astype(np.float32)
    orig_lab = to_lab(orig).astype(np.float32)
    M = (mask > 0)
    BM = orig_lab * M
    B2M = np.sum(np.square(orig_lab), axis=2, keepdims=True) * M[:, :, :1]
    kern = jt.ones_like(B2M[..., np.newaxis])
    B2M_res = conv2d(jt.float32(B2M[np.newaxis, ...]), kern)

    A2 = np.square(fillup_lab)
    A2M_res = conv2d(jt.float32(A2[np.newaxis, ...]), jt.float32(M[::-1, ::-1, :, np.newaxis]))
    ABM_res = conv2d(jt.float32(fillup_lab[np.newaxis, ...]), jt.float32(BM[::-1, ::-1, :, np.newaxis]))

    res = A2M_res + B2M_res - 2 * ABM_res

    res_w, res_h, _, _ = res.shape
    scene_w, scene_h, _ = orig.shape
    idx_min = np.argmin(res.data)
    idx_min = idx_min // res_h, idx_min % res_h
    return idx_min


def get_best_match_prim(orig_scene, match):
    image_to_compare = orig_scene.copy()
    r, c, _ = match.shape
    ir, ic, _ = image_to_compare.shape
    min_ssd = None
    for x in range(r):
        for y in range(c):
            # compare to sample image to start off with...
            # mse(imageA, imageB, mask=0)
            # assume x,y is top left corner,
            imageA = match[x:x + ir, y:y + ic, :]

            if imageA.shape[0] != ir or imageA.shape[1] != ic:
                continue
            # add the mask
            current_ssd = ssd(imageA, image_to_compare)
            if current_ssd == None:
                pass
            elif min_ssd == None:
                min_ssd = current_ssd
                best_sample = imageA
                best_x = x
                best_y = y
            elif min_ssd > current_ssd:
                min_ssd = current_ssd
                best_sample = imageA
                best_x = x
                best_y = y
    return best_x, best_y, best_sample, min_ssd
    # 199, 0, 1132227401.0


def graph_cut(orig_scene, fillup_scene, mask_scene):
    diff = np.absolute(orig_scene - fillup_scene).sum(axis=2)
    shape = mask_scene.shape
