import copy
import os
import cv2
import numpy as np
import time

from context_matching import get_best_match, get_best_match_prim


def load_imgs(idx, base_dir='data'):
    orig_name, mask_name = os.path.join(base_dir, f'input{idx}.jpg'), os.path.join(base_dir, f'input{idx}_mask.jpg')
    orig = cv2.imread(orig_name)
    mask = cv2.imread(mask_name)
    src_dir = os.path.join(base_dir, f'input{idx}')
    fill_ups = sorted(os.listdir(src_dir))
    print(fill_ups)
    return orig, mask, [cv2.imread(os.path.join(src_dir, fill)) for fill in fill_ups]


def preprocess(orig, mask, local_context_size=80):
    orig = np.copy(orig)
    mask = np.copy(mask)
    w, h, c = mask.shape

    masked_range = np.where(mask == 0)
    x_min = max(min(masked_range[0]) - local_context_size, 0)
    x_max = min(max(masked_range[0]) + local_context_size, w-1)
    y_min = max(min(masked_range[1]) - local_context_size, 0)
    y_max = min(max(masked_range[1]) + local_context_size, h-1)

    orig_win = orig[x_min:x_max, y_min:y_max]
    mask = mask[x_min:x_max, y_min:y_max]
    mask_dilated = cv2.dilate(255 - mask, np.ones((local_context_size, local_context_size), dtype=np.uint8))

    # cv2.imshow('dilate', orig_win * (mask_dilated > 0))
    # cv2.imshow('orig', mask)
    # cv2.waitKey(0)

    return orig_win, mask, mask_dilated


def tik():
    global time_base
    time_base = time.time()


def tok(name="(undefined)"):
    print("Time used in {}: {}".format(name, time.time() - time_base))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    orig_src, mask_src, fillups = load_imgs(1)
    orig_scene, mask_scene, mask_dilated = preprocess(orig_src, mask_src)
    tik()
    a = get_best_match(orig_scene, fillups[0], mask_dilated)
    tok("jittor FFT")
    # a = get_best_match_prim(orig_scene * (mask_dilated > 0), fillups[0])
    print(a)
