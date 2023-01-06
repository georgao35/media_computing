import copy
import os
import cv2
import numpy as np
import time
import pickle
import jittor as jt
from IPython import embed

from context_matching import get_best_match, get_best_match_prim, graph_cut
from picture_merging import poisson_blending


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
    mask = np.copy(mask)[:, :, 0]
    w, h = mask.shape

    mask = 255 - mask
    mask = (mask > 0).astype('uint8')
    # dilation with layer to distinguish two patches during graph cut
    mask_boundary = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8))
    mask_dilated = cv2.dilate(mask_boundary, np.ones((3, 3), dtype=np.uint8), iterations=local_context_size - 2)
    mask_res = cv2.dilate(mask_dilated, np.ones((3, 3), dtype=np.uint8))

    mask_res *= 3
    mask_res[(mask_res > 0) & (mask_dilated == 0)] = 1
    mask_res[(mask_boundary > 0) & (mask == 0)] = 2
    masked_range = np.where(mask_res > 0)
    mask_res[mask > 0] = 0

    # x_min = max(min(masked_range[0]) - local_context_size, 0)
    # x_max = min(max(masked_range[0]) + local_context_size, w-1)
    # y_min = max(min(masked_range[1]) - local_context_size, 0)
    # y_max = min(max(masked_range[1]) + local_context_size, h-1)
    x_min, x_max, y_min, y_max = min(masked_range[0]), max(masked_range[0])+1, min(masked_range[1]), max(masked_range[1])+1

    orig_window = orig[x_min:x_max, y_min:y_max]
    mask_res = mask_res[x_min:x_max, y_min:y_max]
    mask_orig = mask[x_min:x_max, y_min:y_max]

    # cv2.imshow('res', mask_res * 83)
    # cv2.imshow('orig', mask.astype('uint8') * 255)
    # cv2.imshow('dilate', mask_dilated.astype('uint8') * 255)
    # cv2.imshow('boundary', mask_boundary.astype('uint8') * 255)
    # cv2.waitKey(0)

    return orig_window, mask_orig, mask_res, (x_min, x_max, y_min, y_max)


def tik():
    global time_base
    time_base = time.time()


def tok(name="(undefined)"):
    print("Time used in {}: {}".format(name, time.time() - time_base))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_i = 4
    Load = True
    orig_src, mask_src, fillups = load_imgs(data_i)
    orig_scene, mask_orig, mask_context, scene_range = preprocess(orig_src, mask_src)
    best_match_loc, candidate_scenes = [], []
    roiw, roih, _ = orig_scene.shape

    jt.flags.use_cuda = 0
    if os.path.exists(f'match/{data_i}/best_match_loc.pkl') and Load:
        with open(f'match/{data_i}/best_match_loc.pkl', 'rb') as f:
            best_match_loc = pickle.load(f)
        for i in range(len(best_match_loc)):
            match_scene = np.load(f'match/{data_i}/best_scene_{i}.npy')
            # padding: some matched parts are not the same as the mask
            if roiw > match_scene.shape[0]:
                match_scene = np.pad(match_scene, [(0, roiw - match_scene.shape[0]), (0, 0), (0, 0)],
                                     'reflect')
            if roih > match_scene.shape[1]:
                match_scene = np.pad(match_scene, [(0, 0), (0, roih - match_scene.shape[1]), (0, 0)],
                                     'reflect')
            match_scene = match_scene[:roiw, :roih]
            candidate_scenes.append(match_scene)
    else:
        for i, fillup in enumerate(fillups):
            scale = max(roiw / fillup.shape[0], roih / fillup.shape[1])
            if scale > 1.0:
                # if the candidate is smaller, make it bigger
                fillup = cv2.resize(fillup, (0, 0), fx=scale * 1.1, fy=scale * 1.1)
            tik()
            idx_min = get_best_match(orig_scene, fillup, (mask_context > 0))
            # idx_min = get_best_match_prim(orig_scene * (mask_dilated > 0), fillups[0])
            tok("jittor FFT")
            best_match_loc.append(idx_min)
            best_match_scene = fillup[idx_min[0]: idx_min[0] + roiw,
                                      idx_min[1]: idx_min[1] + roih]
            candidate_scenes.append(best_match_scene)
            np.save(f'match/{data_i}/best_scene_{i}.npy', best_match_scene)
            print(idx_min, best_match_scene.shape)
        with open(f'match/{data_i}/best_match_loc.pkl', 'wb') as f:
            pickle.dump(best_match_loc, f)
    # a = get_best_match_prim(orig_scene * (mask_dilated > 0), fillups[0])
    # tok("naive")
    cv2.imshow('orig_scene', orig_scene)
    # print(a)
    # generate graph cut
    if os.path.exists(f'match/{data_i}/segments.pkl') and Load:
        with open(f'match/{data_i}/segments.pkl', 'rb') as f:
            segments = pickle.load(f)
    else:
        segments = []
        for candidate in candidate_scenes:
            segment_res, flow_cost = graph_cut(orig_scene, candidate, mask_context)
            segment_res[mask_orig > 0] = 2
            segments.append(segment_res)
        with open(f'match/{data_i}/segments.pkl', 'wb') as f:
            pickle.dump(segments, f)
    # cv2.imshow('segmentation', segment_res * 126)
    # cv2.waitKey(0)
    # merge
    print(scene_range)
    jt.flags.use_cuda = 1
    for i in range(len(candidate_scenes)):
        print(f'blending: {i}')
        segment_res = segments[i]
        poisson_blend_mask = (segment_res == 2).astype('uint8') * 255
        blend_res = poisson_blending(orig_src, candidate_scenes[i], poisson_blend_mask, (scene_range[0], scene_range[2]))
        cv2.imwrite(f'data/output{data_i}/{i}.png', blend_res)
    # cv2.imshow('own_blend', blend_res)
    # cv2.waitKey(0)

    # cv2.waitKey(0)
