import itertools
import random
import math
import numpy as np

from typing import Tuple, List
from PIL import Image, ImageCms

from utils import *


def update_weight(x, target):
    dist = math.sqrt(sum([(i - j) ** 2 for i, j in zip(x, target)]))
    return 1 - math.exp(-dist * dist / 6400)


def k_means(means, bins, k=5, max_iter=1000, add_black=True):
    if add_black:
        # add black into means
        means.append((0, -128, -128))
    means = np.array(means)
    mean_cnt = means.shape[0]
    # k-means loop
    for _ in range(max_iter):
        cluster_cnt = np.zeros(mean_cnt)
        cluster_sum = np.zeros((mean_cnt, 3))
        for color, cnt in bins.items():
            # for each point, calculate its nearest mean and record
            color_np = np.array(RGBtoLAB(color))
            dist = np.sum(np.square(color_np - means), axis=1)
            clust_idx = dist.argmin()
            cluster_sum[clust_idx] += color_np * cnt
            cluster_cnt[clust_idx] += cnt
        # calculate the mean as new group of centers
        means_update = np.nan_to_num(cluster_sum / cluster_cnt.reshape((6, 1)), nan=0.0)
        if add_black:
            means_update[-1] = np.array((0, -128, -128))
        # if converged, stop the loop
        if (means_update == means).all():
            break
        else:
            means = means_update
    # sort the collected means in ascending sequence
    sort_idx = np.argsort(means, axis=0)[:, 0][::-1]
    return means[sort_idx][:k]


def get_means(bins, k=5, random_init=True):
    if random_init:
        # original k-means initialization
        return random.choices(list(bins.keys()), k=k)
    else:
        # improved k-means initialization
        bins_count = sorted([(cnt, RGBtoLAB(color)) for color, cnt in bins.items()], reverse=True)
        # select k init means based on distance in LAB space
        means = []
        for i in range(k):
            origin = bins_count.pop(0)
            means.append(origin[1])
            # update weights and sort
            bins_count = sorted(list(map(lambda x: (x[0] * update_weight(x[1], means[-1]), x[1]), bins_count)),
                                reverse=True)
        return means


def dividing_bin(image: Image.Image, n=16):
    # calculate the rgb color space bins and count
    colors: List[Tuple[int, Tuple[int]]] = image.getcolors(image.width * image.height)
    bins = {}
    for cnt, color in colors:
        bins[color] = cnt
    channel_len = 256 // n
    wght_sum, weights = {}, {}
    for idx in itertools.product(range(n), repeat=3):
        weights[idx] = 0
        wght_sum[idx] = [0, 0, 0]
    for color in bins:
        idx = tuple([c // channel_len for c in color])
        weights[idx] += bins[color]
        for i in range(3):
            wght_sum[idx][i] += color[i] * bins[color]
    # calculate color means and weights
    res = {}
    for idx in weights:
        if weights[idx] == 0:
            continue
        for i in range(3):
            wght_sum[idx][i] /= weights[idx]
        res[tuple(wght_sum[idx])] = weights[idx]

    return res


def build_palettes(image: Image.Image, k=5, bin_n=16):
    bins = dividing_bin(image, bin_n)

    means = get_means(bins, k=k, random_init=False)

    res = k_means(means, bins)
    print(res)
    return [color for color in res]
