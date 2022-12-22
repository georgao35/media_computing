import jittor
import cv2
import numpy as np


def ssd(img1, img2):
    return np.sum(np.square(img1[img2 > 0].astype("float") - img2[img2 > 0].astype("float")))


def get_best_match(orig, fillup):
    pass


def graph_cut(img1, img2, mask):
    pass
