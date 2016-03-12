import numpy as np
import os


def save_prob_map_txt(prob_img, file, normalize=False):
    if normalize:
        prob_img = prob_img.astype(np.float)
        prob_img = normalize(prob_img)

    with open(file, "w") as f:
        s = ""
        for i in xrange(prob_img.shape[0]):
            for j in xrange(prob_img.shape[1]):
                for k in xrange(prob_img.shape[2]):
                    s += prob_img[i,j,k] + " "
            s[-1] = "\n"
        f.write(s)


def save_prob_map_bin(prob_img, file, normalize=False):
    if normalize:
        prob_img = prob_img.astype(np.float)
        prob_img = normalize(prob_img)

    with open(file, "wb") as f:
        prob_img.tofile(f)


def normalize(img):
    assert(img.shape[2] == 3)
    return np.apply_along_axis(lambda t: t / np.sum(t), 2, img)
