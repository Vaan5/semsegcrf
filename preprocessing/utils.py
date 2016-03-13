import numpy as np
import os
from PIL import Image


def save_prob_map_txt(prob_img, file, normalize=False):
    if normalize:
        prob_img = prob_img.astype(np.float)
        prob_img = normalize_prob_img(prob_img)

    with open(file, "w") as f:
        s = ""
        for i in xrange(prob_img.shape[0]):
            for j in xrange(prob_img.shape[1]):
                for k in xrange(prob_img.shape[2]):
                    s += str(prob_img[i,j,k]) + " "
            s = s[0:-1]
            s += "\n"
        f.write(s)


def save_prob_map_bin(prob_img, file, normalize=False):
    if normalize:
        prob_img = prob_img.astype(np.float)
        prob_img = normalize_prob_img(prob_img)

    with open(file, "wb") as f:
        prob_img.tofile(f)


def normalize_prob_img(img):
    return np.apply_along_axis(lambda t: t / np.sum(t), 2, img)


def load_image(path):
    img = Image.open(path)
    return np.array(img)


def save_image(img_array, path):
    img = Image.fromarray(img_array)
    img.save(path)