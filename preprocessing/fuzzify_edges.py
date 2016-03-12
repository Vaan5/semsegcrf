import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import kitti_labels

def save_prob_map(prob_img):
    with open("file.txt", "w") as f:
        for i in xrange(prob_img.shape[0]):
            for j in xrange(prob_img.shape[1]):
                sum_ = np.sum(prob_img[i,j,:])
                print i, j
                print prob_img[i,j,:]
                prob_img[i,j,:] /= sum_
                print prob_img[i,j,:]
                f.write(str(prob_img[i,j,:]) + "\n")


def load_image(path):
    img = Image.open(args.input_img)
    return np.array(img)

def find_edges(img):
    changed = np.copy(img)
    im2 = np.copy(img)
    img = np.apply_along_axis(kitti_labels.getLabel, 2, img)
    prob_img = np.zeros((img.shape[0], img.shape[1], kitti_labels.NUMBER_OF_LABELS))

    # iterate through image and look at top and right neighbours (8-neighbourhood)
    for i in xrange(img.shape[0]):
        for j in xrange(img.shape[1]):
            current_pixel = img[i,j]

            if current_pixel in kitti_labels.DONT_CARE_LABELS:
                # put uniform distribution
                print "TU"
                prob_img[i,j,:] = 16
                print prob_img[i,j]
                continue

            prob_img[i,j,current_pixel] = 16

            # check right
            n_i = i
            n_j = j + 1

            if n_j < img.shape[1] and current_pixel != img[n_i,n_j]:
                positions = []
                index = n_j - 1
                counter = 0
                while index >= 0 and img[i, index] == current_pixel and counter < 8:
                    positions.append(index)
                    index -= 1
                    counter += 1

                positions = list(reversed(positions))
                if img[n_i, n_j] not in kitti_labels.DONT_CARE_LABELS:
                    index = n_j
                    counter = 0
                    while index < img.shape[1] and img[i, index] == img[n_i, n_j] and counter < 8:
                        positions.append(index)
                        index += 1
                        counter += 1

                num = len(positions)
                delta = int(16 / num)

                in1 = 16
                in2 = 16 - delta * num
                for ind in positions:
                    changed[i, ind, :] = (0,0,0)
                    prob_img[i, ind, current_pixel] = in1
                    if img[n_i, n_j] not in kitti_labels.DONT_CARE_LABELS:
                        prob_img[i, ind, img[n_i, n_j]] = in2
                    in2 += delta
                    in1 -= delta

                print current_pixel, img[n_i, n_j]
                print len(positions)
                print positions
                for ind in positions:
                    print ind
                    print prob_img[n_i, ind, :]

            # check top
            # check top right
            # check top left

    plt.imshow(changed)
    plt.figure()
    plt.imshow(im2)
    plt.show()
    save_prob_map(prob_img)

def main(args):
    img = load_image(args.input_img)
    find_edges(img)
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_img', help="path to input image")
    args = parser.parse_args()

    main(args)