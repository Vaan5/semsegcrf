import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

import kitti_labels
import utils


def _search_right(i, j, img, current_pixel, max_counter, prob_img, changed_img, search_width):
    n_i = i
    n_j = j + 1
    if n_j < img.shape[1] and current_pixel != img[n_i,n_j]:
        positions = []
        index = n_j - 1
        counter = 0
        while index >= 0 and img[i, index] == current_pixel and counter < max_counter:
            positions.append(index)
            index -= 1
            counter += 1

        positions = list(reversed(positions))
        if img[n_i, n_j] not in kitti_labels.DONT_CARE_LABELS:
            index = n_j
            counter = 0
            while index < img.shape[1] and img[i, index] == img[n_i, n_j] and counter < max_counter:
                positions.append(index)
                index += 1
                counter += 1

        num = len(positions)
        delta = int(search_width / num)

        in1 = search_width
        in2 = search_width - delta * num
        for ind in positions:
            changed_img[i, ind, :] = (0,0,0)
            prob_img[i, ind, current_pixel] += in1
            in1 -= delta
            if img[n_i, n_j] not in kitti_labels.DONT_CARE_LABELS:
                prob_img[i, ind, img[n_i, n_j]] += in2
                in2 += delta


def _search_top(i, j, img, current_pixel, max_counter, prob_img, changed_img, search_width):
    n_i = i - 1
    n_j = j
    if n_i >= 0 and current_pixel != img[n_i,n_j]:
        positions = []
        index = n_i + 1
        counter = 0
        while index < img.shape[0] and img[index, n_j] == current_pixel and counter < max_counter:
            positions.append(index)
            index += 1
            counter += 1

        positions = list(reversed(positions))

        if img[n_i, n_j] not in kitti_labels.DONT_CARE_LABELS:
            index = n_i
            counter = 0
            while index >= 0 and img[index, n_j] == img[n_i, n_j] and counter < max_counter:
                positions.append(index)
                index -= 1
                counter += 1

        num = len(positions)
        delta = int(search_width / num)

        in1 = search_width
        in2 = search_width - delta * num
        for ind in positions:
            changed_img[ind, n_j, :] = (0,0,0)
            prob_img[ind, n_j, current_pixel] += in1
            in1 -= delta
            if img[n_i, n_j] not in kitti_labels.DONT_CARE_LABELS:
                prob_img[ind, n_j, img[n_i, n_j]] += in2
                in2 += delta


def _search_top_right(i, j, img, current_pixel, max_counter, prob_img, changed_img, search_width):
    n_i = i - 1
    n_j = j + 1
    if n_i >= 0 and n_j < img.shape[1] and current_pixel != img[n_i,n_j]:
        positions = []
        index1 = n_i + 1
        index2 = n_j - 1
        counter = 0
        while index1 < img.shape[0] and index2 >= 0 and img[index1, index2] == current_pixel and counter < max_counter:
            positions.append((index1, index2))
            index1 += 1
            index2 -= 1
            counter += 1

        positions = list(reversed(positions))

        if img[n_i, n_j] not in kitti_labels.DONT_CARE_LABELS:
            index1 = n_i
            index2 = n_j
            counter = 0
            while index1 >= 0 and index2 < img.shape[1] and img[index1, index2] == img[n_i, n_j] and counter < max_counter:
                positions.append((index1, index2))
                index1 -= 1
                index2 += 1
                counter += 1

        num = len(positions)
        delta = int(search_width / num)

        in1 = search_width
        in2 = search_width - delta * num
        for ind1, ind2 in positions:
            changed_img[ind1, ind2, :] = (0,0,0)
            prob_img[ind1, ind2, current_pixel] += in1
            in1 -= delta
            if img[n_i, n_j] not in kitti_labels.DONT_CARE_LABELS:
                prob_img[ind1, ind2, img[n_i, n_j]] += in2
                in2 += delta


def _search_top_left(i, j, img, current_pixel, max_counter, prob_img, changed_img, search_width):
    n_i = i - 1
    n_j = j - 1
    if n_i >= 0 and n_j >= 0 and current_pixel != img[n_i,n_j]:
        positions = []
        index1 = n_i + 1
        index2 = n_j + 1
        counter = 0
        while index1 < img.shape[0] and index2 < img.shape[1] and img[index1, index2] == current_pixel and counter < max_counter:
            positions.append((index1, index2))
            index1 += 1
            index2 += 1
            counter += 1

        positions = list(reversed(positions))

        if img[n_i, n_j] not in kitti_labels.DONT_CARE_LABELS:
            index1 = n_i
            index2 = n_j
            counter = 0
            while index1 >= 0 and index2 >= 0 and img[index1, index2] == img[n_i, n_j] and counter < max_counter:
                positions.append((index1, index2))
                index1 -= 1
                index2 -= 1
                counter += 1

        num = len(positions)
        delta = int(search_width / num)

        in1 = search_width
        in2 = search_width - delta * num
        for ind1, ind2 in positions:
            changed_img[ind1, ind2, :] = (0,0,0)
            prob_img[ind1, ind2, current_pixel] += in1
            in1 -= delta
            if img[n_i, n_j] not in kitti_labels.DONT_CARE_LABELS:
                prob_img[ind1, ind2, img[n_i, n_j]] += in2
                in2 += delta


def find_edges(img, input, output, binary=False, textual=False, interactive=False, changed="", n8=False, search_width=16):
    # create necessary image copies
    changed_img = np.copy(img)
    img_copy = np.copy(img)
    img = np.apply_along_axis(kitti_labels.get_label, 2, img)
    prob_img = np.zeros((img.shape[0], img.shape[1], kitti_labels.NUMBER_OF_LABELS))

    # iterate through image and look at top and right neighbours (8-neighbourhood)
    for i in xrange(img.shape[0]):
        for j in xrange(img.shape[1]):
            current_pixel = img[i,j]

            if current_pixel in kitti_labels.DONT_CARE_LABELS:
                # put uniform distribution if the label is unknown
                prob_img[i,j,:] = search_width
                continue

            # set the probability for the read label
            if np.sum(prob_img[i,j,:]) == 0:
                prob_img[i,j,current_pixel] = search_width

            # seach range in each direction
            max_counter = search_width / 2

            # check right neighbour
            _search_right(i, j, img, current_pixel, max_counter, prob_img, changed_img, search_width)
            # check top
            _search_top(i, j, img, current_pixel, max_counter, prob_img, changed_img, search_width)

            if n8:
                # check top right
                _search_top_right(i, j, img, current_pixel, max_counter, prob_img, changed_img, search_width)
                # check top left
                _search_top_left(i, j, img, current_pixel, max_counter, prob_img, changed_img, search_width)

    if interactive:
        plt.imshow(changed_img)
        plt.figure()
        plt.imshow(img_copy)
        plt.show()

    if changed:
        print "Saving image with change pixels - {}".format(changed)
        utils.save_image(changed_img, changed)

    if binary:
        print "Saving potentials - binary format - {}".format(output + ".bin")
        utils.save_prob_map_bin(prob_img, output + ".bin", True)

    if textual:
        print "Saving potentials - textual format - {}".format(output + ".txt")
        utils.save_prob_map_txt(prob_img, output + ".txt", True)


def find_edges_in_multiple_images(input_, output, binary=False, textual=False, interactive=False, changed="", n8=False, search_width=16):
    # check parameter validity
    if not os.path.isdir(input_):
        raise ValueError("Invalid directory path {}".format(input))

    if not os.path.isdir(output):
        os.makedirs(output)

    if changed and not os.path.isdir(changed):
        os.makedirs(changed)

    # get all input images
    input_images = os.listdir(input_)

    # process each image individually
    for file in input_images:
        try:
            img = utils.load_image(os.path.join(input_, file))
            # file is an image - continue
            print file
            filename = file
            index = file.rfind('.')
            if index != -1:
                filename = filename[0:index]

            changed_ = changed
            if changed:
                changed_ = os.path.join(changed_, file)

            find_edges(img, os.path.join(input_, file), os.path.join(output, filename), binary=binary, 
                textual=textual, interactive=interactive, changed=changed_, n8=n8, search_width=search_width)
        except IOError:
            print "{} - not a valid image file -> skip".format(file)
            continue


def main(input_, output, directory=False, binary=False, textual=False, interactive=False, changed="", n8=False, search_width=16):
    if directory:
        find_edges_in_multiple_images(input_, output, binary=binary, textual=textual, interactive=interactive, changed=changed, n8=n8, search_width=search_width)
    else:
        try:
            img = utils.load_image(input_)
            find_edges(img, input_, output, binary=binary, textual=textual, interactive=interactive, changed=changed, n8=n8, search_width=search_width)
        except IOError:
            print "{} - not a valid image file -> skip".format(file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help="path to the input image or directory (if -d flag is present)")
    parser.add_argument('output', help="path to the output image or directory (if -d flag is present)")
    parser.add_argument('-d', help="flag used to specify that input and output arguments are directories", action="store_true")
    parser.add_argument('-b', help="potentials in binary format", action="store_true", default=False)
    parser.add_argument('-t', help="potentials in textual format", action="store_true", default=False)
    parser.add_argument('-i', help="program stops after processing each image and displays it", action="store_true", default=False)
    parser.add_argument('--changed', nargs="?", default="", help="path to output image or directory (if -d flag is present) showing fuzzified edges")
    parser.add_argument('-n8', action="store_true", default=False, help="use 8-neighbourhood")
    parser.add_argument('--search_width', type=int, action="store", default=16, help="edge fuzzy-zone width")
    args = parser.parse_args()

    main(args.input, args.output, args.d, args.b, args.t, args.i, args.changed, args.n8, args.search_width)