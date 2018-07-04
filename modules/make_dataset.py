import os

import numpy as np
from PIL import Image
from scipy.misc import imresize

FOLDER_NAME = '../images/'
DATA_LABEL = ['A', 'L', 'R', 'T', 'W']
USE_REMOVE_LIST = False
NEW_IMAGE_ROW = 97
NEW_IMAGE_COL = 93


def save_train_test_label():
    folder_list = os.listdir(FOLDER_NAME)
    first_dict = dict()
    second_dict = dict()
    label_dict = dict()
    remove_set = get_remove_set()
    for folder in folder_list:
        print(folder)
        for p in os.listdir(os.path.join(FOLDER_NAME, folder)):
            if USE_REMOVE_LIST and p[1:8] in remove_set:
                continue

            tag = p[1:5]
            if p.endswith('png'):
                path = os.path.join(FOLDER_NAME, folder, p)
                im = read_image_file(path)
                im = imresize(im[:-32, :], (NEW_IMAGE_ROW, NEW_IMAGE_COL))
                im = im.reshape((NEW_IMAGE_ROW * NEW_IMAGE_COL,))
                if p.startswith('f'):
                    first_dict[tag] = im
                elif p.startswith('s'):
                    second_dict[tag] = im

            elif p.endswith('txt'):
                for line in open(os.path.join(FOLDER_NAME, folder, p)):
                    if line.startswith('Class:'):
                        class_number = DATA_LABEL.index(line[-2:].strip())
                        label_dict[tag] = class_number

    l1 = sorted(list(first_dict.keys()))
    l2 = sorted(list(second_dict.keys()))
    assert l1 == l2

    first_arr = np.zeros((1650 if USE_REMOVE_LIST else 2000, NEW_IMAGE_ROW * NEW_IMAGE_COL), dtype=np.uint8)
    second_arr = np.zeros((1650 if USE_REMOVE_LIST else 2000, NEW_IMAGE_ROW * NEW_IMAGE_COL), dtype=np.uint8)
    label_arr = np.zeros((1650 if USE_REMOVE_LIST else 2000,), dtype='<U1')

    for i, tag in enumerate(l1):
        first_arr[i] = first_dict[tag]
        second_arr[i] = second_dict[tag]
        label_arr[i] = label_dict[tag]

    path = '../dataset/images_{}_{}.npz'.format(NEW_IMAGE_ROW, NEW_IMAGE_COL)
    np.savez_compressed(path, sample1=first_arr, sample2=second_arr, label=label_arr)
    return path


def get_remove_set():
    res = set()
    for line in open('remove_list.txt'):
        res.add(line.strip())
    return res


def read_image_file(path):
    im = Image.open(path)
    pixels = im.getdata()
    data = np.array(list(pixels))
    return data.reshape((512, 512))


if __name__ == '__main__': save_train_test_label()
