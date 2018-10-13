# -*- coding: UTF-8 -*-

import os
import io

import numpy as np
from matplotlib import image
import tensorflow as tf
from skimage import io as skio

# 读取一张图片
def read_single_img(class_path, img_name):
    """
    Read an image and store in ndarray.

    Arguments
    ---
    im_path--image path on your driver

    Returns
    ---
    im_array--ndarray, of shape (m, n, 3) for RGB image
    """

    img_path = class_path + '/' + img_name
    img = image.imread(img_path)

    return img

def get_class_name(dataset_path, split_name):
    all_class_name = []

    if split_name == 'train':
        split_file_path = dataset_path + '/' + split_name + 'classes.txt'
    elif split_name == 'valid':
        split_file_path = dataset_path + '/' + split_name + 'classes.txt'
    elif split_name == 'test':
        split_file_path = dataset_path + '/' + split_name + 'classes.txt'
    elif split_name == 'all':
        split_file_path = dataset_path + '/' + 'classes.txt'
    else:
        print("没有叫做 " + str(split_name) + " 的数据划分！！！\n")
        return

    split_file = io.open(split_file_path, 'r')

    class_name = ' '
    while class_name != '':
        class_name = split_file.readline().rstrip('\n')
        # 确保没有加入无效的名称
        if len(class_name) != 0:
            all_class_name.append(class_name)

    split_file.close()

    return all_class_name

def read_class_img(split_path, class_name):
    """Read a class of images. e.g. antelope

    Args:
        class_name: which class of images you want to read.

    Returns:
        cls_img_array: a class of image in numpy array, of shape (num of images, height, width, channels)
    """
    class_path = split_path + '/' + class_name
    all_img_name = os.listdir(class_path)
    
    img_count = 1
    for img_name in all_img_name:
        # 读取当前图片
        img = read_single_img(cls_path, img_name)

        # 需要考虑读的是不是当前类别的第一张图片
        if img_count == 1:
            a_class_img = img
        else:
            a_class_img = np.vstack((a_class_img, img))

        img_count += 1

    return a_class_img

# 读取一个数据划分的图片
def read_a_split_img(dataset_path, split_name):
    """Read a data split of images into dictionary.

    Args:
        dataset_path: the path where your dataset stored
        split_name: corresponding split to read

    Returns:
        split_img: python dictionary containing a split of images
    """
    a_split_img = {}
    all_class_name = get_class_name(dataset_path, split_name)

    for class_name in all_class_name:
        class_path = dataset_path + '/' + class_name
        a_class_img = read_class_img(class_path, class_name)
        a_split_img[class_name] = a_class_img

    return a_split_img

def read_split_attribute(dataset_path, split_name):
    print("读取 " + str(split_name) + " 划分的属性")
    print("====================")

    all_attribute = []
    split_attribute = []

    attribute_file_path = dataset_path + '/' + r'predicate-matrix-binary.txt'
    attribute_matrix_file = io.open(attribute_file_path, 'r')

    # 读取整个数据集的属性向量
    attribute_count = 0
    while attribute_count != 50:
        attribute = np.array(attribute_matrix_file.readline().split(' '), dtype=int)
        all_attribute.append(attribute)
        attribute_count += 1

    # 读取当前数据划分(split_name)的属性向量
    all_class_name = get_class_name(dataset_path, 'all')
    split_class_name = get_class_name(dataset_path, split_name)

    # 下面功能的实现的前提是：classes.txt中的类别和predicate-matrix-binary.txt中的属性是一一对应的
    for class_name in split_class_name:
        # 找到我们想要的图片类别在所有图片类别中的位置
        attribute_index = all_class_name.index(class_name)
        # 上面类别的位置也就是对应的属性向量在所有属性向量中的位置
        split_attribute.append(all_attribute[attribute_index])

    # split_attribute = np.asarray(split_attribute, dtype=int)

    attribute_matrix_file.close()

    print("====================")
    print(str(split_name) + "属性读取完成\n")

    return split_attribute