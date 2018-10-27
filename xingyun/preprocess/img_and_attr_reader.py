# -*- coding: UTF-8 -*-

import os
import io

import numpy as np
from matplotlib import image
import tensorflow as tf
from skimage import io as skio

from preprocess.utils import resize_img

#################################图片读取#######################################

def read_single_img(class_path, img_name):
    """Read an image and store in ndarray.

    Args:
        class_path: the path of your images ordered in class
        img_name: the name of which image you want read

    Returns:
        resize_img: resized image of shape (224, 224, 3) for RGB image
    """

    single_img_path = class_path + '/' + img_name
    img = image.imread(single_img_path)
    # 在读取每一张图片的同时进行裁剪
    resized_img = resize_img(img)

    return resized_img

def read_class_img(all_img_path, class_name):
    """Read a class of images. e.g. antelope

    Args:
        class_name: which class of images you want to read.

    Returns:
        cls_img_array: a class of image in numpy array, of shape (num of images, height, width, channels)
    """
    print("--------------------")
    print("开始读取" + str(class_name) + "类的图片")
    print("--------------------")

    class_path = all_img_path + '/' + class_name
    all_img_name = os.listdir(class_path)
    
    class_img = []
    img_count = 1
    for img_name in all_img_name:
        # 读取当前图片
        img = read_single_img(class_path, img_name)
        # print("第" + str(img_count) + "张图片读取完成")

        class_img.append(img)
        img_count += 1

    a_class_img = np.asarray(class_img, dtype=np.float32)
    # print("形状：" + str(a_class_img.shape))

    print("--------------------")
    print(str(class_name) + "类的图片读取完成")
    print("--------------------\n")

    return a_class_img

# 读取一个数据划分的图片
def read_split_img(dataset_path, split_name):
    """Read a data split of images into dictionary.

    Args:
        dataset_path: the path where your dataset stored
        split_name: corresponding split to read

    Returns:
        split_img: python dictionary containing a split of images
    """
    print("开始读取" + str(split_name) + "数据划分的图片\n")

    a_split_img = {}
    all_class_name = get_class_name(dataset_path, split_name)

    dataset_path_in_detail = dataset_path + r'/JPEGImages'
    for class_name in all_class_name:
        a_class_img = read_class_img(dataset_path_in_detail, class_name)
        a_split_img[class_name] = a_class_img

    print(str(split_name) + "数据划分的图片读取完成")

    return a_split_img

############################属性向量读取和处理###################################

def img_count(all_img_path, class_name):
    """Get the number of images in a class.

    Args:
        all_img_path: the path where all the images stored
        class_name: which class you want to count

    Returns:
        num_of_imgs: how many images in a class
    """
    class_path = all_img_path + '/' + class_name
    all_img_name = os.listdir(class_path)

    num_of_imgs = len(all_img_name)

    return num_of_imgs

def expand_attr(attr, class_name, all_img_path):
    """
    """
    # 得到类别名称和对应的属性后再进行扩充
    class_img_count = img_count(all_img_path, class_name)
    # 得到图片的数量然后进行扩充
    expanded_attr = np.tile(attr, (class_img_count, 1))

    return expanded_attr

def read_all_attrs(dataset_path):
    """Get all the attribute vectors.

    Args:
        dataset_path: the path where the data stored

    Returns:
        all_attrs: all the attributes, one class one vector
    """
    attr_file_path = dataset_path + '/' + r'predicate-matrix-binary.txt'
    attr_file = io.open(attr_file_path, 'r')

    all_attrs = []

    attr_count = 0
    while attr_count != 50:
        attr = np.array(attr_file.readline().split(' '), dtype=int)
        all_attrs.append(attr)
        attr_count += 1

    attr_file.close()

    return all_attrs

def read_and_expand_split_attr(dataset_path, split_name):
    """Get attributes of a split of classes.

    Args:
        dataset_path: 
        split_name:

    Returns:
        split_attrs: 
    """
    all_img_path = dataset_path + '/' + 'JPEGImages'

    all_attrs = read_all_attrs(dataset_path)

    all_class_name = get_class_name(dataset_path, 'all')
    split_class_name = get_class_name(dataset_path, split_name)

    # 下面功能的实现的前提是：classes.txt中的类别和predicate-matrix-binary.txt中的属性向量是一一对应的
    attr_count = 1
    for class_name in split_class_name:
        attr_index = all_class_name.index(class_name)
        # print(class_name + ': ' + str(attr_index + 1))
        correspond_attr = all_attrs[attr_index]
        # 对属性向量进行扩充
        expanded_attr = expand_attr(correspond_attr, class_name, all_img_path)

        if attr_count == 1:
            split_attrs = expanded_attr
        else:
            split_attrs = np.vstack((split_attrs, expanded_attr))

        attr_count += 1

    return split_attrs