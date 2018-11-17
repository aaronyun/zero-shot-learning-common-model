# -*- coding: UTF-8 -*-

import os
import io

import numpy as np
from matplotlib import image
import tensorflow as tf
from skimage import io as skio

from dap.utils import resize_img, writer
from dap.utils import class_name_of_split, img_count

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

def read_cls_img(all_img_path, class_name):
    """Read a class of images. e.g. antelope

    Args:
        class_name: which class of images you want to read.

    Returns:
        cls_img_array: a class of image in numpy array, of shape (num of images, height, width, channels)
    """
    print("--------------------")
    print("%s class reading begin" % class_name)

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

    print("%s class reading complete" % class_name)
    print("--------------------")

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
    print("\n//////////Reading %s split image//////////" % split_name)

    a_split_img = {}
    all_class_name = class_name_of_split(dataset_path, split_name)

    dataset_path_in_detail = dataset_path + r'/JPEGImages'
    for class_name in all_class_name:
        a_class_img = read_cls_img(dataset_path_in_detail, class_name)
        a_split_img[class_name] = a_class_img

    print("//////////Reading completed//////////\n")

    return a_split_img