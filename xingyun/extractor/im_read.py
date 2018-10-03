# -*- coding: UTF-8 -*-

import numpy as np
from matplotlib import image
import os
import tensorflow as tf

from vggNet import vgg19

def single_im_read(im_path):
    """
    Read an image and store in ndarray.

    Arguments
    ---
    im_path--image path on your driver

    Returns
    ---
    im_array--ndarray, of shape (m, n, 3) for RGB image
    """

    im_array = image.imread(im_path)

    return im_array

def image_holder(cls_path):
    """
    Create python dictionary to store different type of images.

    Arguments
    ---
    cls_path--path of classes

    Returns
    ---
    im_dic--python dictionary containing classes and corresponding images 
    """

    # hodlder
    im_dic = {}

    # get the name of classes
    cls_list = os.listdir(cls_path)

    for single_cls in cls_list:
        im_dic[single_cls] = []
    
    return im_dic

# def im_set_read(image_holder, image_path, set_name):
#     """
#     Read images of the same class and put it in the holder.

#     Arguments
#     image_holder--python dictionary to store images in the form of ndarray
#     image_path--path of current image set

#     Returns
#     ---
#     image_set--a list containing current set of images, ndarray
#     """

#     # 取得当前类别图片的列表
#     im_name_list = os.listdir(image_path)
    
#     # 对上面类别里的图片进行提取
#     for im_name in im_name_list:
#         # 当前图片的路径
#         current_im_path = image_path + '/' + str(im_name)
#         # 读取当前图片
#         current_im = single_im_read(current_im_path)
#         # 将每一张图片喂给vgg19
#         im_tensor = tf.convert_to_tensor(current_im, tf.float32)
        
#         # 将当前图片的ndarray形式加入到容器
#         image_holder[set_name].append(current_im)

#     # 将数值转为float32
#     image_set_float = np.asarray(image_holder[set_name], np.float32)
#     # 再把得到的的图片转为tensor后喂给vgg19
#     image_set_tensor = tf.convert_to_tensor(image_set_float)
#     predictions, softmax, fc3, params = vgg19(image_set_tensor, keep_prob=1)

#     # 返回提取到的这个类别的图片数据以将其写入到外部文件中
#     return fc3


# def im_read(path):
#     """
#     Read images and stored in a dictionary.

#     Arguments
#     ---
#     path--directory stored images

#     Returns
#     ---
#     im_dic--dictionary containing all the images, with class label
#     """

#     # create holder
#     im_dic = holder_create(path)

#     cls_list = os.listdir(path)

#     # count
#     count = 1
#     # iterate classes
#     for single_cls in cls_list:
#         # path of different type of image
#         cls_path = path + '/' + str(single_cls)

#         # get the list of current type of images
#         im_list = os.listdir(cls_path)
#         # put all the images of current type into corresponding list
#         for im_name in im_list:
#             # get the path of a image
#             im_path = cls_path + '/' + str(im_name)
#             # convert current image to ndarray
#             im_array = single_im_read(im_path)
#             # put it in the dictionary with right class label
#             im_dic[single_cls].append(im_array)

#         count += 1

#         print("Oh! All the " + str(single_cls) + " has been stored!\nThis this #%dth# class." % (count))
#         print("====================")

#     return im_dic

# # function test
# images = im_read("E:\\test_datasets\\AWA\\JPEGImages")
# test_im = images['antelope'][0]
# print(str(test_im.shape))
# plt.show(test_im)