# -*- coding: UTF-8 -*-

import os

import numpy as np
from matplotlib import image
import tensorflow as tf
from skimage import io as skio

# 读取一张图片
def single_img_read(upper_path, img_name):
    """
    Read an image and store in ndarray.

    Arguments
    ---
    im_path--image path on your driver

    Returns
    ---
    im_array--ndarray, of shape (m, n, 3) for RGB image
    """

    # 当前图片的路径
    img_path = upper_path + '/' + img_name

    img = image.imread(img_path)

    return im_array

# 获取对应数据划分里图片类别的名字
def get_cls(data_type):
    cls_list = []
    if data_type == 'train':
        data_path = r'/home/xingyun/datasets/AWA/trainclasses.txt'
    elif data_type == 'valid':
        data_path = r'/home/xingyun/datasets/AWA/validclasses.txt'
    elif data_type == 'test':
        data_path = r'/home/xingyun/datasets/AWA/testclasses.txt'
    elif data_type == 'all':
        data_path = r'/home/xingyun/datasets/AWA/classes.txt'
    else:
        print("没有叫做 " + str(data_type) + " 的数据划分！！！\n")
        return

    cls_file = io.open(data_path, 'r')

    cls_name = ' '
    while cls_name != '':
        # 读取类别的名称并去掉换行符
        cls_name = cls_file.readline().rstrip('\n')
        # 确保没有加入无效的名称
        if len(cls_name) != 0:
            cls_list.append(cls_name)

    cls_file.close()

    return cls_list

# 根据给出的图片类别名称读取这个类别的图片
def cls_img_reader(cls_path, cls_name):
    """Read a class of images. e.g. antelope

    Args:
        cls_name: which class of images you want to read.

    Returns:
        cls_img_array: a class of image in numpy array, of shape (num of images, height, width, channels)
    """

    # 图片名字的列表
    img_name_list = os.listdir(cls_path)
    
    img_count = 1 # 图片计数器
    for img_name in img_name_list:
        # 读取当前图片
        img = single_img_read(cls_path, img_name)

        # 需要考虑读的是不是当前类别的第一张图片
        if img_count == 1:
            cls_img = img
        else:
            cls_img = np.vstack((cls_img, img))

        img_count += 1

    return cls_img

# 读取一个数据划分的图片
def split_img_reader(dataset_path, split_name):
    """Read a data split of images into dictionary.

    Args:
        dataset_path: the path where your dataset stored
        split_name: corresponding split to read

    Returns:
        img_split_dic: python dictionary containing a split of images
    """

    img_split_dic = {} # 存放图片的字典

    cls_name_list = get_cls(split_name) # 当前数据划分包含的类别列表

    for cls_name in cls_name_list:
        cls_path = dataset_path + '/' + cls_name
        cls_img = cls_img_reader(cls_path, cls_name)
        img_split_dic[cls_name] = cls_img

    return img_split_dic


# def image_holder(cls_path):
#     """
#     Create python dictionary to store different type of images.

#     Arguments
#     ---
#     cls_path--path of classes

#     Returns
#     ---
#     im_dic--python dictionary containing classes and corresponding images 
#     """

#     # hodlder
#     im_dic = {}

#     # get the name of classes
#     cls_list = os.listdir(cls_path)

#     for single_cls in cls_list:
#         im_dic[single_cls] = []
    
#     return im_dic

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