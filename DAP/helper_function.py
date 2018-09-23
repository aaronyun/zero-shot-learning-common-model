# -*- coding: UTF-8 -*-

import io
import os

from skimage import io as skio
import numpy as np
from sklearn import svm


# 根据给出的图片类别名称读取这个类别的图片
def batch_img_reader(batch_name):

    # 用列表保存这一批图片
    batch_img_array = []

    # path是当前类别的路径，后面还要加上图片名字才能读取
    path = 'F:\\datasets\\AWA\\JPEGImages\\' + str(batch_name)
    # 图片名字的列表
    img_list = os.listdir(path)

    for img_name in img_list:
        # 图片的路径
        img_path = path + str(img_name)
        img_array = skio.imread(img_path)
        batch_img_array.append(img_array)

    return batch_img_array