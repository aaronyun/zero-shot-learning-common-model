# -*- coding: UTF-8 -*-

import io

from skimage import io as skio
import numpy as np
from sklearn import svm


# 根据给出的图片类别名称读取这个类别的图片
def batch_img_reader(batch_name):

    path = 'F:\\datasets\\AWA\\JPEGImages\\' + str(batch_name)
    
    

    return batch_img_array