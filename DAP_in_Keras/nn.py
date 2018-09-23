# -*- coding: UTF-8 -*-

import io

from sklearn import svm
import numpy as np
from skimage import io as skio

from keras.models import Model
from keras import layers

# 存储训练用的图片数据
train_img_dic = {}
train_attribute_array = []

# 用来保存85个二分类支持向量机
svm_85 = []

################################################################################

# 首先是数据处理
# 先取得一个测试用的小数据集
# 图片：(num_examples, num_pixles)
# 标签：(num_examples, binary_feature)

# 训练用图片

train_img_array = skio.imread(r'F:\datasets\AWA\JPEGImages\antelope\antelope_10001.jpg')
# 训练图片对应的属性向量
attribute_matrix = io.open(r'F:\datasets\AWA\predicate-matrix-binary.txt', 'r')
attirbute_array = np.array(attribute_matrix.readline().split(' '), dtype=int)

# 训练数据
train_X = np.reshape(train_img_array, (1, -1)) # 1D vector
train_Y = np.reshape(attirbute_array, (85,))

################################################################################

# 对每一个图片级别的属性训练一个支持向量机
# 实现从‘图片-属性’层的操作：进来一张新的图片，判断它包含哪些属性


################################################################################

# 上面得到了从图片到属性的判断
# 本质上应该还需要一个从类别到属性的预测
# 但AWA数据集中用来测试的类别已经直接给出了属性标签

################################################################################

# 最后用贝叶斯定理来实现对图像类别的预测