# -*- coding: UTF-8 -*-

import io

from sklearn import svm, naive_bayes
import numpy as np
from skimage import io as skio

from keras.models import Model
from keras import layers

from helper_function import img_reader, attribute_reader, svm_train
from helper_function import attr_predict, random_test_pair

################################################################################

# 首先是数据处理

# 训练数据
train_img, cls_list = img_reader(dataset_name='AWA', data_type='train')
train_attr = attribute_reader('train')
# 测试数据
test_img, _ = img_reader(dataset_name='AWA', data_type='test')
test_cls = test_img.keys()
test_attr = attribute_reader('test')

################################################################################

# 对每一个图片级别的属性训练一个支持向量机
# 实现从‘图片-属性’层的操作：进来一张新的图片，判断它包含哪些属性

# 对应85个属性的支持向量机
svm_85 = svm_train(train_img, train_attr, cls_list)

################################################################################

# 上面实现了从图片到属性的分类
# 本质上应该还需要一个从类别到属性的预测
# 但AWA数据集中用来测试的类别已经直接给出了属性标签

################################################################################

# 最后用贝叶斯定理来实现对图像类别的预测
# 实际上是将属性向量作为训练数据，而类别作为标签来训练一个贝叶斯分类器
# 最后新的图片进来时，先用支持向量机得到预测的属性向量，然后在用训练好的贝叶斯来分类

# 用测试数据来训练贝叶斯分类器
clf_init = naive_bayes.GaussianNB()
bayes_clf = clf_init.fit(np.array(test_attr), test_cls)

################################################################################

# 最后才是根据支持向量机预测属性向量，再根据贝叶斯预测类别
test_img, attr = random_test_pair(test_img, test_attr)

attribute = attr_predict(test_img, svm_85)
cls_prediction = bayes_clf.predict(attribute)
