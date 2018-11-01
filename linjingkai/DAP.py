# -*- coding: UTF-8 -*-

import io
import os

import numpy as np
import tensorflow as tf
from sklearn import svm, naive_bayes


#################################数据训练函数####################################

def train_single_svm(train_features, train_attr):
    classifier = svm.SVC()
    classifier.fit(train_features, train_attr)

    return svm


def train_85_svm(train_features, train_attr):
    """Train 85 svm for each attribute.

    Args:
        train_features: features used to training, of shape (num of examples, 512)
        train_attr: attributes used to training, of shape (num of examples, num of attributes)

    Returns:
        svm_85: all 85 support vector machine corresponding to 85 attributes
    """
    svm_85 = []
    attrbute = train_attr.T
    attrbute = np.array(attrbute).reshape([-1, 1])
    print("the attrbute shape is", attrbute)

    for attr_index in range(85):
        features = train_features
        attr = attrbute[attr_index]

        print("当前训练第 " + str(attr_index) + " 个支持向量机")
       # train_single_svm(features, attr)
        print("当前支持向量机训练完成")

    return svm_85


def bayes_train(test_img, test_attr):
    # 未知类别的种类
    test_cls = test_img.keys()

    # 训练一个伯努利朴素贝叶斯分类器
    clf = naive_bayes.BernoulliNB()
    # 将未知类别和对应的属性向量作为训练数据
    clf.fit(test_attr, test_cls)

    return clf

feature_train_image = np.load("/data0/linjingkai/DataSet/AWA/train_image.npy")
print("当前图片特征大小： ", feature_train_image.shape)
feature_train_label = np.load("/data0/linjingkai/DataSet/AWA/train_label.npy")
print("当前标签大小： ",feature_train_label.shape)