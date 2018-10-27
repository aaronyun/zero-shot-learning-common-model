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

    for attr_index in range(85):
        features = train_features
        attr = train_attr[:][attr_index]

        print("当前训练第 " + str(attr_index) + " 个支持向量机")
        train_single_svm(features, attr)
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