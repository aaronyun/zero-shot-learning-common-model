# -*- coding: UTF-8 -*-

import io
import os

import numpy as np
import tensorflow as tf
from sklearn import svm, naive_bayes


#################################数据训练函数####################################

def svm_85_train(train_img, train_attr):
    # 由于二分类支持向量机的训练数据必须有两个类别
    # 所以训练数据只能把多个图片类别的数据整个成一次的训练数据

    svm_85 = []
    train_cls_list = train_img.keys()

    # 85个属性对应85个SVM
    for attr_index in range(85):
        clf = svm.SVC()

        # 将对应数据划分里的所有图片组成一个ndarray (num_all_examples, 224*224*5)
        batch_index = 0
        for batch_name in train_cls_list:
            attr = train_attr[batch_index][attr_index]  # 获取的是单一值
            # 先将每个类别当前的属性扩展成(num_examples,)的形状
            if attr == 0:
                attr_temp = np.ravel(np.zeros((train_img[batch_name].shape[0],1), dtype=int))
            else:
                attr_temp = np.ravel(np.ones((train_img[batch_name].shape[0],1), dtype=int))

            # 如果当前是第一次取图片
            # 就初始化img_for_train和attr_for_train
            if batch_index == 0:
                img_for_train = train_img[batch_name]
                attr_for_train = attr_temp
            else:
            # 如果不是第一次取图片
            # 就把之前取到的图片和属性堆叠起来
                img_for_train = np.vstack((img_for_train, train_img[batch_name]))
                attr_for_train = np.concatenate((attr_for_train, attr_temp), axis=None)

            batch_index += 1

        clf.fit(img_for_train, attr_for_train)
        svm_85.append(clf)
    
    return svm_85

def bayes_train(test_img, test_attr):
    # 未知类别的种类
    test_cls = test_img.keys()
    
    # 训练一个伯努利朴素贝叶斯分类器
    clf = naive_bayes.BernoulliNB()
    # 将未知类别和对应的属性向量作为训练数据
    clf.fit(test_attr, test_cls)

    return clf