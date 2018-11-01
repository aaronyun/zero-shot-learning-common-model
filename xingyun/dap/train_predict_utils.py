# -*- coding: UTF-8 -*-

import io
import os

import numpy as np
import tensorflow as tf
from sklearn import svm, naive_bayes

from preprocess.utils import get_class_name


################################# 训练分类器 ###################################

def train_single_svm_clf(train_data, train_label):
    """
    """
    single_svm_clf = svm.LinearSVC()
    single_svm_clf.fit(train_data, train_label)

    return single_svm_clf

def train_svm_clfs(train_data, train_label):
    """Train 85 svm for each attribute.

    Args:
        train_data: features used to training, of shape (num of examples, 512)
        train_label: attributes used to training, of shape (num of examples, num of attributes)

    Returns:
        smv_clfs: all 85 support vector machine corresponding to 85 attributes
    """
    smv_clfs = []

    for attr_index in range(0, 85):
        features = train_data
        attr = train_label[:, attr_index]

        print("\n====================")
        print("SVM count: " + str(attr_index+1))
        print("Current svm training begining...\n")
        current_svm = train_single_svm_clf(features, attr)
        print("Training completed!")
        print("====================\n")

        smv_clfs.append(current_svm)

    return smv_clfs

def train_bayes(test_data, test_label):
    """
    """
    print("\n====================")
    print("Bayes classifier training begining...\n")
    bayes_clf = naive_bayes.BernoulliNB()
    bayes_clf.fit(test_data, test_label)
    print("Training completed!")
    print("====================\n")

    return bayes_clf

################################ 数据训练与预测 #################################

def train_model(data):
    """
    """
    with tf.Session() as sess:
        print("\n////////// Training process //////////\n\n")
        svm_clfs = train_svm_clfs(data['train_data'], data['train_label'])

        bayes_clf = train_bayes(data['test_data'], data['test_label'])

        model = {'svm_clfs':svm_clfs, 'bayes_clf':bayes_clf}
        print("\n\n////////// Training completed //////////\n")

    return model

def predict(test_data, test_label, model):
    """
    """
    svm_clfs = model['svm_clfs']
    bayes_clf = model['bayes_clf']

    # 先用训练好的svm得到图片对应的属性向量
    attr_type_count = 1
    for attr_index in range(0, 85):
        svm_for_current_attr = svm_clfs[attr_index]
        # 用SVM预测出一批图片是否有当前属性，得到的是 ()
        attr_predict_result = svm_for_current_attr.predict(test_data)
        print(attr_predict_result)
        print(attr_predict_result.shape)

        if attr_type_count == 1:
            attr = attr_predict_result
        else:
            attr = np.hstack((attr, attr_predict_result))

        attr_type_count += 1

    # 再用bayes和得到的属性向量预测类别
    predict_result = bayes_clf.predict(attr)

    return predict_result