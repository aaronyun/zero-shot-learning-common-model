# -*- coding: UTF-8 -*-

import io
import os

import numpy as np
import tensorflow as tf
from sklearn import svm, naive_bayes
import sklearn.metrics as metrics

from preprocess.utils import get_class_name, reader


#################################### train #####################################

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

########################### predicting and evaluating ##########################

def attr_predict(test_features, svm_clfs):
    """
    """
    attr = []
    # 先用训练好的svm得到图片对应的属性向量
    for attr_index in range(0, 85):
        svm_for_current_attr = svm_clfs[attr_index]
        # 用SVM预测出一批图片是否有当前属性
        attr_predict_result = svm_for_current_attr.predict(test_features)

        attr.append(attr_predict_result)
    
    attr = np.array(attr, dtype=int)
    attr = attr.T

    return attr

def cls_predict(test_attr, bayes_clf):
    """
    """
    # 再用bayes和得到的属性向量预测类别
    predict_result = bayes_clf.predict(test_attr)

    return predict_result

def predict(test_data, model):
    """
    """
    svm_clfs = model['svm_clfs']
    bayes_clf = model['bayes_clf']

    attr_result = attr_predict(test_data, svm_clfs)
    cls_result = cls_predict(attr_result, bayes_clf)

    return attr_result, cls_result

def evaluate(test_attr, test_attr_predict, test_cls, test_cls_predic):
    """
    """
    # 计算每个支持向量机的AUC，以及平均值
    overall_auc_score = 0
    num_of_svm = test_attr.shape[1]
    print("\n--------------------")
    for attr_index in range(num_of_svm):
        x = test_attr[:][attr_index]
        x_predict = test_attr_predict[:][attr_index]
        auc = metrics.roc_auc_score(x, x_predict)
        overall_auc_score += auc
        print("AUC score for " + str(attr_index+1) + "th SVM: " + str(auc))
    print("--------------------\n")
    print("\n====================")
    print("Average AUC score on test data: " + str(overall_auc_score/num_of_svm))
    print("====================\n")

    accuracy = metrics.accuracy_score(test_cls, test_cls_predic)
    print("Accuracy: " + str(accuracy * 100) + "%\n\n")
    
    return
