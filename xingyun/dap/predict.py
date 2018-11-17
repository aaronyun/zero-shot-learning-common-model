# -*- coding: UTF-8 -*-

import sys
sys.path.append('..')

import numpy as np
from dap.utils import vector_sim_compute, class_name_of_split

def attr_predict(test_data, svm_clfs):
    """
    """
    attrs_exist_prob = []
    attrs_exist_binary = []

    for attr_index in range(0, 85):
        print("====================")
        print("Prediction for %d attribute" % (attr_index + 1))
        svm_for_current_attr = svm_clfs[attr_index]
        
        # 概率预测输出
        attr_predict_proba = svm_for_current_attr.predict_proba(test_data)
        '''取第二列作为属性存在的概率'''
        no_proba, yes_proba = np.hsplit(attr_predict_proba, 2)
        attr_predict_proba = np.ravel(yes_proba)
        # print("当前属性的概率预测取值的形状：%s" % str(attr_predict_proba.shape))
        # 二值预测输出
        attr_predict_binary = svm_for_current_attr.predict(test_data)
        # print("当前属性的binary预测取值的形状：%s" % str(attr_predict_binary.shape))
        print("Current attribute prediction complete")
        print("====================")        

        attrs_exist_prob.append(attr_predict_proba)
        attrs_exist_binary.append(attr_predict_binary)
    
    attrs_exist_prob = np.array(attrs_exist_prob)
    attrs_exist_binary = np.array(attrs_exist_binary)

    attrs_exist_prob = attrs_exist_prob.T
    attrs_exist_binary = attrs_exist_binary.T

    # print("最后得到的测试集的概率和二值属性预测的形状：%s %s" % (str(attrs_exist_prob.shape), str(attrs_exist_binary.shape)))

    return attrs_exist_prob, attrs_exist_binary

def cls_predict(attr_prob_predict, all_test_attr, test_cls_name):
    """
    """
    print("====================")
    print("Class prediction begin")

    cls_predict = []
    for attr_prob in attr_prob_predict:
        attr_sim = []
        for test_attr in all_test_attr:
            sim = vector_sim_compute(attr_prob, test_attr)
            attr_sim.append(sim)

        current_cls = test_cls_name[attr_sim.index(max(attr_sim))]
        cls_predict.append(current_cls)

    print("Class prediction complete")
    print("====================")

    return cls_predict
