# -*- coding: UTF-8 -*-
import sys
sys.path.append('..')
from time import time

import numpy as np
import tensorflow as tf

from dap.attr_process import read_split_attr
from dap.feature_process import extract

from dap.train import train_svm_clfs
from dap.predict import attr_predict, cls_predict
from dap.evaluate import evaluate

from dap.utils import reader, class_name_of_split, extend_array

# EPOCHS = 20
DATASET_PATH = r'/data0/xingyun/AWA'

with tf.Session() as sess:
    '''特征提取'''
    # all_split_name = ['train', 'test', 'valid']
    # for split_name in all_split_name:
    #         extract(DATASET_PATH, split_name)

    '''获取数据'''
    train_features = reader(split_name='train', data_type='features')
    expanded_train_attr = reader(split_name='train', data_type='attributes')

    test_features = reader(split_name='test', data_type='features')
    expanded_test_attr = reader(split_name='test', data_type='attributes')
    test_attr = read_split_attr(DATASET_PATH, 'test')

    test_cls_name = class_name_of_split(DATASET_PATH, 'test')
    extended_test_cls_name = extend_array(test_cls_name)

    '''模型训练'''
    train_start = time()
    svm_clfs = train_svm_clfs(train_features, expanded_train_attr)
    train_consume = (time() - train_start)
    print("训练耗时：%f" % train_consume)

    '''对测试集预测'''
    predict_start = time()
    attr_prob, attr_binary = attr_predict(test_features, svm_clfs)
    cls_result = cls_predict(attr_prob, test_attr, test_cls_name)
    predict_consume = (time() - predict_start)
    print("预测耗时：%f" % predict_consume)

    '''评估：计算AUC和准确率'''
    # cls_result = np.array(cls_result)
    # print(str(cls_result.shape))
    # cls_result = np.reshape(cls_result, (extended_test_cls_name.shape[0],))
    # assert(cls_result.shape == extended_test_cls_name.shape)
    evaluate(expanded_test_attr, attr_binary, extended_test_cls_name, cls_result)