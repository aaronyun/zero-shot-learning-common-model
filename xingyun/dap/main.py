# -*- coding: UTF-8 -*-

import numpy as np

from preprocess.utils import reader, get_class_name, extend_array
from dap.train_predict_utils import train_model, predict, evaluate
from preprocess.feature_process import extract

# EPOCHS = 20
DATASET_PATH = r'/data0/xingyun/AWA'

# 特征提取
# all_split_name = ['train', 'test', 'valid']
# for split_name in all_split_name:
#         extract(DATASET_PATH, split_name)

svm_train_data = reader(split_name='train', data_type='features')
svm_train_label = reader(split_name='train', data_type='attributes')

bayes_train_data = reader(split_name='test', data_type='attributes')
bayes_train_label = get_class_name(DATASET_PATH, 'test')

test_data = reader(split_name='test', data_type='features')
test_label = bayes_train_label

# data = {'train_data':train_data, 'train_label':train_label, 
#         'test_data':test_data_for_train, 'test_label':test_label_expanded}

# # train
# model = train_model(data)

# # predict
# attr_result, cls_result= predict(test_data_for_predict, model)

# # evaluate
# test_attr_true = test_data_for_train
# test_attr_predict = attr_result
# test_cls_true = test_label_expanded
# test_cls_predict = cls_result
# evaluate(test_attr_true, test_attr_predict, test_cls_true, test_cls_predict)

print(str(train_data.shape))
print(str(test_data_for_predict.shape))