# -*- coding: UTF-8 -*-

import numpy as np

from preprocess.utils import reader, get_class_name, extend_array
from dap.train_predict_utils import train_model, predict, evaluate

# EPOCHS = 20
DATASET_PATH = r'/data0/xingyun/AWA'

train_data = reader('train', 'features')
train_label = reader('train', 'attributes')

test_data_for_train = reader('test', 'attributes')
test_data_for_predict = reader('test', 'features')
test_label = np.array(get_class_name(DATASET_PATH, 'test'), dtype=np.str)
test_label_expanded = extend_array(test_label)

data = {'train_data':train_data, 'train_label':train_label, 
        'test_data':test_data_for_train, 'test_label':test_label_expanded}

# train
model = train_model(data)

# predict
attr_result, cls_result= predict(test_data_for_predict, model)

# evaluate
test_attr = test_data_for_train
test_attr_predict = attr_result
test_cls = test_label_expanded
test_cls_predict = cls_result
evaluate(test_attr, test_attr_predict, test_cls, test_cls_predict)