# -*- coding: UTF-8 -*-
import numpy as np

from preprocess.utils import reader, get_class_name, extend_array
from dap.train_predict_utils import train_model, predict

# EPOCHS = 20
DATASET_PATH = r'/data0/xingyun/AWA'

train_data = reader('train', 'features')
train_label = reader('train', 'attributes')

test_data = reader('test', 'features')
test_label = np.array(get_class_name(DATASET_PATH, 'test'), dtype=np.str)
test_label_expanded = extend_array(test_label)
# print("Final shape: " + str(test_label_expanded.shape))

data = {'train_data':train_data, 'train_label':train_label, 
        'test_data':test_data, 'test_label':test_label_expanded}

# train
# print("\n********************")
# print("Current epoch: " + str(epoch))
# print("********************\n")
model = train_model(data)

# predict
result = predict(test_data, test_label, model)
print(result)