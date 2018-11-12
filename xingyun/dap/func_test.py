import os

from preprocess.feature_process import extract
from preprocess.utils import reader, get_class_name
from preprocess.img_and_attr_reader import read_and_expand_split_attr
from dap.train_predict_utils import train_svm_clfs

# import tensorflow as tf

dataset_path = r'/data0/xingyun/AWA'
all_split_name = ['train', 'valid', 'test', 'trainvalid']

# 特征提取函数测试
# feature_extractor(r'/data0/xingyun/AWA', 'train')
# feature_extractor(r'/data0/xingyun/AWA', 'valid')
# features = reader('valid')
# print(str(features.shape))

# 属性读取和扩充函数测试
# for split_name in all_split_name:
#     read_and_expand_split_attr(dataset_path, split_name)

# train_attr = reader('train', 'attributes')
# test_attr = reader('test', 'attributes')
# valid_attr = reader('valid', 'attributes')
# trainvalid_attr = reader('trainvalid', 'attributes')
# print(str(train_attr.shape))
# print(str(test_attr.shape))
# print(str(valid_attr.shape))
# print(str(trainvalid_attr.shape))

# 支持向量机训练测试
svm_train_data = reader(split_name='train', data_type='features')
svm_train_label = reader(split_name='train', data_type='attributes')

svm_clfs = train_svm_clfs(svm_train_data, svm_train_label)
test_fea = reader('test', 'features')[10, :]
test_attr_predict = svm_clfs[8].predict(test_fea)
print(test_attr_predict)

# 更新后的特征提取测试]
# for split_name in all_split_name:
#     extract(dataset_path, split_name)

# 特征提取时类别顺序检测
# test_class_name = get_class_name(dataset_path, 'test')
# print(test_class_name)