import sys
sys.path.append("..")
import os

from sklearn import svm, metrics
import numpy as np
import tensorflow as tf

from dap.feature_process import extract 
from dap.utils import reader, class_name_of_split
from dap.train import train_svm_clfs
from dap.predict import attr_predict, cls_predict

dataset_path = r'/data0/xingyun/AWA'
all_split_name = ['train', 'valid', 'test', 'trainvalid']


'''特征提取函数测试'''
# feature_extractor(r'/data0/xingyun/AWA', 'train')
# feature_extractor(r'/data0/xingyun/AWA', 'valid')
# features = reader('valid')
# print(str(features.shape))

'''数据读取测试'''
# train_data = reader('train', 'features')
# train_label = reader('train', 'attributes')
# test_data = reader('test', 'features')
# test_label = reader('test', 'attributes')
# valid_data = reader('valid', 'features')
# trainvalid_data = reader('trainvalid', 'features')
# print(str(train_data.shape))
# print(str(train_label.shape))
# print(str(test_data.shape))
# print(str(valid_data.shape))
# print(str(trainvalid_data.shape))

'''属性读取和扩充函数测试'''
# for split_name in all_split_name:
#     read_and_expand_split_attr(dataset_path, split_name)


'''单个支持向量机输出测试'''
# clf = svm.SVC(kernel='linear', probability=True)
# train_label = train_label[:,0]
# clf.fit(train_data, train_label)
# test_data = test_data[0:5,:]
# prediction = clf.predict_proba(test_data)
# print(str(prediction.shape))
# print(prediction)

'''支持向量机训练测试'''
# svm_train_data = reader(split_name='train', data_type='features')
# svm_train_label = reader(split_name='train', data_type='attributes')

# svm_clfs = train_svm_clfs(svm_train_data, svm_train_label)
# test_fea = reader('test', 'features')[10, :]
# test_attr_predict = svm_clfs[8].predict(test_fea)
# print(test_attr_predict)

'''更新后的特征提取测试'''
# for split_name in all_split_name:
#     extract(dataset_path, split_name)

'''特征提取时类别顺序检测'''
# test_class_name = get_class_name(dataset_path, 'test')
# print(test_class_name)

'''检查支持向量机的预测是否有问题'''
# with tf.Session() as sess:
#     svm_clfs = train_svm_clfs(train_data, train_label)
#     attr_predict, _ = attr_predict(test_data, svm_clfs)
#     attr_true = test_label

#     for i in [1,100,200,300,400,500]:
#         acc = metrics.accuracy_score(attr_true[i,:], attr_predict[i,:])
#         print(acc)

'''读取划分类别测试'''
# all_cls_name = class_name_of_split(dataset_path, 'all')
# print(all_cls_name)

'''类别预测函数测试'''
# attr_prob_predict = np.array(((0.7,0.5,0.3,0.8),(0.1,0.4,0.6,0.8),(0.2,0.3,0.5,0.1),(0.6,0.1,0.7,0.9)))
# all_test_attr = np.array(((1,0,0,0),(0,1,0,0)))
# test_cls_name = ['第一类', '第二类']
# cls_result = cls_predict(attr_prob_predict, all_test_attr, test_cls_name)
# print(cls_result)