import os

from preprocess.feature_process import feature_extractor
from preprocess.utils import reader
from preprocess.img_and_attr_reader import read_and_expand_split_attr

# import tensorflow as tf

dataset_path = r'/data0/xingyun/AWA'

# 特征提取函数测试
# feature_extractor(r'/data0/xingyun/AWA', 'train')
# feature_extractor(r'/data0/xingyun/AWA', 'valid')
# features = reader('valid')
# print(str(features.shape))

# 属性读取和扩充函数测试
# read_and_expand_split_attr(dataset_path, 'valid')
# test_attr = reader('test', 'attributes')
# print(test_attr.shape)

# initializer = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(initializer)
    
    # 训练用数据
# features = reader('train', 'features')
# attr = reader('train', 'attributes')
# count = 0
# for i in attr[:, 66]:
#     count = count + i
# print(count)

# 支持向量机训练测试
# svm_85 = train_85_svm(features, attr)


# path = os.getcwd()
# print(path)

# 更新后的特征提取测试
all_split_name = ['train', 'test', 'valid']
for split_name in all_split_name:
    feature_extractor(dataset_path, split_name)