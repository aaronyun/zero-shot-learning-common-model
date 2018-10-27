from preprocess.feature_process import feature_extractor, feature_reader
from preprocess.img_and_attr_reader import read_and_expand_split_attr
from dap.classifier import train_85_svm

dataset_path = r'/data0/xingyun/AWA'

# 特征提取函数测试
# feature_extractor(r'/data0/xingyun/AWA', 'train')
# features = feature_reader('valid')
# print(str(features.shape))

# 属性读取和扩充函数测试
# valid_attrs = read_and_expand_split_attr(dataset_path, 'train')
# print(valid_attrs.shape)

# 支持向量机训练测试
