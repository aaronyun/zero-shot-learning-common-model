from preprocess.img_and_attr_reader import read_split_attribute
from preprocess.feature_process import feature_extractor, feature_reader

# 属性提取函数测试
# split_attribute = read_split_attribute(r'F:\datasets\AWA', 'valid')
# print(split_attribute)

# 特征提取函数测试
feature_extractor(r'/data0/xingyun/AWA', 'train')
# features = feature_reader('valid')
# print(str(features.shape))