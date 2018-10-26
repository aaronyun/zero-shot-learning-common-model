from preprocess.feature_process import feature_extractor, feature_reader
from preprocess.img_and_attr_reader import read_and_expand_split_attr

dataset_path = r'/data0/xingyun/AWA'

# 特征提取函数测试
<<<<<<< HEAD
# feature_extractor(dataset_path, 'valid')
=======
feature_extractor(r'/data0/xingyun/AWA', 'valid')
>>>>>>> 05c6961ef22f3e40913ca31d4d756a12608bbd38
# features = feature_reader('valid')
# print(str(features.shape))

# 属性读取和扩充函数测试
valid_attrs = read_and_expand_split_attr(dataset_path, 'train')
print(valid_attrs.shape)