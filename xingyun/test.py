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
# train_attr = reader('valid', 'attributes')
# print(train_attr.shape)

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


path = os.getcwd()
print(path)



# # 属性预测
# def attr_predict(img, svm_list):
#     # 保存得到的属性
#     attr = []
#     # 用前面得到的85个SVM预测新图片的属性值
#     for svm_id in len(svm_list):
#         attr_temp = svm_list[svm_id].predict(img)
#         attr.append(attr_temp)

#     return np.array(attr)

# # 类别预测
# def cls_predict(attr, bayes_clf):

#     cls_name = bayes_clf.predict(attr)

#     return cls_name

# # 最终得到的模型
# def model():
#     # 将整个训练过程集成到这个函数
#     # 实现调用这个函数并传入相应的数据后直接得到dap模型
#     # 返回的模型可以直接传入参数进行预测
#     return