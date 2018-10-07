# -*- coding: UTF-8 -*-

from helper_function import img_reader, attribute_reader, svm_85_train
from helper_function import bayes_train, get_cls
from model import attr_predict, cls_predict

################################################################################

print("====================")
print("开始读取数据")
print("====================\n")

# 训练数据
train_img = img_reader(dataset_name='AWA', set_type='train')
train_attr = attribute_reader('train')

# 验证数据
valid_img = img_reader(dataset_name='AWA', set_type='valid')
valid_attr = attribute_reader('valid')

# 测试数据
test_img = img_reader(dataset_name='AWA', set_type='test')
test_cls = test_img.keys()
test_attr = attribute_reader('test')

print("====================")
print("数据读取完成")
print("====================\n")

################################################################################

# 对每一个图片级别的属性训练一个支持向量机
# 实现从‘图片-属性’层的操作：进来一张新的图片，判断它包含哪些属性

print("____开始训练支持向量机____")

# 对应85个属性的支持向量机
svm_85 = svm_85_train(train_img, train_attr)

print("____支持向量机训练完成____")

################################################################################

# 上面实现了从图片到属性的分类
# 本质上应该还需要一个从类别到属性的预测
# 但AWA数据集中用来测试的类别已经直接给出了属性标签

################################################################################

# 最后用贝叶斯定理来实现对图像类别的预测
# 实际上是将属性向量作为训练数据，而类别作为标签来训练一个贝叶斯分类器

print("____开始训练贝叶斯分类器____")

bayes_clf = bayes_train(test_img, test_attr)

print("____贝叶斯分类器训练完成____")

# 即先用支持向量机得到预测的属性向量，然后再用训练好的贝叶斯分类器得到类

################################################################################

# 最后用支持向量机预测属性向量，再根据贝叶斯预测类别
# 用一张未知类别的图片进行测试
# print("____用一张图片进行测试____")
# predict_img = skio.imread(r'/home/xingyun/datasets/AWA/JPEGImages/hippopotamus/hippopotamus_10001.jpg')

# print("测试图片类别：hippopotamus")
# attribute = attr_predict(predict_img, svm_85)
# cls_name = cls_predict(attribute, bayes_clf)
# print("模型预测类别："+ str(cls_name))
