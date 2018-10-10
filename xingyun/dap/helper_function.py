# -*- coding: UTF-8 -*-

import io
import os

from skimage import io as skio
from skimage import transform

import numpy as np
import tensorflow as tf
from sklearn import svm, naive_bayes

from extractor.vggNet import vgg19


#################################数据处理函数####################################

# # 进行特征提取并保存
# def vgg_extract(img_set_dic, keep_prob=1):
#     """Extract the feature of images with vgg19.

#     Args:
#         img_set_dic: images need to be process
#         keep_prob: parameter in vggNet

#     Returns:
#         vgg_img_dic: python dictionary stored precessed images
#     """
#     # 打开文件
#     feature_file = io.open('img_features.txt', 'w')

#     set_list = img_set_dic.keys() # 得到所有的类别以进行循环处理
#     batch_count = 0
#     for set_name in set_list:
#         batch_img = img_set_dic[set_name]

#         # 创建session运行vgg
#         with tf.session() as sess:
#             tf.run(tf.global_variables_initializer())
            
#             # 进行特征提取
#             # 注意：features是一个tensor，形状是(num of examples, 1, 1, 1000)
#             predictions, softmax, fc3, params = vgg19(batch_img, keep_prob)

#             # 将得到的tensor转化为ndarray
#             if batch_count == 0:
#                 batch_features = fc3.eval().squeeze()
#             else:
#                 batch_features.vstack(fc3.eval().squeeze())

#         # 多次调用np.load()函数时会重写文件内容
#         np.save('features.npy', batch_features)

#     return

# 获取对应数据划分里有哪些图片类别
def get_cls(data_type):
    cls_list = []
    if data_type == 'train':
        data_path = r'/home/xingyun/datasets/AWA/trainclasses.txt'
    elif data_type == 'valid':
        data_path = r'/home/xingyun/datasets/AWA/validclasses.txt'
    elif data_type == 'test':
        data_path = r'/home/xingyun/datasets/AWA/testclasses.txt'
    elif data_type == 'all':
        data_path = r'/home/xingyun/datasets/AWA/classes.txt'
    else:
        print("没有叫做 " + str(data_type) + " 的数据划分！！！\n")
        return

    cls_file = io.open(data_path, 'r')

    cls_name = ' '
    while cls_name != '':
        # 读取类别的名称并去掉换行符
        cls_name = cls_file.readline().rstrip('\n')
        # 确保没有加入无效的名称
        if len(cls_name) != 0:
            cls_list.append(cls_name)

    cls_file.close()

    return cls_list

# # 根据给出的图片类别名称读取这个类别的图片
# def batch_img_reader(batch_name):
#     """Read a batch of images.

#     A batch means the one of the image classes in your dataset.

#     Args:
#         batch_name: which image class you want to read.

#     Returns:
#         batch_img_array: a batch of image in numpy array, of shape (num_img, 224*224*3)
#     """

    
#     # path是当前类别的路径，后面还要加上图片名字才能读取
#     path = r'/home/xingyun/datasets/AWA/JPEGImages/' + str(batch_name)
#     # 图片名字的列表
#     img_list = os.listdir(path)
    
#     img_count = 1
#     for img_name in img_list:
#         # 图片的路径
#         img_path = path + '/' + str(img_name)
#         img = skio.imread(img_path)

#         # 裁剪图像
#         img_resized = transform.resize(img, (224, 224))
#         # 把图像转换成(1, 224*224*3)
#         img_flatten = np.reshape(img_resized, (1, -1))

#         if img_count == 1:
#             batch_img_array = img_flatten
#         else:
#             batch_img_array = np.vstack((batch_img_array, img_flatten))

#         img_count += 1
#     # print((batch_img_array.shape))
#     return batch_img_array

# # 读取训练或测试集的图片
# def img_reader(dataset_name, set_type):
#     print("读取 " + str(set_type) + " 数据划分\n")

#     img_dic = {}
#     # 要获取的数据划分包含的类别
#     cls_list = get_cls(set_type)

#     batch_count = 1
#     for batch_name in cls_list:
#         print("当前读取图片类别：" + str(batch_name))
#         batch_img = batch_img_reader(batch_name)
#         img_dic[batch_name] = batch_img
#         print(str(batch_count) + ":" + str(batch_name) + "类图片读取完成\n")
#         batch_count += 1

#     print(str(set_type) + " 数据划分读取完成\n")

#     return img_dic

# 读取类别对应的属性向量
def attribute_reader(data_type):
    print("读取 " + str(data_type) + " 集属性")

    # 全部属性向量
    all_attr_list = []
    # 指定数据划分对应的属性向量
    attr_list = []

    attribute_matrix = io.open(r'/home/xingyun/datasets/AWA/predicate-matrix-binary.txt', 'r')

    # 先把所有的属性向量放到列表里
    index = 0
    while index != 50:
        attribute_array = np.array(attribute_matrix.readline().split(' '), dtype=int)
        all_attr_list.append(attribute_array)
        index += 1

    # 再获取数据划分对应的属性向量
    all_cls_list = get_cls('all') # 取所有类别
    # print(all_cls_list)
    correspond_cls_list = get_cls(data_type) # 取我们想要的类别
    # print(correspond_cls_list)
    for element in correspond_cls_list:
        # 找到我们想要的类别在所有类别中的位置
        attr_index = all_cls_list.index(element)
        # 上面类别的位置也就是对应的属性向量在所有属性向量中的位置
        attr_list.append(all_attr_list[attr_index])

    attribute_matrix.close()

    print(str(data_type) + "属性读取完成\n")

    return attr_list

#################################数据训练函数####################################

def svm_85_train(train_img, train_attr):
    # 由于二分类支持向量机的训练数据必须有两个类别
    # 所以训练数据只能把多个图片类别的数据整个成一次的训练数据

    svm_85 = []
    train_cls_list = train_img.keys()

    # 85个属性对应85个SVM
    for attr_index in range(85):
        clf = svm.SVC()

        # 将对应数据划分里的所有图片组成一个ndarray (num_all_examples, 224*224*5)
        batch_index = 0
        for batch_name in train_cls_list:
            attr = train_attr[batch_index][attr_index]  # 获取的是单一值
            # 先将每个类别当前的属性扩展成(num_examples,)的形状
            if attr == 0:
                attr_temp = np.ravel(np.zeros((train_img[batch_name].shape[0],1), dtype=int))
            else:
                attr_temp = np.ravel(np.ones((train_img[batch_name].shape[0],1), dtype=int))

            # 如果当前是第一次取图片
            # 就初始化img_for_train和attr_for_train
            if batch_index == 0:
                img_for_train = train_img[batch_name]
                attr_for_train = attr_temp
            else:
            # 如果不是第一次取图片
            # 就把之前取到的图片和属性堆叠起来
                img_for_train = np.vstack((img_for_train, train_img[batch_name]))
                attr_for_train = np.concatenate((attr_for_train, attr_temp), axis=None)

            batch_index += 1

        clf.fit(img_for_train, attr_for_train)
        svm_85.append(clf)
    
    return svm_85

def bayes_train(test_img, test_attr):
    # 未知类别的种类
    test_cls = test_img.keys()
    
    # 训练一个伯努利朴素贝叶斯分类器
    clf = naive_bayes.BernoulliNB()
    # 将未知类别和对应的属性向量作为训练数据
    clf.fit(test_attr, test_cls)

    return clf


# 得到一个新的类别来测试
# def random_test_pair(test_img, test_attr):
#     # 取那个类别的图片
#     num_cls = len(test_img.keys())
#     cls_index = np.random.randint(0, num_cls)

#     # 取那张图片
#     num_img = len(len(test_img[cls_index]))
#     img_index = np.random.randint(0, num_img)

#     test_img = test_img[test_img[cls_index][img_index]]
#     attr_vector = test_attr[cls_index]

#     return test_img, attr_vector

# # 得到图片的预测属性向量
# def attr_predict(test_img, svm_list):
#     attr = []
#     for svm_id in len(svm_list):
#         attr_temp = svm_list[svm_id].predict(test_img)
#         attr.append(attr_temp)

#     return attr