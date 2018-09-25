# -*- coding: UTF-8 -*-

import io
import os

from skimage import io as skio
from skimage import transform
import numpy as np
from sklearn import svm

# 获取训练或者是测试数据有哪些类别
def get_cls(data_type):
    cls_list = []
    if data_type == 'train':
        data_path = 'F:\\datasets\\AWA\\trainclasses.txt'
    elif data_type == 'test':
        data_path = 'F:\\datasets\\AWA\\testclasses.txt'
    else:
        data_path = 'F:\\datasets\\AWA\\classes.txt'

    cls_file = io.open(data_path, 'r')

    cls_name = 'initial'
    while cls_name != '':
        # 读取类别的名称
        # 并去掉换行符
        cls_name = cls_file.readline().rstip('\n')
        cls_list.append(cls_name)

    cls_file.close()

    return cls_list

# 根据给出的图片类别名称读取这个类别的图片
def batch_img_reader(batch_name):

    # 用列表保存这一批图片
    batch_img_array = []

    # path是当前类别的路径，后面还要加上图片名字才能读取
    path = 'F:\\datasets\\AWA\\JPEGImages\\' + str(batch_name)
    # 图片名字的列表
    img_list = os.listdir(path)

    for img_name in img_list:
        # 图片的路径
        img_path = path + str(img_name)
        img = skio.imread(img_path)

        # 裁剪图像
        img_resized = transform.resize(img, (224, 224))

        # 把图像转换成(1, 224*224*3)
        img_flatten = np.reshape(img_resized, (1, -1))

        batch_img_array.append(img_flatten)

    return batch_img_array

# 读取训练或测试集的图片
def img_reader(dataset_name, data_type):

    path = 'F:\\datasets' + dataset_name

    # 得到图片类别的列表
    all_cls_list = os.listdir(path + 'JPEGImages')

    img_dic = {}
    cls_list = get_cls(data_type)

    for batch_name in all_cls_list:
        if batch_name in cls_list:
            batch_img = batch_img_reader(batch_name)
            img_dic[batch_name] = batch_img
        else:
            continue

    return img_dic, cls_list

# 读取类别对应的属性向量
def attribute_reader(data_type):
    # 用来存放属性向量的列表，包含全部值
    all_attribute = []
    # 用来存放对应类别的全部值
    attribute_list = []

    attribute_matrix = io.open(r'F:\datasets\AWA\predicate-matrix-binary.txt', 'r')

    # 先把所有的属性向量放到列表里
    index = 0
    while index != 50:
        attribute_array = np.array(attribute_matrix.readline().split(' '), dtype=int)
        all_attribute.append(attribute_array)
        index += 1

    # 取出对应类别的属性向量
    all_attribute = get_cls('all')
    coresspond_attribute = get_cls(data_type)
    for element in coresspond_attribute:
        correspond_index = all_attribute.index(element)
        attribute_list.append(all_attribute[correspond_index])
        # 然后根据数据类型来取属性向量
        
    attribute_matrix.close()

    return attribute_list


# 训练85个对应属性的支持向量机
def svm_train(train_img, train_attr, cls_list):

    svm_85 = []

    for attr in range(85):
        clf = svm.SVC()
        for batch_name, index in cls_list, len(train_attr):
            current_batch = train_img[batch_name]
            attr = train_attr[index][attr]

            # 拟合数据
            clf.fit(current_batch, attr)
    
        svm_85.append(clf)
    
    return svm_85


# 得到一个新的类别来测试
def random_test_pair(test_img, test_attr):
    # 取那个类别的图片
    num_cls = len(test_img.keys())
    cls_index = np.random.randint(0, num_cls)

    # 取那张图片
    num_img = len(len(test_img[cls_index]))
    img_index = np.random.randint(0, num_img)

    test_img = test_img[test_img[cls_index][img_index]]
    attr_vector = test_attr[cls_index]

    return test_img, attr_vector

# 得到图片的预测属性向量
def attr_predict(test_img, svm_list):
    attr = []
    for svm_id in len(svm_list):
        attr_temp = svm_list[svm_id].predict(test_img)
        attr.append(attr_temp)

    return attr