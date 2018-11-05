# -*- coding: UTF-8 -*-

import io
import os

import numpy as np
import tensorflow as tf
from sklearn import svm, naive_bayes
from sklearn.externals import joblib
from sklearn import datasets,metrics,model_selection,preprocessing
import time
from sklearn.naive_bayes import GaussianNB

os.environ["CUDA_VISIBLE_DEVICES"]="5,6"
#################################数据训练函数####################################

test_batch= []
cls_label= []
auc_cls = []
filename = 'AUC.txt'

#np.set_printoptions(threshold=np.nan)


def train_single_svm(train_features, train_attr,test_image, test_attr):
    start_time = time.time()
    classifier = svm.LinearSVC()

    classifier.fit(train_features, train_attr)
    print(train_attr.sum())

    print(("build svm cost: %ds" % (time.time() - start_time)))
    test_attr  = np.array(test_attr, dtype=int)

    test_batch.append(classifier.predict(test_image))
    print("预测值：", classifier.predict(test_image).sum())
    print("标签值：", test_attr.sum())

    metrics.f1_score(test_attr, classifier.predict(test_image))
#
    fpr, tpr, thresholds = metrics.roc_curve(test_attr, classifier.decision_function(test_image),
                                             pos_label=1)
#
#
    auc = metrics.auc(fpr, tpr)
    if(str(auc).strip()!='nan'):

        print("auc: " +str(auc)+"!")
        auc_cls.append(auc)

    return classifier,auc_cls




def get_cls():
    cls_list = []
    data_path = r'/data0/linjingkai/DataSet/AWA/testclasses.txt'

    cls_file = io.open(data_path, 'r')

    cls_name = ' '
    while cls_name != '':
        # 读取类别的名称并去掉换行符
        cls_name = cls_file.readline().rstrip('\n')
        # 确保没有加入无效的名称
        if len(cls_name) != 0:
            cls_list.append(cls_name)
    print("当前运行类别为：", cls_list)
    print(" ")

    cls_file.close()

    return cls_list

def get_Num(cls):

    path = r'/data0/linjingkai/DataSet/AWA/JPEGImages/' + str(cls)
    pathDir = os.listdir(path)
    count = 0

    for fn in pathDir:  # fn 表示的是文件名
        count = count + 1

    return count

def read_cls_attr(cls):
    with open('/data0/linjingkai/DataSet/AWA/classes.txt') as file_object:
        index = 0
        for line in file_object:
            #print(str(line))
            if(str(line).strip() == str(cls)):
                return index
            index+=1

        #print("当前类别所在序号：", index)

    # Attr_Array = np.loadtxt("/data0/linjingkai/DataSet/AWA/predicate-matrix-binary.txt")


def red_predicate_matrix_binary():
    with open('/data0/linjingkai/DataSet/AWA/classes.txt') as file_object:
        all_index_batch = []
        #index = 0
        for i in range(50):
            all_index_batch.append(i)


        #print("当前类别所在序号：", index)

        Attr_Array = np.loadtxt("/data0/linjingkai/DataSet/AWA/predicate-matrix-binary.txt")
    return Attr_Array,all_index_batch


def bayes_train(test_img, test_attr):
    clf = GaussianNB()
    clf.fit(test_img, test_attr)


    return clf

feature_train_image = np.load("train_fc7_image.npy")
print("当前训练图片特征大小： ", feature_train_image.shape)
feature_train_label = np.load("train_fc7_label.npy")
print("当前训练标签大小： ",feature_train_label.shape)
attrbute = np.array(feature_train_label, dtype=int)
attrbute = attrbute.T

test_image = np.load("test_fc7_image.npy")
test_label = np.load("test_fc7_label.npy")
print("当前测试图片特征大小： ", test_image.shape)
print("当前测试标签大小： ",test_label.shape)
test_attriute = np.array(test_label, dtype=int)
test_attriute = test_label.T
# svm= []
with tf.Session() as sess:
    start_time = time.time()
    for attr_index in range(85):
        attr = attrbute[attr_index]
        test_attr = test_attriute[attr_index]
        attr = attr.flatten()
        test_attr = test_attr.flatten()
##
        print("当前训练第 " + str(1 + attr_index) + " 个支持向量机")
#
        SVM = train_single_svm(feature_train_image, attr, test_image,test_attr)
##
        print("当前支持向量机训练完成")
   # with open(filename,'w') as file_object:
   #     for i in range(85):
   #         file_object.write(str(auc_cls[i])+'\n')
##
##
    Array_all_bach = np.array(test_batch)
    Array_all_bach = Array_all_bach.T
#
    class_list = get_cls()
#
    for image_class in class_list:
        img_Num = get_Num(image_class)
        print("当前类别：", str(image_class))
        print("当前类别数目：", img_Num)
        for i in range(img_Num):
            count_cls = read_cls_attr(image_class)
            cls_label.append(count_cls)
        print("当前类别所在序号：", count_cls)

    cls_Array = np.array(cls_label, dtype = int)
    cls_Array.flatten()
    print(cls_Array)
   # print(cls_Array.shape)
   # print(cls_Array)

    # 加载贝叶斯训练数据
    attr_all,index_all = red_predicate_matrix_binary()
    attr_all =  np.array(attr_all, dtype = int)
    print("attr_all", attr_all.shape)
    index_all = np.array(index_all, dtype = int)
    index_all.flatten()
    print(index_all.shape)

    sum = 0
    i = 0
    for line in auc_cls:
        sum = float(line) + sum
        i += 1
    print("平均AUC：", sum / i)

    # 训练贝叶斯
    print(Array_all_bach)

    bayes = bayes_train(Array_all_bach, cls_Array)
    #predict = bayes.predict(attr_all)
    predict = bayes.predict(Array_all_bach)
    print(predict)
    count = 0
    for left, right in zip(predict, cls_Array):
        if left == right:
            count += 1
    print("正确率：", count / len(cls_Array))
    print(("all preform cost: %ds" % (time.time() - start_time)))





