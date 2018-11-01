# -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np

from preprocess.utils import get_class_name, writer
from preprocess.img_and_attr_reader import read_split_img
from preprocess.vgg19 import Vgg19

def feature_extractor(dataset_path, split_name):
    """Extract the feature of images with vgg19.
    
    Write the features extracted into a .npy file.

    Args:
        dataset_path: the dataset's path on your machine
        split_name: which split of images need to be extracted
    """
    print("开始提取" + str(split_name) + "数据划分的特征")
    print("======================\n")

    a_split_img = read_split_img(dataset_path, split_name)

    initializer = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(initializer)

        class_index = 1
        all_class_name = a_split_img.keys()
        for class_name in all_class_name:
            print("正在提取" + str(class_name) + "类的特征")

            class_img = a_split_img[class_name]
            vgg = Vgg19()
            vgg.build(class_img)
            fc_result = (vgg.fc8).eval()
            # print(str(class_name) + "类特征的形状: " + str(rc_result.shape))

            if class_index == 1:
                class_img_features = fc_result
            else:
                class_img_features = np.vstack((class_img_features, fc_result))

            # print("class_img_features的形状: " + str(class_img_features.shape))
            class_index += 1

            print(str(class_name) + "类的特征提取完成\n")
    
    writer(split_name, 'features', class_img_features)

    print("\n" + str(split_name) + "数据划分的特征提取完成")
    print("======================")