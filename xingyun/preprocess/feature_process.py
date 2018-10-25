# -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np

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

            class_image = a_split_img[class_name]
            vgg = Vgg19() 
            vgg.build(class_image)
            fc_result = (vgg.fc8).eval()
            # print(str(class_name) + "类特征的形状: " + str(rc_result.shape))

            if class_index == 1:
                class_img_features = fc_result
            else:
                class_img_features = np.vstack((class_img_features, fc_result)

            # print("class_img_features的形状: " + str(class_img_features.shape))
            class_index += 1

            print(str(class_name) + "类的特征提取完成\n")

    feature_file_name = split_name + '_features'
    np.save(feature_file_name, class_img_features)

    print("\n" + str(split_name) + "数据划分的特征提取完成")
    print("======================")

    return

def feature_reader(split_name):
    """Read a split of features stored in corresponding file.

    Args:
        split_name: the split which you want to read correspongding features

    Returns:
        features: features with regard to the given split
    """
    feature_file_name = split_name + '_features.npy'
    features = np.load(feature_file_name)

    return features