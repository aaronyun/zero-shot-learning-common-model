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

    split_img = read_split_img(dataset_path, split_name)
    # python字典中的key的存储是散序的，用key()取是顺序会变化
    all_class_name = split_img.keys()

    print("\n\n========================================")
    print("Features of %s split is under extractoring..." % split_name)
    print("========================================\n")

    with tf.Session() as sess:
        class_index = 1
        for class_name in all_class_name:
            print("extracting features for %s class" % class_name)

            class_img = split_img[class_name]
            vgg = Vgg19()
            vgg.build(class_img)
            fc_result = (vgg.fc7).eval()

            if class_index == 1:
                class_img_features = fc_result
            else:
                class_img_features = np.vstack((class_img_features, fc_result))

            class_index += 1

            print("features of %s class have been extractored" % class_name)
    
    writer(split_name, 'features', class_img_features)

    print("\n========================================")
    print("Features extractoring of %s split completed" % split_name)
    print("========================================\n\n")