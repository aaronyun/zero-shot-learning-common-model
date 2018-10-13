# -*- coding: UTF-8 -*-

import io
import tensorflow as tf
import numpy as np

from preprocess.img_and_attr_reader import read_a_split_img
from preprocess.img_transform import resize_a_split_img
from preprocess.vggNet import vgg19

def feature_extractor(dataset_path, split_name):
    """Extract the feature of images with vgg19.
    
    Write the features extracted into a text file.

    Args:
        dataset_path: the dataset's path on your machine
        split_name: which split of images need to be extracted
    """
    a_split_of_img = read_a_split_img(dataset_path, split_name)
    resized_a_split_of_img = resize_a_split_img(a_split_of_img)

    all_class_name = resized_a_split_of_img.keys()
    class_index = 1
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for class_name in all_class_name:

            tf.reset_default_graph()

            class_image = resized_split_image[class_name]
            ignore_1, ignore_2, class_image_features, ignore_3 = vgg19(class_image, keep_prob=1)
            class_image_features = class_image_features.eval()

            if class_index == 1:
                class_image_features_stack = class_image_features
            else:
                class_image_features_stack = np.vstack((class_image_features_stack, class_image_features))
    
    feature_file_name = split_name + '_features.txt'
    np.save(feature_file_name, class_image_features_stack)

    return

def feature_reader(split_name):
    """Read a split of features stored in corresponding file.

    Args:
        split_name: the split which you want to read correspongding features

    Returns:
        features: features with regard to the given split
    """
    feature_file_name = split_name + '_features.txt'
    features = np.load(feature_file_name)

    return features