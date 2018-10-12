# -*- coding: UTF-8 -*-

import io
import tensorflow as tf
import numpy as np

from extractor.img_and_attr_reader import *
from extractor.vggNet import vgg19

def feature_extractor(dataset_path, split_name):
    """Extract the feature of images with vgg19.
    
    Write the features extracted into a text file.

    Args:
        split_name: which split of images need to be extracted
    """
    feature_file_name = split_name + '_features.txt'
    feature_file = io.open(feature_file_name, 'a')

    split_image = read_split_img(dataset_path, split_name)
    resized_split_image = resized_split_image(split_image)

    all_class_name = resized_split_image.keys()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for class_name in all_class_name:

            tf.reset_default_graph()

            # print("开始提取" + str(class_name) + "类图片的特征")
            class_image = resized_split_image[class_name]
            ignore_1, ignore_2, class_image_features, ignore_3 = vgg19(class_image, keep_prob=1)
            # print(str(class_name)+ "类的图片特征提取完成")

            # print("将" + str(class_name) + "类的图片特征写入文件中")
            feature_file.write("%s\n" % class_name)
            feature_file.write(str(class_image_features))
            # print(str(key) + "类的图片特征写入完成")

        print("恭喜!全部写入完成！！！")
        feature_file.close()

    return


    # # 打开文件
    # feature_file = io.open('img_features.txt', 'w')

    # set_list = img_set_dic.keys() # 得到所有的类别以进行循环处理
    # batch_count = 0
    # for set_name in set_list:
    #     batch_img = img_set_dic[set_name]

    #     # 创建session运行vgg
    #     with tf.session() as sess:
    #         tf.run(tf.global_variables_initializer())
            
    #         # 进行特征提取
    #         # 注意：features是一个tensor，形状是(num of examples, 1, 1, 1000)
    #         predictions, softmax, fc3, params = vgg19(batch_img, keep_prob)

    #         # 将得到的tensor转化为ndarray
    #         if batch_count == 0:
    #             batch_features = fc3.eval().squeeze()
    #         else:
    #             batch_features.vstack(fc3.eval().squeeze())

    #     # 多次调用np.load()函数时会重写文件内容
    #     np.save('features.npy', batch_features)