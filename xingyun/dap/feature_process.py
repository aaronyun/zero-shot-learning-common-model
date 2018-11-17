# -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np

from dap.utils import class_name_of_split, writer
from dap.img_process import read_cls_img
from dap.vgg19 import Vgg19

def get_batch(batch_count, batch_size, cls_img):
    """
    Args:
        batch_count: the batch index for which you want to fetch
        cls_img: the class of images to be seperate
    """
    complete_batch_num = cls_img.shape[0]//batch_size
    
    # 因为 batch_count 表示的是第几个 batch，所以要减 1
    batch_count = batch_count - 1

    if (batch_count > complete_batch_num):
        batch_start = complete_batch_num * 50

        batch_img = cls_img[batch_start:]
    else:
        batch_start = batch_count * batch_size
        batch_end = (batch_count + 1) * batch_size

        batch_img = cls_img[batch_start: batch_end]

    return batch_img

def extract(dataset_path, split_name):
    """Extract feature for a split of images with vgg19.
    
    Write the features extracted into split_features.npy.

    Args:
        dataset_path: the dataset's path on your machine
        split_name: which split of images need to be extracted
    """

    all_img_path = dataset_path + r'/JPEGImages'

    sess = tf.Session()
    img_holder = tf.placeholder('float', shape=(None, 224, 224, 3))

    print("\n\n==========")
    vgg = Vgg19()
    vgg.build(img_holder)
    print("==========")

    print("\n\n========================================")
    print("Features of %s split is under extracting..." % split_name)
    print("========================================\n")

    cls_count = 1
    batch_size = 50

    all_cls_name = class_name_of_split(dataset_path, split_name)
    for cls_name in all_cls_name:
        print("\n++++++++++++++++++++++++++++++++++++++++")
        print("extracting features for %s class" % cls_name)

        cls_img = read_cls_img(all_img_path, cls_name)
        batch_num = cls_img.shape[0]//batch_size
        for batch_count in range(1, batch_num + 1):
            # 取一个batch的图片
            batch_img = get_batch(batch_count, 50, cls_img)
            # 提取当前 batch 的特征
            fc7 = sess.run(vgg.fc7, feed_dict={img_holder:batch_img})

            # 堆叠各个 batch 的特征
            if batch_count == 1:
                cls_img_features = fc7
            else:
                cls_img_features = np.vstack((cls_img_features, fc7))

        # 处理剩下的图片
        last_batch_img = get_batch(batch_num+1, 50, cls_img)
        # print("last batch shape: %s" % str(last_batch_img.shape))
        fc7 = sess.run(vgg.fc7, feed_dict={img_holder:last_batch_img})

        # 将最后一组特征和之前的堆叠起来
        cls_img_features = np.vstack((cls_img_features, fc7))
        print("shape of features of %s class: %s" % (cls_name, str(cls_img_features.shape)))

        # 测试一次性对一个类有没有问题
        # cls_img_features = sess.run(vgg.fc7, feed_dict={img_holder:cls_img})

        # 对提取到的各个类别的特征进行堆叠
        if cls_count == 1:
            split_img_features = cls_img_features
        else:
            split_img_features = np.vstack((split_img_features, cls_img_features))

        cls_count += 1

        print("features of %s class have been extractored" % cls_name)
        print("++++++++++++++++++++++++++++++++++++++++\n")

    sess.close()

    print("//////////Current split features has shape of: %s//////////" % str(split_img_features.shape))
    writer(split_name, 'features', split_img_features)

    print("\n========================================")
    print("Features extractoring of %s split completed" % split_name)
    print("========================================\n\n")