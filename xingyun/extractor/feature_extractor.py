# -*- coding: UTF-8 -*-

import io
import tensorflow as tf
import numpy as np

from im_read import image_holder
from im_preprocess import load_image_set
from vggNet import vgg19

# 从终端获取图片类别的路径
target_path = input("输入图片类别所在的路径：\n")
print("正在提取。。。")

# 打开文本文件进行写入
feature_file = io.open('features_extracted.txt', 'w+')
# 创建图片的容器
holder = image_holder(target_path)

# 对每一类图片进行裁剪然后提取特征
for key in holder.keys():
    tf.reset_default_graph()

    print("开始处理" + str(key) + "类的图片")
    # 首先找到当前图片类别的路径
    path_temp = target_path + '/' + str(key)
    # 对当前类别的图片进行裁剪并保存到临时的列表中
    croped_img_set = load_image_set(path_temp)
    # 将当前类别的图片转为tensor然后输入到vgg19得到结果
    tensor_image_set = tf.convert_to_tensor(croped_img_set)
    predictions, softmax, vgg_image_set, parameters = vgg19(tensor_image_set, keep_prob=1)

    print("将" + str(key) + "类的图片结果写入文件中。。。")

    # 将得到的图片结果写入到文本文件中
    feature_file.write("%s: \n" % key)
    feature_file.write(str(vgg_image_set))
    print(str(key) + "类的图片结果写入完成。")

print("恭喜！全部写入完成！！！")
# 关闭文本文件
feature_file.close()

# images = im_read("../datasets/AWA/JPEGImages")

# # python dictionary used to stored features
# feature_dic = {}

# # file stor the features
# feature_file = io.open("features_extracted.txt", "r+")

# # iterate different type of animals
# for key in images.keys():
#     # get a set of images
#     inputs_tensor = tf.convert_to_tensor(images[key], dtype=tf.int64)
#     # vgg
#     predictions, softmax, fc3, params = vgg19(inputs_tensor, keep_prob=1)

#     # put the output of convolution operation in the dictionary
#     feature_dic[key] = fc3

#     # write features into text
#     feature_file.write(str(key) + ":\n" + str(fc3))

# feature_file.close()