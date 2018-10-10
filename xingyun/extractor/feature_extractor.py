# -*- coding: UTF-8 -*-

import io
import tensorflow as tf
import numpy as np

from im_read import image_holder
from im_preprocess import load_image_set
from vggNet import vgg19


# 已经读取完了图片，然后进行特征提取
# 输入：数据划分的图片的字典
# 输出：包含特征的文本文件
def feature_extractor(img_set_dic):
    """Extract the feature of images with vgg19.
    
    Write the features extracted in text file.

    Args:
        img_set_dic: images need to be extracted
    """

    feature_file = io.open('features_extracted.txt', 'w+')

    # 当前数据划分包含的所有图片类别
    img_cls_name = img_set_dic.keys()

    # 对每一类图片进行裁剪然后提取特征
    for cls_name in img_cls_name:
        tf.reset_default_graph()

        print("开始提取" + str(cls_name) + "类的图片的特征")
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

    print("恭喜!全部写入完成！！！")
    # 关闭文本文件
    feature_file.close()


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