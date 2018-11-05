import numpy as np
import tensorflow as tf

import vgg19
import utils
import os

os.environ["CUDA_VISIBLE_DEVICES"]="3,4,5"


# with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:

all_batch = []
Attr_label = []

class_list = utils.get_cls()

with tf.Session() as sess:
    images = tf.placeholder("float", [None, 224, 224, 3])

    vgg = vgg19.Vgg19()
    with tf.name_scope("content_vgg"):
        vgg.build(images)
        count_cls = 0
    for image_class in class_list:
        img_Num = utils.get_Num(image_class)

        print("当前图片类别：", image_class)
        print("当前图片总数目：", img_Num)
        print("当前类别序号：", count_cls)
        print(" ")
        attr = utils.read_cls_attr(image_class)

        sess.graph.finalize()
        for i in range(0, img_Num):

            batch, count = utils.img_deal(image_class, i)

            fc7 = sess.run(vgg.fc7, feed_dict={images: batch})
            all_batch.append(fc7)
            Attr_label.append(attr)
            #Attr_label.append(count_cls)

            if (count == img_Num):
                print("当前类特征提取完毕！")
                print(" ")
                break
        count_cls += 1
    print("所有特征提取完毕！")

    Array_all_bach = np.array(all_batch)
    print("图片特征数组规格：", Array_all_bach.shape)
    np.save("test_fc7_image.npy", Array_all_bach.reshape(-1, 4096))
#
    Attr_Array = np.array(Attr_label)
    print("标签属性数组：", Attr_Array.shape)
    np.save("test_fc7_label.npy", Attr_Array)










   # images = tf.placeholder("float", [None, 224, 224, 3])
   # feed_dict = {images: batch}
   # vgg = vgg19.Vgg19()
   # with tf.name_scope("content_vgg"):
   #     vgg.build(images)
   # fc6 = sess.run(vgg.fc6, feed_dict=feed_dict)
   # print(fc6)
   # np.save('feature_train.npy', fc6)
   # c = np.load("feature_train.npy")
   # print(c.shape)

   # utils.print_prob(prob[0], './synset.txt')
   # utils.print_prob(prob[1], './synset.txt')
