# coding: UTF-8
'''''''''''''''''''''''''''''''''''''''''''''''''''''
   file name: vgg19.py
   create time: 2017年04月13日 星期四 10时40分26秒
   author: Jipeng Huang
   e-mail: huangjipengnju@gmail.com
   github: https://github.com/hjptriplebee
'''''''''''''''''''''''''''''''''''''''''''''''''''''
import tensorflow as tf
from skimage import io as skio
from skimage import transform
import numpy as np
import io
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

model_path =  r'/data0/linjingkai/DataSet/AWA/VGG19.ckpt'

# define different layer functions
# we usually don't do convolution and pooling on batch and channel

def maxPoolLayer(x, kHeight, kWidth, strideX, strideY, name, padding = "SAME"):
    """max-pooling"""
    return tf.nn.max_pool(x, ksize=[1, kHeight, kWidth, 1],
                          strides=[1, strideX, strideY, 1], padding=padding, name=name)

def dropout(x, keepPro, name = None):
    """dropout"""
    return tf.nn.dropout(x, keepPro, name)

def fcLayer(x, inputD, outputD, reluFlag, name):
    """fully-connect"""
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape=[inputD, outputD], dtype=tf.float32)
        b = tf.get_variable("b", [outputD], dtype=tf.float32)
        out = tf.nn.xw_plus_b(x, w, b, name=scope.name)
        if reluFlag:
            return tf.nn.relu(out)
        else:
            return out

def convLayer(x, kHeight, kWidth, strideX, strideY,
              featureNum, name, padding="SAME"):
    """convlutional"""
    channel = int(x.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape=[kHeight, kWidth, channel, featureNum])
        b = tf.get_variable("b", shape=[featureNum])
        featureMap = tf.nn.conv2d(x, w, strides=[1, strideY, strideX, 1], padding=padding)
        out = tf.nn.bias_add(featureMap, b)
        #return tf.nn.relu(tf.reshape(out, featureMap.get_shape().as_list()), name=scope.name)
        return tf.nn.relu(out)

def VGG19(x, keepPro):
    """build model"""
    conv1_1 = convLayer(x, 3, 3, 1, 1, 64, "conv1_1")
    conv1_2 = convLayer(conv1_1, 3, 3, 1, 1, 64, "conv1_2")
    pool1 = maxPoolLayer(conv1_2, 2, 2, 2, 2, "pool1")

    conv2_1 = convLayer(pool1, 3, 3, 1, 1, 128, "conv2_1")
    conv2_2 = convLayer(conv2_1, 3, 3, 1, 1, 128, "conv2_2")
    pool2 = maxPoolLayer(conv2_2, 2, 2, 2, 2, "pool2")

    conv3_1 = convLayer(pool2, 3, 3, 1, 1, 256, "conv3_1")
    conv3_2 = convLayer(conv3_1, 3, 3, 1, 1, 256, "conv3_2")
    conv3_3 = convLayer(conv3_2, 3, 3, 1, 1, 256, "conv3_3")
    conv3_4 = convLayer(conv3_3, 3, 3, 1, 1, 256, "conv3_4")
    pool3 = maxPoolLayer(conv3_4, 2, 2, 2, 2, "pool3")

    conv4_1 = convLayer(pool3, 3, 3, 1, 1, 512, "conv4_1")
    conv4_2 = convLayer(conv4_1, 3, 3, 1, 1, 512, "conv4_2")
    conv4_3 = convLayer(conv4_2, 3, 3, 1, 1, 512, "conv4_3")
    conv4_4 = convLayer(conv4_3, 3, 3, 1, 1, 512, "conv4_4")
    pool4 = maxPoolLayer(conv4_4, 2, 2, 2, 2, "pool4")

    conv5_1 = convLayer(pool4, 3, 3, 1, 1, 512, "conv5_1")
    conv5_2 = convLayer(conv5_1, 3, 3, 1, 1, 512, "conv5_2")
    conv5_3 = convLayer(conv5_2, 3, 3, 1, 1, 512, "conv5_3")
    conv5_4 = convLayer(conv5_3, 3, 3, 1, 1, 512, "conv5_4")
    pool5 = maxPoolLayer(conv5_4, 2, 2, 2, 2, "pool5")

    fcIn = tf.reshape(pool5, [-1, 7 * 7 * 512])
    fc6 = fcLayer(fcIn, 7 * 7 * 512, 4096, True, "fc6")
    dropout1 = dropout(fc6, keepPro)

    fc7 = fcLayer(dropout1, 4096, 4096, True, "fc7")
    dropout2 = dropout(fc7, keepPro)

    fc8 = fcLayer(dropout2, 4096, 40, True, "fc8")

    return fc8, pool5

def get_cls():
    cls_list = []
    data_path = r'/data0/linjingkai/DataSet/AWA/trainclasses.txt'

    cls_file = io.open(data_path, 'r')

    cls_name = ' '
    while cls_name != '':
        # 读取类别的名称并去掉换行符
        cls_name = cls_file.readline().rstrip('\n')
        # 确保没有加入无效的名称
        if len(cls_name) != 0:
            cls_list.append(cls_name)
    print("类别为：", cls_list)

    cls_file.close()

    return cls_list

def img_deal(cls_list):
    batch = []
    label = []

    # 添加图片路径
    count = 1
    for img_class in cls_list:
        path = r'/data0/linjingkai/DataSet/AWA/JPEGImages/' + str(img_class)
        pathDir = os.listdir(path)
        #count = 0
        #for fn in os.listdir(path):  # fn 表示的是文件名
        #    count = count + 1
        #Num = count/1000

        sample = random.sample(pathDir, 1)

        for name in sample:
            img_path = path + '/' + str(name)
            img = skio.imread(img_path)
            # 裁剪图像
            img_resized = transform.resize(img, (224, 224))
            label.append(count)
           # print(img_resized.shape)
           # print(name)

            batch.append(img_resized)


        count += 1

    batch_img = np.stack(batch)

    return batch_img, label


xs = tf.placeholder(tf.float32, [None, 224,224,3])
ys = tf.placeholder(tf.int32, [None,])
keep_prob = tf.placeholder(tf.float32)

logit, feature = VGG19(xs, 0.5)

clist = get_cls()

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(ys, 40), logits= logit))

train_step = tf.train.AdamOptimizer(10e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(logit, 1), tf.argmax(tf.one_hot(ys, 40), 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
saver = tf.train.Saver() 
sess.run(tf.global_variables_initializer())


for i in range(10000):
    img, lab = img_deal(clist)

    a, _, result = sess.run([cross_entropy,train_step,accuracy], feed_dict={xs: img, ys: lab, keep_prob: 0.5})
    if i % 2 == 0:

        print("current step：", i)
        print("损失率:", a)
        print("准确率：", result)
        print(" ")

save_path = saver.save(sess, model_path)
print("路径：", model_path)







