# -*- coding: UTF-8 -*-

import tensorflow as tf

#--------------------HELPER FUNCTIONS---------------------#

def conv_op(inputs, num_layer, channels_out, kernel_h=3, kernel_w=3, stride_h=1, stride_w=1, params=[]):
    """
    Create convolution layer and compute.

    Arguments
    ---
    inputs--input of current layer, tensor of shape (number of examples, height, width , channels)
    kernel_h--height of kernel in current convolution layer
    kernel_w--width of kernel in current convolution layer
    channels_out--channels of output of current layer(number of channels)
    stride_h--stride's height
    stride_w--stride's width
    params--list used to store parameters in this layer

    Returns
    ---
    activation--output of this convolution layer
    """

    # get the channel number of input
    channels_in = inputs.get_shape()[-1].value

    # create kernel
    kernel = tf.get_variable(name="kernel"+num_layer, shape=[kernel_h, kernel_w, channels_in, channels_out], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
    # create biases
    biases_init = tf.constant(0.0, dtype=tf.float32, shape=[channels_out])
    biases = tf.Variable(biases_init, trainable=True, name='conv_bias'+num_layer)

    # convolution operation
    conv = tf.nn.conv2d(inputs, kernel, strides=(1,stride_h,stride_w,1), padding='SAME')

    # computation
    z = tf.nn.bias_add(conv, biases)
    activation = tf.nn.relu(z)

    params += [kernel, biases]

    return activation

def fc_op(inputs, num_layer, channels_in, channels_out, params):
    """
    Create full-connected layer and compute.

    Arguments
    ---
    inputs--input of current layer, of shape (number of examples, fc_in_length)
    channels_out--channels of output
    params--list used to store parameters in this layer

    Returns
    ---
    activation--output of this layer
    """

    # create kernel(weight)
    # the shape of kernel is (channels_in, channels_out), this is because tf.nn.relu_layer() computes x*w + b rather than w*x + b
    kernel = tf.get_variable(name="kernel"+num_layer, shape=[channels_in, channels_out], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

    # create biases
    biases = tf.get_variable(name='biases'+num_layer, shape=[channels_out], dtype=tf.float32, initializer=tf.constant_initializer())

    # full-connected operation
    activation = tf.nn.relu_layer(inputs, kernel, biases)

    params += [kernel, biases]

    return activation

def maxpool_op(inputs, kernel_h=2, kernel_w=2, stride_h=2, stride_w=2):
    """
    Max pooling operation.

    Arguments
    ---
    inputs--images to be max-pooling, of shape (number of examples, height, width, channels)
    kernel_h--height of kernel
    kernel_w--width of kernel
    stride_h--stride's height
    stride_w--stride's width

    Returns
    ---
    max_pool--output of max-pooling layer
    """

    max_pool = tf.nn.max_pool(inputs, ksize=[1,kernel_h,kernel_w,1], strides=[1,stride_h,stride_w,1], padding='SAME')

    return max_pool

#--------------------VGG-19 structure--------------------#

def vgg19(inputs, keep_prob):
    """VGG19 network.

    Args:
        inputs: inputs of shape (num of examples, height, width, channels)
        keep_prob: ratio of eliminated neurons

    Returns:
        predictions: the output of vggnet
        softmax: output in softmax formation
        fc3: the last full-connected layer's output
        params: list of params used in this network
    """

    # number of images
    num_img = inputs.shape[0]

    # list used to store parameters
    params = []

    #----------PART I----------#

    conv1_1 = conv_op(inputs, num_layer='conv1_1', channels_out=64, params=params)

    conv1_2 = conv_op(conv1_1, num_layer='conv1_2',channels_out=64, params=params)

    pool_1 = maxpool_op(conv1_2)

    #----------PART II----------#

    conv2_1 = conv_op(pool_1, num_layer='conv2_1', channels_out=128, params=params)

    conv2_2 = conv_op(conv2_1, num_layer='conv2_2',  channels_out=128, params=params)

    pool_2 = maxpool_op(conv2_2)

    #----------PART III----------#

    conv3_1 = conv_op(pool_2, num_layer='conv3_1', channels_out=256, params=params)

    conv3_2 = conv_op(conv3_1, num_layer='conv3_2', channels_out=256, params=params)

    conv3_3 = conv_op(conv3_2, num_layer='conv3_3', channels_out=256, params=params)

    conv3_4 = conv_op(conv3_3, num_layer='conv3_4', channels_out=256, params=params)

    pool_3 = maxpool_op(conv3_4)

    #----------PART IV----------#

    conv4_1 = conv_op(pool_3, num_layer='conv4_1', channels_out=512, params=params)

    conv4_2 = conv_op(conv4_1, num_layer='conv4_2',  channels_out=512, params=params)

    conv4_3 = conv_op(conv4_2, num_layer='conv4_3', channels_out=512, params=params)

    conv4_4 = conv_op(conv4_3, num_layer='conv4_4', channels_out=512, params=params)

    pool_4 = maxpool_op(conv4_4)

    #----------PART V-----------#
    
    conv5_1 = conv_op(pool_4, num_layer='conv5_1', channels_out=512, params=params)

    conv5_2 = conv_op(conv5_1, num_layer='conv5_2', channels_out=512, params=params)

    conv5_3 = conv_op(conv5_2, num_layer='conv5_3', channels_out=512, params=params)

    conv5_4 = conv_op(conv5_3, num_layer='conv5_4', channels_out=512, params=params)

    conv5_5 = conv_op(conv5_4, num_layer='conv5_5', channels_out=512, params=params)

    pool_5 = maxpool_op(conv5_5) # shape: (num, 7, 7, 512)

    #----------FLATTEN----------#

    # get the shape of convolutional output
    conv_out_shape = pool_5.get_shape()

    # compute the length of vector input into full-connected layer
    fc_in_length = conv_out_shape[1].value * conv_out_shape[2].value * conv_out_shape[3].value

    # flatten the output to be a vector
    flatten = tf.reshape(pool_5, [num_img, fc_in_length], name='flatten')

    #----------FC----------#

    # full-connected layer 1
    fc1 = fc_op(flatten, num_layer='fc1', channels_in=fc_in_length, channels_out=4096, params=params)
    fc1_drop = tf.nn.dropout(fc1, keep_prob, name='fc1_drop')

    # full-connected layer 2
    fc2 = fc_op(fc1_drop, num_layer='fc2', channels_in=4096, channels_out=4096, params=params)
    fc2_drop = tf.nn.dropout(fc2, keep_prob, name='fc2_drop')

    # full-connected layer 3
    fc3 = fc_op(fc2_drop, num_layer='fc3', channels_in=4096, channels_out=1000, params=params)
    
    #----------SOFTMAX----------#

    softmax = tf.nn.softmax(fc3)
    predictions = tf.argmax(softmax, axis=1)

    return predictions, softmax, fc3, params

#--------------------VGG-16 structure--------------------#

def vgg16():
    return