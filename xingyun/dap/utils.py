# -*- coding: UTF-8 -*-
import io
import os

import numpy as np
from skimage import transform

########################### functionality utilities ############################

def class_name_of_split(dataset_path, split_name):
    """Fetch all the class name in a split.

    Args:
        dataset_path: the path where the split file stored
        split_name: which split you want to know

    Returns:
        class_name_of_split: python list stored all the class name of a split
    """

    if split_name == 'train':
        split_file_path = dataset_path + '/' + split_name + 'classes.txt'
    elif split_name == 'valid':
        split_file_path = dataset_path + '/' + split_name + 'classes.txt'
    elif split_name == 'test':
        split_file_path = dataset_path + '/' + split_name + 'classes.txt'
    elif split_name == 'trainvalid':
        split_file_path = dataset_path + '/' + split_name + 'classes.txt'
    elif split_name == 'all':
        split_file_path = dataset_path + '/' + 'classes.txt'
    else:
        print("\nWarning: No split called " + str(split_name) + "\n")
        return

    class_name_of_split = []
    split_file = io.open(split_file_path, 'r')

    cls_name = ' '
    while len(cls_name) != 0:
        cls_name = split_file.readline().rstrip('\n')
        # 确保最后一次循环时读到的空字符串不被写入
        if cls_name != '':
            class_name_of_split.append(cls_name)

    split_file.close()

    return class_name_of_split

############################ file process utilities ############################

def writer(split_name, data_type, data):
    """Write data into .npy file.

    Args:
        split_name: which split you are tackling
        data_type: string, 'features' or 'attributes'
        data: the data need to be stored
    """
    if data_type == 'features':
        file_name = split_name + '_features'
    elif data_type == 'attributes':
        file_name = split_name + '_attributes'
    else:
        print("\nWarning: The file type you specified is not needed!\n")
        return

    np.save(file_name, data)

    return

def reader(split_name, data_type):
    """Read a split of features or attributes stored in corresponding file.

    Args:
        split_name: the split which you want to read
        data_type: string, 'features' or 'attributes'

    Returns:
        data: features or attributes with regard to the given split
    """
    if data_type == 'features':
        file_name = split_name + '_features.npy'
    elif data_type == 'attributes':
        file_name = split_name + '_attributes.npy'
    
    data = np.load('../data/' + file_name)

    return data

############################ data process utilities ############################

def img_count(dataset_path, cls_name):
    """Get the number of images in a class.

    Args:
        dataset_path: the path where all the images stored
        cls_name: which class you want to count

    Returns:
        num_of_imgs: how many images in a class
    """
    class_path = dataset_path + r'/JPEGImages/' + cls_name
    all_img_name = os.listdir(class_path)

    num_of_imgs = len(all_img_name)

    return num_of_imgs

def resize_img(img):
    """Resize an image into shape (224, 224, 3).

    Args:
        img: the image will be resized, ndarray of shape (height, width, channels)

    Returns:
        resized_img: the image has been resized
    """
    # 将图片的灰度值保留在[0,1]区间
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()

    # 以图片中心为基准点，把图片裁剪成正方形
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    croped_img = img[yy: yy + short_edge, xx: xx + short_edge]

    # 将图片缩小为(224, 224)的大小
    resized_img = transform.resize(croped_img, (224, 224, 3), mode='constant')

    return resized_img

# def cls_to_int(cls_in_strings):
#     """Convert list of classes into list of ints.
    
#     Args:
#         cls_in_strings: a list of string represents class name

#     Returns:
#         cls_in_ints: list of int
#     """
#     convert_rule = 
#     {'antelope':1, 'grizzly+bear':2 ,'killer+whale':3 ,'beaver':4, 
#     'dalmatian':5, 'persian+cat':6, 'horse':7, 'german+shepherd':8, 
#     'blue+whale':9, 'siamese+cat':10, 'skunk':11, 'mole':12, 
#     'tiger':13, 'hippopotamus':14, 'leopard':15, 'moose':16, 
#     'spider+monkey':17, 'humpback+whale':18, 'elephant':19, 'gorilla':20, 
#     'ox':21, 'fox':22, 'sheep':23, 'seal':24, 
#     'chimpanzee':25, 'hamster':26, 'squirrel':27, 'rhinoceros':28, 
#     'rabbit':29, 'bat':30, 'giraffe':31, 'wolf':32, 'chihuahua':33, 
#     'rat':34, 'weasel':35, 'otter':36, 'buffalo':37, 
#     'zebra':38, 'giant+panda':39, 'deer':40, 'bobcat':41, 
#     'pig':42, 'lion':43, 'mouse':44, 'polar+bear':45, 
#     'collie':46, 'walrus':47, 'raccoon':48, 'cow':49, 
#     'dolphin':50}

#     cls_in_ints = []
#     for cls_string in cls_in_strings:
#         cls_in_ints.append(convert_rule[cls_string])
#     cls_in_ints = np.array(cls_in_ints)

#     return cls_in_ints

def extend_array(array_to_extend):
    """
    """
    dataset_path = r'/data0/xingyun/AWA'

    cls_count = 1
    for cls_name in array_to_extend:
        img_num_of_cls = img_count(dataset_path, cls_name)
        single_cls_extended = np.repeat(cls_name, img_num_of_cls)

        if cls_count == 1:
            extended_array = single_cls_extended
        else:
            extended_array = np.concatenate((extended_array, single_cls_extended))
        
        cls_count += 1

    return extended_array

# def features_with_chi2_kernel(X, gamma):
#     """
#     """
#     kernel = chi2_kernel(X, gamma=gamma)

#     return kernel

def vector_sim_compute(predict, attr):
    """Compute the similarity between predict attributes probability and test class.

    Args:
        predict: 
        attr: 
    """
    assert(predict.shape == attr.shape)

    similarity = 1
    length = attr.shape[0]
    for index in range(length):
        if attr[index] == 0:
            temp_entry = 1 - predict[index]
        else:
            temp_entry = predict[index]
        
        similarity *= temp_entry

    return similarity