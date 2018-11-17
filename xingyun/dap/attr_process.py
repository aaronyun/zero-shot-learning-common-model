# -*- coding: UTF-8 -*-

import io

import numpy as np

from dap.utils import img_count, class_name_of_split, extend_array, writer


def read_all_attrs(dataset_path):
    """Get all the attribute vectors.

    Args:
        dataset_path: the path where the data stored

    Returns:
        all_attrs: all the attributes, one class one vector
    """
    attr_file_path = dataset_path + '/' + r'predicate-matrix-binary.txt'
    attr_file = io.open(attr_file_path, 'r')

    all_attrs = []

    attr_count = 0
    while attr_count != 50:
        attr = np.array(attr_file.readline().split(' '), dtype=int)
        all_attrs.append(attr)
        attr_count += 1

    attr_file.close()

    return all_attrs

def read_split_attr(dataset_path, split_name):
    """Get attributes of a split of classes.

    Args:
        dataset_path: 
        split_name:

    Returns:
        split_attrs: 
    """
    all_img_path = dataset_path + r'/JPEGImages'

    all_attrs = read_all_attrs(dataset_path)
    all_class_name = class_name_of_split(dataset_path, 'all')
    split_class_name = class_name_of_split(dataset_path, split_name)

    # 下面功能的实现的前提是：classes.txt中的类别和predicate-matrix-binary.txt中的属性向量是一一对应的
    attr_count = 1
    for class_name in split_class_name:
        attr_index = all_class_name.index(class_name)
        correspond_attr = all_attrs[attr_index]

        if attr_count == 1:
            split_attrs = correspond_attr
        else:
            split_attrs = np.vstack((split_attrs, correspond_attr))

        attr_count += 1

    return split_attrs

def expand_attr(attr, class_name, dataset_path):
    """
    """
    # 得到类别名称和对应的属性后再进行扩充
    class_img_count = img_count(dataset_path, class_name)
    # 得到图片的数量然后进行扩充
    expanded_attr = np.tile(attr, (class_img_count, 1))

    return expanded_attr

def expanded_split_attrs(attr_to_expand, split_name):
    """
    """
    dataset_path = r'/data0/xingyun/AWA'
    cls_name = class_name_of_split(dataset_path, split_name)

    expanded_count = 1
    for attr_index in range(attr_to_expand.shape[0]):
        img_count = img_count(dataset_path, cls_name[attr_index])
        expanded_attr = expand_attr(attr_to_expand[attr_index], cls_name[attr_index], dataset_path)

        if expanded_count == 1:
            expanded_split_attr = expanded_attr
        else:
            expanded_split_attr = np.vstack((expanded_attr, expanded_split_attr))

    return expanded_split_attr