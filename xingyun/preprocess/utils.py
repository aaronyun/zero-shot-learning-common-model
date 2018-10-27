# -*- coding: UTF-8 -*-
import numpy as np
import io
from skimage import transform

def get_class_name(dataset_path, split_name):
    """Fetch all the name of a split.

    Args:
        dataset_path: the path where the split file stored
        split_name: which split you want to know

    Returns:
        all_class_name: python list stored all the class name of a split
    """
    all_class_name = []

    if split_name == 'train':
        split_file_path = dataset_path + '/' + split_name + 'classes.txt'
    elif split_name == 'valid':
        split_file_path = dataset_path + '/' + split_name + 'classes.txt'
    elif split_name == 'test':
        split_file_path = dataset_path + '/' + split_name + 'classes.txt'
    elif split_name == 'all':
        split_file_path = dataset_path + '/' + 'classes.txt'
    else:
        print("\nWarning: No split called " + str(split_name) + "\n")
        return

    split_file = io.open(split_file_path, 'r')

    class_name = ' '
    while class_name != '':
        class_name = split_file.readline().rstrip('\n')
        
        if len(class_name) != 0:
            all_class_name.append(class_name)

    split_file.close()

    return all_class_name

def data_writer(split_name, data_type, data):
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
    resized_img = transform.resize(croped_img, (224, 224, 3))

    return resized_img