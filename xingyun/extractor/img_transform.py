# -*- coding: UTF-8 -*-

from skimage import transform
import numpy as np

def resize_single_img(img):
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

def resize_cls_img(cls_of_img):
    """Resize image in a class (e.g. antelope) into shape (224, 224, 3).

    Args:
        cls_of_img: class of images, ndarray of shape (num of images, height, width, channels)

    Returns:
        resized_cls_img: class of images has been resized, of shape (num of images, 224, 224, 3)
    """
    # 用来保存裁剪后的图片的列表
    resized_img_list = []

    # 对每张图片进行裁剪，然后保存到列表中
    for img_index in range(cls_of_img.shape[0]):
        resized_img = resize_single_img(cls_of_img[0])
        img_list.append(resized_img)

    # 将得到的list转换为ndarray
    resized_cls_img = np.asarray(resized_img_list)

    return resized_cls_img

def resize_split_image(split_image):
    """Resize all the images in a data split.

    Args:
        split_image: python dictionary containing all the classes of images

    Returns:
        resized_split_image: python dictionary containing resized images
    """

    resized_split_image = {}
    cls_name_list = split_image.keys()

    for cls_name in cls_name_list:
        resized_cls_img = resize_cls_img(split_image[cls_name])
        resized_split_image[cls_name] = resized_cls_img

    return resized_split_image