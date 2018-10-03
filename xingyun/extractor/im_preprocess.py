from skimage import io
from skimage import transform
import os
import numpy as np

def load_image(path):
    # load image
    img = io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = transform.resize(crop_img, (224, 224, 3))
    return resized_img

def load_image_set(path):

    # 获取当前这个类别的图片的名字
    img_name_list = os.listdir(path)
    # 用来保存裁剪后的图片的列表
    img_list = []

    # 对每张图片进行裁剪，然后保存到列表中
    for single_img in img_name_list:
        resized_img = load_image(path + '/' + str(single_img))
        img_list.append(resized_img)

    # print(str(img_list[0]))
    # print(str(len(img_list)))

    # 把得到的list转为numpy.ndarray
    img_array = np.array(img_list, dtype=np.float32)
    return img_array