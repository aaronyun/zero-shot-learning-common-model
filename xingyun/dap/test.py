# -*- coding: UTF-8 -*-

from skimage import io as skio
from model import attr_predict, cls_predict

# 用一张未知类别的图片进行测试
predict_img = skio.imread('F:\\datasets\\AWA\\JPEGImages\\hippopotamus\\hippopotamus_10001.jpg')
attribute = attr_predict(predict_img, svm_85)
cls_name = cls_predict(attribute, bayes_clf)
print(cls_name)