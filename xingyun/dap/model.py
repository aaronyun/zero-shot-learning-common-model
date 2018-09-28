# -*- coding: UTF-8 -*-

import numpy as np

# 属性预测
def attr_predict(img, svm_list):
    # 保存得到的属性
    attr = []
    # 用前面得到的85个SVM预测新图片的属性值
    for svm_id in len(svm_list):
        attr_temp = svm_list[svm_id].predict(img)
        attr.append(attr_temp)

    return np.array(attr)

# 类别预测
def cls_predict(attr, bayes_clf):

    cls_name = bayes_clf.predict(attr)

    return cls_name

# 最终得到的模型
def model():
    # 将整个训练过程集成到这个函数
    # 实现调用这个函数并传入相应的数据后直接得到dap模型
    # 返回的模型可以直接传入参数进行预测
    return

