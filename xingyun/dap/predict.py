# -*- coding: UTF-8 -*-

def attr_predict(test_features, svm_clfs):
    """
    """
    attr = []
    # 先用训练好的svm得到图片对应的属性向量
    for attr_index in range(0, 85):
        svm_for_current_attr = svm_clfs[attr_index]
        # 用SVM预测出一批图片是否有当前属性
        attr_predict_result = svm_for_current_attr.predict(test_features)

        attr.append(attr_predict_result)
    
    attr = np.array(attr, dtype=int)
    attr = attr.T

    return attr

def cls_predict(test_attr, bayes_clf):
    """
    """
    # 再用bayes和得到的属性向量预测类别
    predict_result = bayes_clf.predict(test_attr)

    return predict_result

def predict(test_data, model):
    """
    """
    svm_clfs = model['svm_clfs']
    bayes_clf = model['bayes_clf']

    attr_result = attr_predict(test_data, svm_clfs)
    cls_result = cls_predict(attr_result, bayes_clf)

    return attr_result, cls_result