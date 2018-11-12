# -*- coding: UTF-8 -*-

from sklearn.metrics import roc_auc_score, accuracy_score

def evaluate(test_attr_true, test_attr_predict, test_cls_true, test_cls_predic):
    """
    """
    # 计算每个支持向量机的AUC，以及平均值
    overall_auc_score = 0
    num_of_svm = test_attr_true.shape[1]
    print("\n--------------------")
    for attr_index in range(num_of_svm):
        x = test_attr_true[:][attr_index]
        x_predict = test_attr_predict[:][attr_index]
        auc = metrics.roc_auc_score(x, x_predict)
        overall_auc_score += auc
        print("AUC score for " + str(attr_index+1) + "th SVM: " + str(auc))
    print("--------------------\n")
    print("\n====================")
    print("Average AUC score on test data: " + str(overall_auc_score/num_of_svm))
    print("====================\n")

    accuracy = metrics.accuracy_score(test_cls_true, test_cls_predic)
    print("Accuracy: " + str(accuracy * 100) + "%\n\n")
    
    return
