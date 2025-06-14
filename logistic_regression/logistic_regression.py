#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   logistic_regression.py
@Time    :   2025/06/14 09:55:41
@Author  :   Liu ZhongFei
@Version :   python3
@Contact :   1658422730@qq.com
'''
from sklearn.metrics import accuracy_score, classification_report
from utils import generate_classification_data, logistic_train, predict, accuracy

if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels, features, lables = generate_classification_data()
    params, grads, cost_list = logistic_train(train_data, train_labels, learning_rate=0.01, num_iterations=10000)
    y_pred =  predict(test_data,params)
    acc = accuracy(test_labels, y_pred)
    print(f"Accuracy: {acc:.2f}")
    cm = classification_report(test_labels, y_pred)
    print("classification_report")
    print(cm)