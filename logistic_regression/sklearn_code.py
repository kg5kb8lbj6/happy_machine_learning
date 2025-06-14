#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   sklearn_code.py
@Time    :   2025/06/14 14:13:11
@Author  :   Liu ZhongFei
@Version :   python3
@Contact :   1658422730@qq.com
'''
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from utils import generate_classification_data

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    train_data, train_labels, test_data, test_labels, features, lables = generate_classification_data()
    clf = LogisticRegression(random_state = 42, max_iter = 10000).fit(train_data, train_labels)
    y_pred = clf.predict(test_data)
    acc = accuracy_score(test_labels, y_pred)
    print(f"Accuracy: {acc:.2f}")
    cm = classification_report(test_labels, y_pred)
    print("classification_report")
    print(cm)