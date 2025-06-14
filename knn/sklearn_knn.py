#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   sklearn_knn.py
@Time    :   2025/06/07 15:38:33
@Author  :   Liu ZhongFei
@Version :   python3
@Contact :   1658422730@qq.com
'''
import time
import numpy as np
from logistic_regression.utils import get_date
from sklearn.neighbors import KNeighborsClassifier

def model(trainData, trainlabel, testData, testlable, k = 3):
    trainData = np.array(trainData); trainlabel = np.array(trainlabel).reshape(-1, 1)  # Reshape to 2D array
    testData = np.array(testData); testlable = np.array(testlable).reshape(-1, 1)  # Reshape to 2D array
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(trainData, trainlabel.ravel())  # ravel() to convert to 1D array
    predictions = knn.predict(testData)
    errorCount = np.sum(predictions != testlable.ravel())
    accuracy = (len(testlable) - errorCount) / len(testlable)
    return accuracy


if __name__ == "__main__":
    trainData, trainlabel = get_date("knn/data/mnist_train.csv")
    testData, testlable = get_date("knn/data/mnist_test.csv")
    time_start = time.time()
    acc = model(trainData, trainlabel, testData, testlable, k = 3)
    print(f"Accuracy: {acc * 100:.2f}%, Time taken: {time.time() - time_start:.2f} seconds")