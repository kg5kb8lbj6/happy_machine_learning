#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   knn.py
@Time    :   2025/06/07 10:00:44
@Author  :   Liu ZhongFei
@Version :   python3
@Contact :   1658422730@qq.com
'''
import time
import numpy as np
from tqdm import tqdm

def get_date(filename):
    """
    Get the date from the filedata.
    param filename: the file path of the data
    output:list of data and labels
    """
    trainData = [];trainlabel = []
    fr = open(filename)
    #strip：去掉每行字符串首尾指定的字符（默认空格或换行符）
    #split：按照指定的字符将字符串切割成每个字段，返回列表形式
    for line in tqdm(fr.readlines(), desc="Reading data"):
        line = line.strip().split(',')
        trainData.append([int(x) for x in line[1:]])
        trainlabel.append(int(line[0]))
    return trainData, trainlabel

def get_distance(trainDataMat, trainlabel, testDataMat, k):
    distList = [0] * len(trainlabel)
    for i in range(len(trainDataMat)):
        curdist = np.sqrt(np.sum(np.square(trainDataMat[i] - testDataMat)))
        distList[i] = curdist
    topKIndex = np.argsort(np.array(distList))[:k]  # Get the indices of the k smallest distances
    labelList = [0] * 10  # Assuming labels are from 0 to 9
    for index in topKIndex:
        labelList[int(trainlabel[index])] += 1
    return labelList.index(max(labelList))


def knn(trainData, trainlabel, testData, testlable, k = 3):
    """
    KNN algorithm to classify the test data.
    param trainData: the training data
    param trainlabel: the labels of the training data
    param testData: the test data to be classified
    param k: the number of nearest neighbors to consider
    output: the predicted label for the test data
    """

    trainDataMat = np.matrix(trainData); trainlabel = np.matrix(trainlabel).T
    testDataMat = np.matrix(testData); testlable = np.matrix(testlable).T
    
    # error
    errorCount = 0.0
    for i in tqdm(range(len(testDataMat)), desc="Classifying test data"):
        y = get_distance(trainDataMat, trainlabel, testDataMat[i], k)
        if y != testlable[i]:
            errorCount += 1.0
    
    return 1 - (errorCount / len(testDataMat))


if __name__ == "__main__":
    trainData, trainlabel = get_date("knn/data/mnist_train.csv")
    testData, testlable = get_date("knn/data/mnist_test.csv")
    start_time = time.time()
    assert len(trainData) == len(trainlabel) and len(testData) == len(testlable)
    acc = knn(trainData, trainlabel, testData, testlable, k = 3)
    print(f"Accuracy: {acc * 100:.2f}%, Time taken: {time.time() - start_time:.2f} seconds")
