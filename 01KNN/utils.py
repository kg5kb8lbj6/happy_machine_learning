import numpy as np


def classify0(inX, dataSet, labels, k):
    """使用的是欧氏距离"""
    dataSetSize = dataSet.shape[0]
    # 将输入的向量inX在行方向上复制dataSetSize次（也就是和DatsSet的行数保持了一致）
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    # 进行平方，并在行方向上进行求和
    distance = (diffMat ** 2).sum(axis = 1)
    distance = distance ** 0.5
    # 对距离进行排序，返回的是索引值, 索引值是从0开始的
    distanceIndex = distance.argsort()
    classCount = {}
    for i in range(k):
        otherlabel = labels[distanceIndex[i]]
        classCount[otherlabel]  = classCount.get(otherlabel, 0) + 1
    # key = lambda  x: x[1]表示根据字典的值进行排序
    sortedClassCount = sorted(classCount.items(), key = lambda x: x[1], reverse = True)
    return sortedClassCount[0][0]