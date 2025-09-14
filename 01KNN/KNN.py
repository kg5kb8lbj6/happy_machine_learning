import numpy as np
from utils import classify0, file2matrix



def createDataSet():
    group = np.array(
        [[1.0, 1.1],
        [1.0, 1.0],
        [0, 0],
        [0, 0.1]]
    )
    labels = ['A', 'A', 'B', 'B']
    return group, labels





if __name__ == "__main__":
    # group, labels = createDataSet()
    # test = classify0([1,1],group, labels, 3)
    # print(group.shape)
    # print(test)
    
    ## 使用 k-近邻算法改进约会网站的配对效果
    path = '01KNN/data/datingTestSet.txt'
    test = file2matrix(path)
    print(test)