#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2025/06/09 20:46:14
@Author  :   Liu ZhongFei
@Version :   python3
@Contact :   1658422730@qq.com
'''
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.utils import shuffle

def load_data():
    """
        load data
    """
    diabetes = load_diabetes()
    data, target  = shuffle(diabetes.data, diabetes.target, random_state=42)
    train_data, train_target = data[:int(len(data) * 0.8)], target[:int(len(target) * 0.8)]
    test_data, test_target = data[int(len(data) * 0.8):], target[int(len(target) * 0.8):]
    train_target = train_target.reshape(-1, 1)
    test_target = test_target.reshape(-1, 1)
    assert train_data.shape[0] == train_target.shape[0] and test_data.shape[0] == test_target.shape[0]
    return train_data, train_target, test_data, test_target


def init_params(dim):
    """
        初始化参数
    """
    w = np.zeros((dim, 1))
    b = 0
    return w, b

def linear_loss(data, target, w, b):
    n_sample = data.shape[0]
    y_hat = np.dot(data, w) + b
    loss = np.sum((y_hat - target) ** 2) / n_sample
    dw = np.dot(data.T, (y_hat - target))
    db = np.sum((y_hat - target)) / n_sample
    return y_hat, loss, dw, db


def linear_predict(data, params):
    """
        线性预测
    """
    w, b = params['w'], params['b']
    y_pred = np.dot(data, w) + b
    return y_pred





if __name__ == "__main__":
    train_data, train_target, test_data, test_target = load_data()
    w, b = init_params(train_data.shape[1])
    # y_hat, loss, dw, db = linear_loss(train_data, train_target, w, b)
    print("Train data shape:", train_data.shape)
    print("Train target shape:", train_target.shape)
    print("Test data shape:", test_data.shape)
    print("Test target shape:", test_target.shape)
    print(f"params w shape:{w.shape}")