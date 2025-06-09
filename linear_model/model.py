#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2025/06/09 20:00:19
@Author  :   Liu ZhongFei
@Version :   python3
@Contact :   1658422730@qq.com
'''
import numpy as np
from tqdm import tqdm
from utils import load_data, init_params, linear_loss
import matplotlib.pyplot as plt



class LinearModel:
    def __init__(self, learning_rate = 0.01 , n_iterations = 10000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    def fit(self, x, y):
        his_loss, epochs = [], []
        # w, b都是参数
        w, b  = init_params(x.shape[1]) 
        for i in tqdm(range(self.n_iterations), desc = "train linear model"):
            y_hat, loss, dw, db = linear_loss(x, y, w, b)
            w += -self.learning_rate * dw
            b += -self.learning_rate * db
            his_loss.append(loss)
            epochs.append(i)
            param = {'w': w, 'b': b}
            grad = {"dw": dw, "db": db}
        return his_loss, epochs, param, grad
if __name__ == "__main__":
    train_data, train_target, test_data, test_target = load_data()
    model = LinearModel(learning_rate = 0.01, n_iterations = 10000)
    his_loss, epochs, param, grad = model.fit(train_data, train_target)
    print(his_loss[:20])