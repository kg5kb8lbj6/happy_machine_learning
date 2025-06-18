#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   loass.py
@Time    :   2025/06/18 20:28:39
@Author  :   Liu ZhongFei
@Version :   python3
@Contact :   1658422730@qq.com
'''
from sklearn.metrics import r2_score

from utils import generate_classification_data, loass_train, predict
import matplotlib.pyplot as plt


if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels, features, lables = generate_classification_data()
    loss, loss_list, params, grads = loass_train(train_data, train_labels, 0.01, 3000)
    y_pred = predict(test_data, params)
    print(r2_score(test_labels, y_pred))
    f = test_data.dot(params['w']) + params['b']
    plt.scatter(range(test_data.shape[0]), test_labels)
    plt.plot(f, color = 'darkorange')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()