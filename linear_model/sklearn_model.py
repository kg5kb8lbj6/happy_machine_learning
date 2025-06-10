#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   knn_model.py
@Time    :   2025/06/10 19:34:39
@Author  :   Liu ZhongFei
@Version :   python3
@Contact :   1658422730@qq.com
'''
from utils import load_data
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

train_data, train_target, test_data, test_target = load_data()
linearmodel = linear_model.LinearRegression()
linearmodel.fit(train_data, train_target)
y_pred = linearmodel.predict(test_data)
print(f"Mean Squared Error: {mean_squared_error(test_target, y_pred)}")
print(f"R^2 Score: {r2_score(test_target, y_pred)}")