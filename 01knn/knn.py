#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   knn.py
@Time    :   2025/10/06 08:50:15
@Author  :   Liu ZhongFei
@Version :   python3
@Contact :   1658422730@qq.com
'''

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load dataset
def get_data():
    data = load_iris()
    x,y = shuffle(data.data, data.target, random_state = 42)
    x = x.astype(np.float32)
    x_trin,x_test,y_train,y_test = train_test_split(x , y, test_size = 0.3, random_state = 42, shuffle = True)
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
    return x_trin, y_train, x_test, y_test

def train_and_evaluate(x_train, y_train, x_test, y_test, n_neighbors=3):
    # Initialize KNN classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    # Train the model
    knn.fit(x_train, y_train)
    
    # Make predictions
    y_pred = knn.predict(x_test)
    
    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")



if __name__ == "__main__":
    x_train, y_train, x_test, y_test = get_data()
    train_and_evaluate(x_train, y_train, x_test, y_test)


