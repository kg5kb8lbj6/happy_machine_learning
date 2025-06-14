#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2025/06/14 10:08:01
@Author  :   Liu ZhongFei
@Version :   python3
@Contact :   1658422730@qq.com
'''
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.datasets import make_classification

def generate_classification_data():
    features, lables = make_classification(
        n_samples = 1000,
        n_features = 2,
        n_redundant = 0,
        random_state = 1,
        n_clusters_per_class = 2)
    rng = np.random.RandomState(42)
    # Add some noise to the features
    features += 2 * rng.uniform(size = features.shape)
    lables = lables.reshape(-1, 1)
    train_data, train_labels = features[:int(features.shape[0] * 0.8)], lables[:int(features.shape[0] * 0.8)]
    test_data, test_labels = features[int(features.shape[0] * 0.8):], lables[int(features.shape[0] * 0.8):]
    assert train_data.shape[0] == train_labels.shape[0] and test_data.shape[0] == test_labels.shape[0]
    return train_data, train_labels, test_data, test_labels,features, lables

def plot_decision_boundary(features, lables):
    lables = [int(x) for x in lables.flatten()]
    unique_labels = set(lables)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k,col in zip(unique_labels, colors):
        x_k=features[lables==k]
        plt.plot(x_k[:,0],x_k[:,1],'o',markerfacecolor=col,markeredgecolor="k",markersize=14)
    plt.title('Simulated binary data set')
    plt.show()
def init_params(dim):
    w = np.zeros((dim, 1))
    b = 0
    return w, b
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic(x, y, w, b):
    train_num = x.shape[0]

    model_output = sigmoid(np.dot(x, w) + b)
    cost = - 1/train_num * np.sum(y * np.log(model_output) + (1- y) * np.log(1 - model_output))
    dw = np.dot( x.T, (model_output - y)) / train_num
    db = np.sum(model_output - y) / train_num
    # 压缩损失数组维度
    cost = np.squeeze(cost)
    return cost, dw, db


def logistic_train(data, labels, learning_rate=0.01, num_iterations=1000):
    w, b = init_params(data.shape[1])
    cost_list = []
    for i in tqdm(range(num_iterations), desc = "Training Logistic Regression"):
        cost, dw, db = logistic(data, labels, w, b)
        w -= learning_rate * dw
        b -= learning_rate * db
        cost_list.append(cost) 
    params = {
        'weights': w,
        'bias': b
    }
    grads = {
        'dw': dw,
        'db': db
    }
    return params, grads, cost_list
    
def predict(data, params):
    pred_y = sigmoid(np.dot(data, params['weights']) + params['bias'])
    for i in range(len(pred_y)):
        if pred_y[i] >=0.5:
            pred_y[i] = 1
        else:
            pred_y[i] = 0
    return pred_y.reshape(-1, 1)

def accuracy(y_test, y_pred):
    correct_count = 0
    for i in range(len(y_test)):
        for j in range(len(y_pred)):
            if y_test[i] == y_pred[j] and i == j:
                correct_count +=1
            
    accuracy_score = correct_count / len(y_test)
    return accuracy_score


if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels,features, lables = generate_classification_data()
    print("Train data shape:", train_data.shape)
    print("Train labels shape:", train_labels.shape)
    print("Test data shape:", test_data.shape)
    print("Test labels shape:", test_labels.shape)
    # plot_decision_boundary(features, lables)
