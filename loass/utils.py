#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2025/06/18 20:27:07
@Author  :   Liu ZhongFei
@Version :   python3
@Contact :   1658422730@qq.com
'''
import numpy as np
from tqdm import tqdm
from sklearn.datasets import make_classification



def generate_classification_data():
    """Generate synthetic classification data for testing purposes."""
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


def initalize_params(n_features):
    weights = np.zeros((n_features, 1))
    bias = 0.0
    return weights, bias
# 定义符号函数
def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0
    

def l1_loss(features, labels, weights, bias, lambda_reg):
    
    num_train, num_features = features.shape
    y_hat = np.dot(features, weights) + bias
    loss =  np.sum((y_hat - labels)**2)/num_train + np.sum(lambda_reg * abs(weights))
    dw = np.dot(features.T, (y_hat - labels)) / num_train + lambda_reg * np.sign(weights)
    db = np.sum((y_hat - labels)) /num_train
    return y_hat, loss, dw, db

def loass_train(features, labels, learning_rate = 0.01, epochs = 1000):
    loss_list = []
    n_samples, n_features = features.shape
    weights, bias = initalize_params(n_features)
    for epoch in tqdm(range(epochs), desc="Training LOASS"):
        y_hat, loss, dw, db = l1_loss(features, labels, weights, bias, 0.1)
        weights += -learning_rate * dw
        bias += -learning_rate * db
        loss_list.append(loss)
        
        if epoch % 300 == 0:
            print('epoch %d loss %f' % (epoch, loss))
        params = {
            'w': weights,
            'b': bias
        }
        grads = {
            'dw': dw,
            'db': db
        }
    return loss, loss_list, params, grads

# 定义预测函数
def predict(X, params):
    w = params['w']
    b = params['b']
    
    y_pred = np.dot(X, w) + b
    return y_pred

if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels, features, lables = generate_classification_data()
    print("Train data shape:", train_data.shape)
    print("Train labels shape:", train_labels.shape)
    print("Test data shape:", test_data.shape)
    print("Test labels shape:", test_labels.shape)
    print("Features shape:", features.shape)
    print("Labels shape:", lables.shape)