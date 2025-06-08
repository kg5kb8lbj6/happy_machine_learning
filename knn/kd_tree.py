#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   kd_tree.py
@Time    :   2025/06/08 07:46:10
@Author  :   Liu ZhongFei
@Version :   python3
@Contact :   1658422730@qq.com
'''

import numpy as np
from collections import deque
class KDTreeNode:
    def __init__(self, point, left=None, right=None):
        """
        KD树节点类
        :param point: 当前节点表示的点（一维数组）
        :param left: 左子树
        :param right: 右子树
        """
        self.point = point   # 当前节点的点坐标
        self.left = left     # 左子树
        self.right = right   # 右子树

class KDTree:
    def __init__(self, data):
        """
        构建KD树
        :param data: 输入数据，形状为 (n_samples, n_features)
        """
        self.n_features = data.shape[1]  # 特征维度
        self.root = self.build_tree(data)  # 构建根节点

    def build_tree(self, data, depth=0):
        """
        递归构建KD树
        :param data: 当前子集数据
        :param depth: 当前深度（用于选择分割维度）
        :return: 根节点
        """
        if len(data) == 0:
            return None

        # 选择分割维度（轮换使用各维度）
        axis = depth % self.n_features

        # 按当前维度排序数据
        data_sorted = data[data[:, axis].argsort()]

        # 选择中位数作为当前节点
        mid = len(data_sorted) // 2
        median_point = data_sorted[mid]

        # 递归构建左右子树
        left = self.build_tree(data_sorted[:mid], depth + 1)
        right = self.build_tree(data_sorted[mid + 1:], depth + 1)

        return KDTreeNode(median_point, left, right)

    def nearest_neighbors(self, target, k=1):
        """
        寻找最近的k个邻居
        :param target: 目标点
        :param k: 近邻数量
        :return: 最近的k个点列表
        """
        # 使用优先队列保存最近的k个点（按距离排序）
        neighbors = []
        self._search(self.root, target, k, 0, neighbors)
        return [point for dist, point in neighbors]

    def _search(self, node, target, k, depth, neighbors):
        """
        递归搜索最近邻
        :param node: 当前节点
        :param target: 目标点
        :param k: 近邻数量
        :param depth: 当前深度
        :param neighbors: 当前最近邻列表
        """
        if node is None:
            return

        axis = depth % self.n_features
        dist = self._distance(node.point, target)

        # 将当前节点加入邻居列表（如果未满k个）
        if len(neighbors) < k:
            neighbors.append((dist, node.point))
            neighbors.sort()  # 按距离排序
        else:
            if dist < neighbors[-1][0]:
                neighbors[-1] = (dist, node.point)
                neighbors.sort()

        # 选择进入左子树还是右子树
        if target[axis] < node.point[axis]:
            self._search(node.left, target, k, depth + 1, neighbors)
        else:
            self._search(node.right, target, k, depth + 1, neighbors)

        # 检查是否需要回溯搜索另一侧
        if abs(target[axis] - node.point[axis]) < (neighbors[-1][0] if neighbors else float('inf')):
            if target[axis] < node.point[axis]:
                self._search(node.right, target, k, depth + 1, neighbors)
            else:
                self._search(node.left, target, k, depth + 1, neighbors)

    def _distance(self, a, b):
        """
        计算两点之间的欧氏距离（平方）
        :param a: 点a
        :param b: 点b
        :return: 距离平方
        """
        return np.sum((a - b) ** 2)

# 示例使用
if __name__ == "__main__":
    # 生成随机二维数据
    data = np.random.rand(100, 2)
    kd_tree = KDTree(data)

    # 查找最近的3个邻居
    target = np.array([0.5, 0.5])
    neighbors = kd_tree.nearest_neighbors(target, k=3)
    print("最近的3个邻居：", neighbors)