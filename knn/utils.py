#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2025/06/07 15:37:44
@Author  :   Liu ZhongFei
@Version :   python3
@Contact :   1658422730@qq.com
'''
import numpy as np
from tqdm import tqdm


def get_date(filename):
    """
    Get the date from the filedata.
    param filename: the file path of the data
    output:list of data and labels
    """
    trainData = [];trainlabel = []
    fr = open(filename)
    #strip：去掉每行字符串首尾指定的字符（默认空格或换行符）
    #split：按照指定的字符将字符串切割成每个字段，返回列表形式
    for line in tqdm(fr.readlines(), desc="Reading data"):
        line = line.strip().split(',')
        trainData.append([int(x) for x in line[1:]])
        trainlabel.append(int(line[0]))
    return trainData, trainlabel

