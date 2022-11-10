
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


def get_cate_t2(list):
    """ 邻域
    根据index开始遍历，如果相邻的index不超过5，则归位一类
    如果超过5 则另起一类
    如果类别里的个数过少，则把这一类去掉；如果和相邻域很近，也可以归结到相邻域
    :param list:
    :return:
    """
    category = []
    cat = 0
    for i in range(len(list) - 1):
        if list[i+1] - list[i] <= 100:
            category.append(cat)
        else:
            cat += 1
            category.append(cat)
    category.append(category[-1])  # 最后增加一个同类型的
    print(len(category), Counter(category))
    return category


def split(example):
    example.reset_index(drop=True, inplace=True)
    temp = example.loc[example['主轴转速'] < -800]
    temp['cat'] = get_cate_t2(list(temp.index))
    temp = temp.reset_index(drop=True)
    return temp


if __name__ == '__main__':
    high_freq = pd.read_pickle('../../data/1 训练用/520-HighFreqDataSet.part1.pkl')
    high_freq = high_freq.loc[high_freq['主轴转速'] != 0]
    data = split(high_freq)
    data.to_csv('520-HighFreqDataSet.part1_split.csv', index=False)
    print(data)
