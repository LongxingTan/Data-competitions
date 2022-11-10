# 划分各刀自己的工序
import numpy as np
import pandas as pd
from scipy import stats
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
        if list[i+1] - list[i] <= 10:
            category.append(cat)
        else:
            cat += 1
            category.append(cat)
    category.append(category[-1])  # 最后增加一个同类型的
    print(Counter(category))
    return category


def find_t2_stage1(example):
    example.reset_index(drop=True, inplace=True)
    threshold = np.quantile(example['X负载'], 0.2)
    temp = example.loc[example['X负载'] > threshold]
    temp['cat'] = get_cate_t2(list(temp.index))
    temp = temp.reset_index(drop=True)
    return temp


def clean_t2_stage1(example):
    data = []
    for col in np.unique(example['cat']):
        l = np.quantile(example.loc[example['cat'] == col, '主轴负载'].values, 0.03)
        h = np.quantile(example.loc[example['cat'] == col, '主轴负载'].values, 0.96)
        temp = example.loc[(example['主轴负载'] > l) & (example['主轴负载'] < h)].reset_index(drop=True)
        data.append(temp)
    data = pd.concat(data, axis=0)
    return data.reset_index(drop=True)


def find_t2_stage2(example):
    example.reset_index(drop=True, inplace=True)


def get_cate_t8(list):
    """ 邻域
    根据index开始遍历，如果相邻的index不超过5，则归位一类
    如果超过5 则另起一类
    如果类别里的个数过少，则把这一类去掉；如果和相邻域很近，也可以归结到相邻域
    :param list:
    :return:
    """
    return


def find_t8_stage(example):
    example.reset_index(drop=True, inplace=True)
    temp = example.loc[example['主轴负载'] > 15]
    temp['cat'] = get_cate_t2(list(temp.index))
    temp = temp.reset_index(drop=True)
    return temp


def find_t9_stage(example):
    # 首先有波峰和波谷两个波，区分后把最大的异常去掉
    example.reset_index(drop=True, inplace=True)
    temp = example.loc[example['X负载'] > 2.9]
    temp['cat'] = get_cate_t2(list(temp.index))
    temp = temp.reset_index(drop=True)
    return temp
