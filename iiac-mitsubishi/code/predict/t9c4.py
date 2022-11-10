import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
T9 Case4 槽深欠佳
训练集： 294：12
验证集： 20： 3
测试集： 1～2

特征：
所有波峰count最小值
1-4号波峰 最小值count
1-4号波峰每个值
"""


def build_data(base_dir):
    train = pd.read_csv(base_dir + '/t9_fea.csv')
    test = pd.read_csv(base_dir + '/t9_fea_test.csv')
    print('Data build finished', train.shape, test.shape)
    return train, test


def run_train2(base_dir, seed=315):
    x_train, x_test = build_data(base_dir)
    y_train = x_train['T9_CASE4']

    x_train_true = x_train.loc[x_train['T9_CASE4'] == True]
    x_train_false = x_train.loc[x_train['T9_CASE4'] == False]
    print(x_train_true['max_min'].mean(), x_train_false['max_min'].mean())

    x_test['T9_CASE4'] = False
    x_test.sort_values(['max_min'], ascending=False, inplace=True)
    x_test.iloc[:2]['T9_CASE4'] = True
    x_test[['SignalFileName', 'max_min', 'T9_CASE4']].to_csv('../../data/result/t9c4_res.csv', index=False)


if __name__ == '__main__':
    base_dir = '../../data/feature'
    run_train2(base_dir)

