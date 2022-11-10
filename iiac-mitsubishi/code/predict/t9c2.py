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


def build_data2(base_dir):
    t9 = pd.read_csv(base_dir + '/t9.csv')
    t9c2_label = pd.read_csv(base_dir + '/1 训练用/Training set.csv', usecols=range(1, 29))
    t9c2_label = t9c2_label[['SignalFileName', 'T9_CASE2']]
    t9c2_label['SignalFileName'] = t9c2_label['SignalFileName'].apply(lambda x: '/' + x)
    # print(t9)
    # print(t9c4_label)

    t9c4 = t9.groupby(['SignalFileName', 'cat']).agg(count=('主轴负载', 'count')).reset_index()


    # t9c4.to_csv('t9c4.csv', index=False)

    # 由于1号峰和5号峰有可能在数据清理时不清楚
    data = t9c4.loc[t9c4['cat'].isin([1, 2, 3, 4])]
    data = data.set_index(['SignalFileName', 'cat'])
    data = data.unstack(level=-1).reset_index()
    data.columns = ['SignalFileName', '1', '2', '3', '4']
    data['min'] = data[['1', '2', '3', '4']].min(axis=1)
    data['max'] = data[['1', '2', '3', '4']].max(axis=1)
    data['mean'] = data[['1', '2', '3', '4']].mean(axis=1)
    data['mean_min'] = data['mean'] - data['min']
    data['max_min'] = data['max'] - data['min']

    data = data.merge(t9c4_label, on='SignalFileName', how='left').reset_index()

    print(data)
    sns.scatterplot(data=data, x="index", y="max_min", hue="T9_CASE4", sizes=5)
    plt.show()


def predict():
    test = pd.read_csv(base_dir + '/t9_test.csv')
    test_count = test.groupby(['SignalFileName', 'cat']).agg(count=('主轴负载', 'count')).reset_index()

    data = test_count.loc[test_count['cat'].isin([1, 2, 3, 4])]
    data = data.set_index(['SignalFileName', 'cat'])
    data = data.unstack(level=-1).reset_index()
    data.columns = ['SignalFileName', '1', '2', '3', '4']

    data['min'] = data[['1', '2', '3', '4']].min(axis=1)
    data['max'] = data[['1', '2', '3', '4']].max(axis=1)
    data['mean'] = data[['1', '2', '3', '4']].mean(axis=1)
    data['mean_min'] = data['mean'] - data['min']
    data['max_min'] = data['max'] - data['min']

    data.to_csv('t9c4_test.csv', index=False)
    sns.scatterplot(data=data, x="index", y="max_min", sizes=5)
    plt.show()
    print(data)


def run_train2(base_dir, seed=315):
    x_train, x_test = build_data(base_dir)
    y_train = x_train['T9_CASE2']

    x_train_true = x_train.loc[x_train['T9_CASE2'] == True]
    x_train_false = x_train.loc[x_train['T9_CASE2'] == False]
    print(x_train_true['mean_diff'].mean(), x_train_false['mean_diff'].mean())
    print(x_train_true['mean_min'].mean(), x_train_false['mean_min'].mean())

    x_test['T9_CASE2'] = False
    x_test.sort_values(['mean_diff'], ascending=False, inplace=True)
    x_test.iloc[:2]['T9_CASE2'] = True  # 特殊情况
    x_test[['SignalFileName', 'mean_diff', 'T9_CASE2']].to_csv('../../data/result/t9c2_res.csv', index=False)


if __name__ == '__main__':
    base_dir = '../../data/feature'
    run_train2(base_dir)

