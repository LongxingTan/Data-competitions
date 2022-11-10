# 测试 与 画图
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
T9特征：
所有波峰count最小值
1-4号波峰 最小值count
1-4号波峰每个值
x负载最大值\中位数。可能之前需要先去掉异常
diff mean, diff sum
"""


def build_data(base_dir):
    t9 = pd.read_csv(base_dir + '/user_data/t9.csv')

    t9_label = pd.read_csv(base_dir + '/1 训练用/Training set.csv', usecols=range(1, 29))
    t9_label = t9_label[['SignalFileName', 'T9_CASE1', 'T9_CASE2', 'T9_CASE3', 'T9_CASE4']]
    t9_label['SignalFileName'] = t9_label['SignalFileName'].apply(lambda x: '/' + x)
    # print(t9)
    # print(t9c4_label)

    t9c4 = t9.groupby(['SignalFileName', 'cat']).agg(count=('主轴负载', 'count')).reset_index()

    for id in np.unique(t9['SignalFileName']):
        temp = t9.loc[t9['SignalFileName'] == id]
        print(t9_label.loc[t9_label['SignalFileName'] == id].iloc[:, 1:])
        ax1 = plt.subplot(2, 1, 1)
        ax2 = plt.subplot(2, 1, 2)

        for cat in range(6):
            temptemp = temp.loc[temp['cat'] == cat, ['主轴负载', 'X负载']]
            temptemp = temptemp.iloc[30: -30]
            ax1.plot(temptemp['主轴负载'])
            ax2.plot(temptemp['X负载'])
        plt.show()

    # t9_agg = t9.groupby(['SignalFileName', 'cat']).agg(count=('主轴负载', 'mean')).reset_index()
    # print(t9_agg)


if __name__ == '__main__':
    base_dir = '../../data'
    build_data(base_dir)
