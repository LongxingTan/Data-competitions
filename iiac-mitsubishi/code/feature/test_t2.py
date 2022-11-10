# 测试 与 画图
import sys
sys.path.append('../dataset')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from filter_data_stage import *

"""

"""


def build_data(base_dir):
    t2 = pd.read_csv(base_dir + '/user_data/t2_s1.csv')

    t2_label = pd.read_csv(base_dir + '/1 训练用/Training set.csv', usecols=range(1, 29))
    t2_label = t2_label[['SignalFileName', 'T2_CASE1', 'T2_CASE2', 'T2_CASE3', 'T2_CASE4']]
    t2_label['SignalFileName'] = t2_label['SignalFileName'].apply(lambda x: '/' + x)

    # for id in np.unique(t2['SignalFileName']):
    for id in ['Signal Package/' + i for i in ['20210409-part_6.pkl']]:
        temp = t2.loc[t2['SignalFileName'] == id]
        print(t2_label.loc[t2_label['SignalFileName'] == id].iloc[:, 1:])
        ax1 = plt.subplot(2, 1, 1)
        ax2 = plt.subplot(2, 1, 2)

        for cat in range(6):
            temptemp = temp.loc[temp['cat'] == cat, ['主轴负载', 'X负载']]
            temptemp = temptemp.iloc[40:-45]
            ax1.plot(temptemp['主轴负载'])
            ax2.plot(temptemp['X负载'])
        plt.show()


if __name__ == '__main__':
    base_dir = '../../data'
    build_data(base_dir)
