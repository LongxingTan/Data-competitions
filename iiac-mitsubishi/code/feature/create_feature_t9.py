# 想到一个特征，实现一个特征，检查一个特征
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_xload(data, training=False):
    data_count = data.groupby(['SignalFileName', 'cat']).agg(count=('主轴负载', 'count')).reset_index()
    data_count = data_count.loc[data_count['cat'].isin([1, 2, 3, 4])]
    data_count = data_count.set_index(['SignalFileName', 'cat'])
    data_count_unstack = data_count.unstack(level=-1).reset_index()
    data_count_unstack.columns = ['SignalFileName', '1', '2', '3', '4']

    data_count_unstack['min'] = data_count_unstack[['1', '2', '3', '4']].min(axis=1)
    data_count_unstack['max'] = data_count_unstack[['1', '2', '3', '4']].max(axis=1)
    data_count_unstack['mean'] = data_count_unstack[['1', '2', '3', '4']].mean(axis=1)
    data_count_unstack['mean_min'] = data_count_unstack['mean'] - data_count_unstack['min']
    data_count_unstack['max_min'] = data_count_unstack['max'] - data_count_unstack['min']

    # filter，去掉前30与后30，
    data_filter = data.groupby(['SignalFileName', 'cat'])['X负载'].apply(lambda x: x[30:-30]).reset_index()
    data1 = data_filter.groupby(['SignalFileName', 'cat']).agg(max_x_load=('X负载', 'max'), median_x_load=(('X负载', 'median'))).reset_index()

    # diff 平均值
    data2 = data1.groupby(['SignalFileName'])['max_x_load'].apply(lambda x: np.mean(np.diff(x))).reset_index()
    data2.columns = ['SignalFileName', 'mean_diff']

    data22 = data1.groupby(['SignalFileName'])['max_x_load'].apply(lambda x: np.sum(np.diff(x))).reset_index()
    data22.columns = ['SignalFileName', 'sum_diff']
    data2 = data2.merge(data22, on=['SignalFileName'], how='left')

    # # 最大值与最小值之差
    data22 = data1.groupby(['SignalFileName'])['max_x_load'].apply(lambda x: np.max(x) - np.min(x))

    data22.columns = ['SignalFileName', 'max_min']
    data2 = data2.merge(data22, on=['SignalFileName'], how='left')

    # count
    data_count = data.groupby(['SignalFileName', 'cat']).agg(count=('主轴负载', 'count'), median=('主轴负载', 'median')).reset_index()
    data_count = data_count.loc[(data_count['count'] > 55) & (data_count['median'] > 20)]

    data23 = data_count.groupby(['SignalFileName']).agg(count_medianmean=('median', 'mean'), count_median_max=('median', 'max')).reset_index()
    data2 = data2.merge(data23, on=['SignalFileName'], how='left')

    data2 = data2.merge(data_count_unstack, on=['SignalFileName'], how='left')

    if training:
        t9_label = pd.read_csv('../../data' + '/1 训练用/Training set.csv', usecols=range(1, 29))
        t9_label = t9_label[['SignalFileName', 'T9_CASE1', 'T9_CASE2', 'T9_CASE3', 'T9_CASE4']]
        # t9_label['SignalFileName'] = t9_label['SignalFileName'].apply(lambda x: '/' + x)

        data2 = data2.merge(t9_label, on=['SignalFileName'], how='left')
        print(data2)
        data2.to_csv('../../data/feature/t9_fea.csv', index=False)
    else:
        data2.to_csv('../../data/feature/t9_fea_test.csv', index=False)


if __name__ == '__main__':
    base_dir = '../../data/user_data'
    t9 = pd.read_csv(base_dir + '/t9.csv')
    get_xload(t9, training=True)

    t9_test = pd.read_csv(base_dir + '/t9_test.csv')
    get_xload(t9_test, training=False)
