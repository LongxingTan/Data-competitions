# 想到一个特征，实现一个特征，检查一个特征
import sys
sys.path.append('../dataset')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from filter_data_stage import *


def get_main_load(data, training=False):
    feature = []
    for id in np.unique(data['SignalFileName']):
        temp = data.loc[data['SignalFileName'] == id]
        temp_median_main = []
        temp_35_main = []
        temp_median_x = []
        temp_min_main = []
        temp_min_mean = []
        temp_min2_main = []
        temp_main_var = []

        scope = max(6, np.max(temp['cat']))
        for cat in range(scope):
            try:
                temptemp = temp.loc[temp['cat'] == cat, ['主轴负载', 'X负载']]
                q1 = np.quantile(temptemp['主轴负载'], 0.03)
                q2 = np.quantile(temptemp['主轴负载'], 0.97)
                temptemp2 = temptemp.loc[(temptemp['主轴负载'] > q1) & (temptemp['主轴负载'] < q2)]

            except:
                print(temptemp)

            if (len(temptemp2) > 30) & (len(temptemp) > 30):  # 注意两列
                temp_median_main.append(np.median(temptemp['主轴负载']))
                temp_35_main.append(np.quantile(temptemp['主轴负载'], 0.35))
                temp_median_x.append(np.median(temptemp['X负载']))
                temp_min_main.append(np.min(temptemp['主轴负载']))
                temp_min2_main.append(np.quantile(temptemp2['主轴负载'], 0.03))
                temp_min_mean.append(np.median(temptemp['主轴负载']) - np.min(temptemp['主轴负载']))
                temp_main_var.append(np.var(temptemp2['主轴负载']))

        if len(temp_median_main) > 4:
            diff_mean = np.mean(np.diff(temp_median_main))
            diff_abs_mean = np.mean(np.abs(np.diff(temp_median_main)))
            diff_median = np.median(np.diff(temp_median_main))
            diff_max = np.max(np.diff(temp_median_main))
            max_min = np.max(temp_median_main) - np.min(temp_median_main)
            diff_mean_35 = np.mean(np.diff(temp_35_main))
            diff_01_abs = np.abs(temp_median_main[0] - temp_median_main[1])
            diff_12_abs = np.abs(temp_median_main[1] - temp_median_main[2])
            diff_23_abs = np.abs(temp_median_main[2] - temp_median_main[3])
            diff_34_abs = np.abs(temp_median_main[3] - temp_median_main[4])
            diff_02_abs = np.abs(temp_median_main[0] - temp_median_main[2])
            diff_03_abs = np.abs(temp_median_main[0] - temp_median_main[3])
            diff_04_abs = np.abs(temp_median_main[0] - temp_median_main[4])
            diff_13_abs = np.abs(temp_median_main[1] - temp_median_main[3])
            diff_15_abs = np.abs(temp_median_main[1] - temp_median_main[-1])
            diff_15 = temp_median_main[1] - temp_median_main[-1]
            diff_23 = temp_median_main[2] - temp_median_main[3]
            diff_12_45 = np.mean(temp_median_main[:2]) - np.mean(temp_median_main[-2:])
            min_min_cat = np.min(temp_min_main)
            max_min_cat = np.max(temp_min_mean)
            diff_min_cat = np.max(temp_min_mean) - np.min(temp_min_main)
            abs_mean_mean_abs = diff_abs_mean - np.abs(diff_mean)
            min_min2_main = np.min(temp_min2_main)
            max_main_var = np.max(temp_main_var)

            feature.append([id, diff_mean, diff_abs_mean, diff_median, diff_max, max_min, diff_mean_35, diff_01_abs, diff_12_abs, diff_23_abs, diff_34_abs, diff_02_abs, diff_03_abs, diff_04_abs, diff_13_abs, diff_15_abs, diff_15, diff_23, diff_12_45, min_min_cat, max_min_cat, diff_min_cat, abs_mean_mean_abs, min_min2_main, max_main_var])

    feature = pd.DataFrame(np.array(feature), columns=['SignalFileName', 'diff_mean', 'diff_abs_mean', 'diff_median', 'diff_max', 'max_xin', 'diff_mean_40', 'diff01_abs', 'diff12_abs', 'diff23_abs', 'diff34_abs', 'diff02_abs', 'diff03_abs', 'diff04_abs', 'diff13_abs', 'diff15_abs', 'diff_15', 'diff_23', 'diff_12_45', 'min_min_cat', 'max_min_cat', 'diff_min_cat', 'abs_mean_mean_abs', 'min_min2_main', 'max_main_var'])
    feature['large1'] = feature.loc[:, ['diff01_abs', 'diff12_abs', 'diff23_abs', 'diff34_abs']].apply(lambda x: np.max(x), axis=1)
    feature['large2'] = feature.loc[:, ['diff01_abs', 'diff12_abs', 'diff23_abs', 'diff34_abs']].apply(lambda x: np.partition(x, -2)[-2], axis=1)
    feature['large12_diff'] = feature['large1'].astype(float) - feature['large2'].astype(float)

    if training:
        t2_label = pd.read_csv('../../data' + '/1 训练用/Training set.csv', usecols=range(1, 29))
        t2_label = t2_label[['SignalFileName', 'T2_CASE1', 'T2_CASE2', 'T2_CASE3', 'T2_CASE4']]
        # t2_label['SignalFileName'] = t2_label['SignalFileName'].apply(lambda x: '/' + x)
        feature = feature.merge(t2_label, on=['SignalFileName'], how='left')
        print(feature)
        feature.to_csv('../../data/feature/t2_fea.csv', index=False)
    else:
        feature.to_csv('../../data/feature/t2_fea_test.csv', index=False)


if __name__ == '__main__':
    base_dir = '../../data/user_data'
    t2 = pd.read_csv(base_dir + '/t2s1.csv')
    get_main_load(t2, training=True)

    t2_test = pd.read_csv(base_dir + '/t2s1_test.csv')
    get_main_load(t2_test, training=False)
