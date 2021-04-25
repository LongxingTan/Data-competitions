#! /usr/bin/env python
# coding=utf-8
# @Author: Longxing Tan, tanlongxing888@163.com
# Transfer the raw data (.nc) into the processed data(.pickle)
# label : 3 columns: year, month, nino3.4
# feature: columns: year, month, log, lat, SST, T300, Ua, Va
# 海表温度异常(SST)，热含量异常(T300)，纬向风异常（Ua），经向风异常（Va）

import pandas as pd
import numpy as np 
from netCDF4 import Dataset
import itertools
import gc


def prepare_2d_feature(data_dir, save=False):
    feature_nc = Dataset(data_dir, "r")

    features = []
    for variable in feature_nc.variables.values():
        feature = np.array(variable[:])
        features.append(feature)

    cols = list(itertools.product(features[4], features[5][:12], features[6], features[7]))
    data = pd.DataFrame(cols, columns=['year', 'month', 'lat', 'lon'])
    data['sst'] = features[0][:, :12, :, :].reshape(-1)
    data['t300'] = features[1][:, :12, :, :].reshape(-1)
    data['ua'] = features[2][:, :12, :, :].reshape(-1)
    data['va'] = features[3][:, :12, :, :].reshape(-1)

    if save:
        data.to_pickle('../user_data/' + data_dir.split('/')[-1].replace('.nc', '') + '.pickle')

    print('prepare 2d feature finished', data.shape)
    del feature_nc
    gc.collect()
    return data


def prepare_2d_label(label_dir, save=False):
    label_nc = Dataset(label_dir, "r")

    labels = []
    for variable in label_nc.variables.values():
        feature = np.array(variable[:, ])
        labels.append(feature)

    value = labels[0]
    index = labels[1]
    columns = labels[2]
    label = pd.DataFrame(value, index=index, columns=columns)

    label_long = label.iloc[:, :12]  # 宽数据转化为长数据，只取前12个月
    label_long = label_long.reset_index().rename(columns={'index': 'year'})
    label_long = pd.melt(label_long, id_vars=['year'], value_vars=[i for i in label_long.columns if i not in ['year']])
    label_long.rename(columns={'variable': 'month', 'value': 'nino'}, inplace=True)
    label_long.sort_values(by=['year', 'month'], ascending=True, inplace=True) 
    label_long.index = range(len(label_long))  

    if save:
        label_long.to_pickle('../user_data/' + label_dir.split('/')[-1].replace('.nc', '') + '.pickle')
    print('prepare 2d label finished', label_long.shape)
    del label_nc
    gc.collect()
    return label_long


if __name__ == '__main__':
    # cd code
    # python dataset/prepare_data.py
    base_dir='../tcdata'
    prepare_2d_feature(data_dir=base_dir + '/enso_round1_train_20210201/SODA_train.nc', save=True)
    prepare_2d_label(label_dir=base_dir + '/enso_round1_train_20210201/SODA_label.nc', save=True)
    prepare_2d_feature(data_dir=base_dir + '/enso_round1_train_20210201/CMIP_train.nc', save=True)
    prepare_2d_label(label_dir=base_dir + '/enso_round1_train_20210201/CMIP_label.nc', save=True)
    print('prepare done')
