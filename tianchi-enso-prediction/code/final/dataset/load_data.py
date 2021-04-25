import sys
sys.path.append('.')  # dataset.read

import os
import gc
from copy import deepcopy
import numpy as np
import itertools
import pandas as pd
import xarray as xr
from .read_data import prepare_data, DataReader
np.random.seed(315)


def load_cmip(data=None, label=None, random_select=True, filter_1_99=False, random_gap=6, filter_scope=None):
    base_dir = '../user_data/'
    if data is None:  # 线下存储为pickle加载，线上端对端需要从nc开始运行
        data = pd.read_pickle(base_dir + 'CMIP_train.pickle')
    if label is None:
        label = pd.read_pickle(base_dir + 'CMIP_label.pickle')
 
    data.fillna(0, inplace=True)  # cmip has nan value, actually the land in cmip and soda is not same, but I don't find a good way to use this
    sst, t300, ua, va, month, label = prepare_data(data, label)    

    filter_out_idx_cmip6 = list(itertools.chain(* [range(151 * i * 12 - 35, 151 * i * 12) for i in range(1, 16)]))  # 每个模式最后35个，没有足够长度构建数据集
    filter_out_idx_cmip5 = list(itertools.chain(* [range(140 * i * 12 - 35 + 2265 * 12, 140 * i * 12 + 2265 * 12) for i in range(1, 18)]))
    idx = [i for i in range(len(label)) if i not in filter_out_idx_cmip5 + filter_out_idx_cmip6]

    if filter_1_99:
        if filter_scope is None:
            low = -1.5
            high = 2.6
        else:
            low = filter_scope[0]
            high = filter_scope[1]
        filter_idx = filter_0_99(label, low, high)  # 2.36/ 2.6
        #filter_idx = filter_0_99(label, -1.65, 2.36)
        idx = [i for i in idx if i not in filter_idx]
    
    data_reader = DataReader([sst, t300, ua, va, month, label], random_select=random_select, idx=idx, random_gap=random_gap) 
    return data_reader


def load_soda(data=None, label=None, random_select=True):
    base_dir = '../user_data/'
    if data is None:  # 线下存储为pickle加载，线上端对端需要从nc开始运行
        data = pd.read_pickle(base_dir + 'SODA_train.pickle')  # 转化， 第一维是年月，
    if label is None:
        label = pd.read_pickle(base_dir + 'SODA_label.pickle') 

    soda_sst, soda_t300, soda_ua, soda_va, soda_month, soda_label = prepare_data(data, label)    
   
    idx = range(len(label) - 35)
    soda_data_reader = DataReader([soda_sst, soda_t300, soda_ua, soda_va, soda_month, soda_label], random_select=random_select, idx=idx)
    return soda_data_reader


def filter_0_99(data, low_threshold, high_threshold):
    # 现在的用的threshold其实是，0.005 ～ 0.995 percentile, (label, -1.65, 2.36) 
    # 还有一个问题，就是是否允许预测值大于，现在只检查了前12个月，还有预测的24个月
    values = data  #.values[:, -1]
    filter_idx = []
    for i in range(len(data) - 12):
        temp = values[i:i+12]
        if np.sum(temp < low_threshold) + np.sum(temp > high_threshold) > 1:
            filter_idx.append(i)
    return filter_idx
