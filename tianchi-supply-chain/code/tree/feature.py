 
import numpy as np
import pandas as pd
from datetime import datetime
from multiprocessing import Process, Manager, Pool


def get_global_feature(data, id_col, target_col, agg_funs=['mean'], feature_cols=[], prefix=None):
    for agg_fun in agg_funs:
        if prefix is None:
            prefix = target_col
        feature_col = 'global_' + prefix + '_' + str(agg_fun)
        feature_cols.append(feature_col)
        data[feature_col] = data.groupby(id_col)[target_col].transform(agg_fun)
    return data


def get_lag_feature(data, lag_columns, lags, id_col, feature_cols=[]):
    for col in lag_columns:
        for lag in lags:
            feature_col = col + '_lag{}'.format(lag)
            feature_cols.append(feature_col)
            data[feature_col] = data.groupby(id_col)[col].shift(lag)
    return data


def get_rolling_feature(data, roll_columns, periods, id_col, agg_funs=['mean'], feature_cols=[], prefix=None):
    for col in roll_columns:
        for period in periods:
            for agg_fun in agg_funs:
                if prefix is None:
                    prefix = col
                feature_col = prefix + '_roll{}_'.format(period) + str(agg_fun)
                feature_cols.append(feature_col)
                data[feature_col] = data.groupby(id_col)[col].transform(lambda x: x.rolling(period).agg(agg_fun))
    return data


def get_lag_rolling_feature(data, roll_columns, lags, period, id_col, agg_funs=['mean'], feature_cols=[], prefix=None):
    # this is used to get the lag rolling, like the last week mean
    for col in roll_columns:
        for lag in lags:
            for agg_fun in agg_funs:
                if prefix is None:
                    prefix = col
                feature_col = prefix + '_lag{}_roll{}_by_{}'.format(lag, period, id_col) + str(agg_fun)
                feature_cols.append(feature_col)
                data[feature_col] = data.groupby(id_col)[col].transform \
                    (lambda x: x.shift(lag).rolling(period).agg(agg_fun))
    return data
