# 准备数据

import os
import numpy as np
import pandas as pd


def process_init(data):
    # 有两列是固定值，其实应该是列名
    data.rename(columns={'geography': 'geography_level_3', 'product': 'product_level_2'}, inplace=True)
    data.drop(['geography_level', 'product_level'], axis=1, inplace=True)
    return data


def concat_cat(data, geo_top, product_top):
    # 类别关系concat
    data = data.merge(geo_top, on=['geography_level_3'], how='left')
    data = data.merge(product_top, on=['product_level_2'], how='left')
    return data


def transform2demand(data):
    # train中qty为资源累计使用量
    data.sort_values(by=['unit', 'ts'], inplace=True)
    data['demand'] = data.groupby(['unit'])['qty'].diff()
    data.dropna(subset=['demand'], axis=0, inplace=True)
    return data


def add_date_info(data):
    data['weekday'] = data['ts'].dt.weekday
    return data


def prepare_data(base_dir):
    """ 转化为时序预测任务，不过任务需要自己定义.
    1. 由于每周一进行预测，转化为每周一预测未来两周demand的预测任务
    """
    demand_train = pd.read_csv(base_dir + '/demand_train.csv')

    demand_train['ts'] = pd.to_datetime(demand_train['ts'], format='%Y-%m-%d')
    demand_train_A = process_init(demand_train)  # 更改列名

    geo_top = pd.read_csv(os.path.join(base_dir, 'geo_topo_round2.csv'))
    product_top = pd.read_csv(os.path.join(base_dir, 'product_topo_round2.csv'))
    demand_train_A = concat_cat(demand_train_A, geo_top, product_top)  # 增加类别
    demand_train_A = transform2demand(demand_train_A)

    demand_train_A = add_date_info(demand_train_A)

    print(demand_train_A)
    os.makedirs('../user_data', exist_ok=True)
    demand_train_A.to_csv('../user_data/demand.csv', index=False)


if __name__ == '__main__':
    base_dir = '../tcdata'
    prepare_data(base_dir)
