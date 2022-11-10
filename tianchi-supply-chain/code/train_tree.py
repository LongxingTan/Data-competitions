# 训练模型, 树模型

import os
import numpy as np
import pandas as pd
import argparse
from lightgbm import LGBMRegressor, LGBMClassifier
from tree.feature import *
from tree.trainer import *
from dataset.prepare_data import *


# ======================================data============================================

def build_data(base_dir, debug=False, online=False):
    demand = pd.read_csv('../user_data/demand.csv')
    demand, feature_cols = get_feature(demand)

    # 目标
    demand = add_target(demand)
    neg_threshold = np.quantile(demand['target'], 0.005)
    print('threshold', neg_threshold)
    demand = demand.loc[demand['target'] >= neg_threshold]

    # 权重
    weight = pd.read_csv(os.path.join(base_dir, 'weight.csv'))
    demand = demand.merge(weight[['unit', 'weight']], on='unit', how='left')
    demand['weight'] = np.sqrt(demand['weight'])

    # 类别变量
    # demand['unit'] = demand['unit'].astype(int)

    if online:
        x_train = demand[feature_cols]
        y_train = demand['target']
        weight_train = demand['weight'].tolist()
        print(x_train.shape, y_train.shape)
        return x_train, y_train, weight_train
    else:
        split_date = '2021-01-13'
        x_train = demand.loc[(demand['ts'] < split_date) & (demand['weekday'] == 0), feature_cols]
        y_train = demand.loc[(demand['ts'] < split_date) & (demand['weekday'] == 0), 'target']
        weight_train = demand.loc[(demand['ts'] < split_date) & (demand['weekday'] == 0), 'weight'].tolist()

        x_valid = demand.loc[(demand['ts'] >= split_date) & (demand['weekday'] == 0), feature_cols]
        y_valid = demand.loc[(demand['ts'] >= split_date) & (demand['weekday'] == 0), 'target']
        weight_valid = demand.loc[(demand['ts'] >= split_date) & (demand['weekday'] == 0), 'weight'].tolist()
        valid = demand.loc[(demand['ts'] >= split_date) & (demand['weekday'] == 0)]
        print(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape)
        return x_train, y_train, x_valid, y_valid, valid, weight_train, weight_valid


# ======================================target============================================

def add_target(demand):
    demand['target'] = demand.groupby(['unit'])['demand'].transform(
        lambda x: x.shift(-21).rolling(14).sum(skipna=True))  # 未来第2周到第3周的需求
    demand.dropna(subset=['target'], inplace=True)
    return demand


# ======================================feature============================================

def get_feature(data, demand_col='demand'):
    feature_cols = []

    # lag roll feature
    data = get_lag_rolling_feature(data, roll_columns=[demand_col], lags=[0, 7, 14, 21], period=7, id_col='unit',
                                   agg_funs=['max', 'sum', 'std'], feature_cols=feature_cols, prefix='demand')

    return data, feature_cols


# ======================================train============================================

def run_train(base_dir, debug=False, online=False):
    if online:
        x_train, y_train, weight_train = build_data(base_dir, debug=debug, online=online)
        x_valid, y_valid, weight_valid = None, None, None
    else:
        x_train, y_train, x_valid, y_valid, valid, weight_train, weight_valid = build_data(base_dir, debug=debug,
                                                                                           online=online)

    lgb_params = {
        'objective': 'mae',
        'boosting_type': 'gbdt',
        'learning_rate': 0.004,
        'n_estimators': 4000,
        'num_leaves': 2 ** 5 - 1,
        'min_data_in_leaf': 2 ** 3 - 1,
        'bagging_fraction': 0.8,
        'feature_fraction': 0.8,
        'seed': seed,
    }

    fit_params = {
        'eval_metric': ['l1'],
        # 'sample_weight': weight_train,
        # 'eval_sample_weight': [weight_valid],
        'verbose': 100,
        # 'early_stopping_rounds': 100,
    }

    model = LGBMRegressor(**lgb_params)
    trainer = Trainer(model)

    trainer.train(x_train, y_train, x_valid, y_valid, categorical_feature=None, fit_params=fit_params,
                  importance_method='auto')
    trainer.save_model(model_dir='./lgb.pkl')

    if not online:
        trainer.get_feature_importance(columns=x_train.columns, save=True,
                                       save_dir='../user_data/feature_importance.csv')

        y_valid_pred = trainer.predict(x_valid)
        valid['pred'] = y_valid_pred
        valid.to_csv('../user_data/valid_prediction.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--online', help="True/False", default=False, type=lambda x: (str(x).lower() == 'true'))
    args = parser.parse_args()

    seed = 315
    base_dir = '../tcdata'
    debug = True
    run_train(base_dir, debug, online=args.online)
