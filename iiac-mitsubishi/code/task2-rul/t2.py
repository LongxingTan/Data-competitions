
import sys
sys.path.insert(0, '../ML-tools')

import numpy as np
import pandas as pd
from TabularTool.trainer import Trainer
from TabularTool.val2 import CV
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score, roc_auc_score, mean_squared_error
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold


def build_data(base_dir):
    train = pd.read_csv(base_dir + '/t2_fea.csv')
    test = pd.read_csv(base_dir + '/t2_fea_test.csv')
    print('Data build finished', train.shape, test.shape)

    label = pd.read_csv('../../data/1 训练用/Training set.csv')
    train['t2_rul'] = label['T2']
    return train, test


def run_train(base_dir, seed=315):
    lgb_params = {
        'objective': 'mse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.004,  # 设置0.1的时候，过拟合
        'n_estimators': 2500,
        'num_leaves': 2 ** 3 - 1,
        'min_data_in_leaf': 2 ** 3 - 1,
        'bagging_fraction': 0.7,
        'feature_fraction': 0.9,
        'seed': 2020,
    }

    fit_params = {
        'eval_metric': ['l2', 'rmse'],
        'verbose': 100,
        'early_stopping_rounds': 100,
    }

    x_train, x_test = build_data(base_dir)
    y_train = x_train['t2_rul']
    feature_cols = [i for i in list(x_train.columns) if i not in ['SignalFileName', 'T2_CASE1', 'T2_CASE2', 'T2_CASE3', 'T2_CASE4', 't2_rul']]

    data_split = KFold(n_splits=5, random_state=seed, shuffle=True)
    model = LGBMRegressor(**lgb_params)
    trainer = Trainer(model)

    trainer.train(x_train[feature_cols], y_train, categorical_feature=None, fit_params=fit_params)

    cv = CV(trainer, data_split)
    valid_oof, pred = cv.run(x_train, y_train, x_test=x_test, split_groups=None, feature_cols=feature_cols,
                             categorical_feature=None, fit_params=fit_params,
                             final_eval_metric=[mean_squared_error],
                             predict_method='predict')
    cv.get_feature_importance(columns=feature_cols, save=True, save_dir='../../data', plot=True)
    #
    # submit = pd.DataFrame()
    # submit['SignalFileName'] = x_test['SignalFileName']
    # submit['score'] = pred
    # submit.sort_values('score', ascending=False, ignore_index=True, inplace=True)
    # submit['T2_CASE1'] = False
    # submit.iloc[:6]['T2_CASE1'] = True
    # submit.to_csv('../../data/result/t2c1_res.csv', index=False)
    # print(np.mean(pred), len(pred))
    # return


def get_f1_score(y_true, y_pred):
    y_pred = [1 if i > 0.5 else 0 for i in y_pred]
    return f1_score(y_true, y_pred)


def get_precision_score(y_true, y_pred):
    y_pred = [1 if i > 0.5 else 0 for i in y_pred]
    return precision_score(y_true, y_pred)


def get_recall_score(y_true, y_pred):
    y_pred = [1 if i > 0.5 else 0 for i in y_pred]
    return recall_score(y_true, y_pred)


if __name__ == '__main__':
    base_dir = '../../data/feature'
    run_train(base_dir)
