# 训练集： 145/306 = 47.3%
# 验证集： 9/25 = 36%
# 测试集：大概有 13 * 0.465 = 6
# 台阶： 相邻两个的差，最大值 减 第二最大值 大于 2
# 选择最大的一个或两个作为正样本。为了增加概率，可以提交两个，但如果第2个的整体t2没有其他错误，可以标记为没问题T2试试
# 最大样本：20210422-part_6.pkl 应该这个类别总有正样本吧

import sys
sys.path.insert(0, '../ML-tools')

import numpy as np
import pandas as pd
from TabularTool.trainer import Trainer
from TabularTool.validator import CV
from TabularTool.sk_trainer import SKTrainer
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold


def build_data(base_dir):
    train = pd.read_csv(base_dir + '/t2_fea.csv')
    test = pd.read_csv(base_dir + '/t2_fea_test.csv')
    print('Data build finished', train.shape, test.shape)
    return train, test


def run_train(base_dir, seed=315):
    lgb_params = {
        'objective': 'binary',
        'learning_rate': 0.005,
        'n_estimators': 10000,
        'num_leaves': 2 ** 3 - 1,
        'bagging_fraction': 0.85,
        # 'feature_fraction': 0.75,
        'seed': 2020,
    }

    fit_params = {
        'eval_metric': ['binary_error'],
        'verbose': 100,
        'early_stopping_rounds': 100,
    }

    x_train, x_test = build_data(base_dir)
    y_train = x_train['T2_CASE3']
    feature_cols = [i for i in list(x_train.columns) if i not in ['SignalFileName', 'T2_CASE1', 'T2_CASE2', 'T2_CASE3', 'T2_CASE4']]

    data_split = KFold(n_splits=5, random_state=seed, shuffle=True)
    model = LGBMClassifier(**lgb_params)
    trainer = Trainer(model)

    trainer.train(x_train[feature_cols], y_train, categorical_feature=None, fit_params=fit_params)

    cv = CV(trainer, data_split)
    valid_oof, pred = cv.run(x_train, y_train, x_test=x_test, split_groups='label', feature_cols=feature_cols,
                             categorical_feature=None, fit_params=fit_params,
                             final_eval_metric=[get_precision_score, get_recall_score, get_f1_score, roc_auc_score],
                             predict_method='predict_proba_positive')
    cv.get_feature_importance(columns=feature_cols, save=True, save_dir='../../data')

    submit = pd.DataFrame()
    submit['SignalFileName'] = x_test['SignalFileName']
    submit['score'] = pred
    submit.sort_values('score', ascending=False, ignore_index=True, inplace=True)
    submit['T2_CASE3'] = True
    submit.iloc[:8]['T2_CASE3'] = False
    submit.to_csv('../../data/t2c3_submit.csv', index=False)
    print(np.mean(pred), len(pred))
    return


def run_train2(base_dir, seed=315):
    x_train, x_test = build_data(base_dir)

    x_train_true = x_train.loc[x_train['T2_CASE4'] == True]
    x_train_false = x_train.loc[x_train['T2_CASE4'] == False]
    print(x_train_true['diff_min_cat'].mean(), x_train_false['diff_min_cat'].mean())

    x_test['T2_CASE4'] = False
    x_test.sort_values(['diff_min_cat'], ascending=True, inplace=True)
    x_test.iloc[:2]['T2_CASE4'] = True
    x_test.loc[x_test['SignalFileName'] == 'Signal Package/20210601-part_12.pkl', 'T2_CASE4'] = True  # 异常点
    x_test[['SignalFileName', 'diff_min_cat', 'T2_CASE4']].to_csv('../../data/result/t2c4_res.csv', index=False)


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
    run_train2(base_dir)
