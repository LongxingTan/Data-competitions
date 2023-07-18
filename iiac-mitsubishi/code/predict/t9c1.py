# T9C1
# 训练集： 145/306 = 47.3%
# 验证集： 9/25 = 36%
# 测试集：大概有 13 * 0.465 = 6
# BugFixed: 直接扔进模型，概率大的反而是负样本，不知道为啥。分特别低的原因

import sys

sys.path.insert(0, "../ML-tools")

from TabularTool.trainer import Trainer
from TabularTool.validator import CV
from lightgbm import LGBMClassifier, LGBMRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold


def build_data(base_dir):
    train = pd.read_csv(base_dir + "/t9_fea.csv")
    test = pd.read_csv(base_dir + "/t9_fea_test.csv")
    print("Data build finished", train.shape, test.shape)
    return train, test


def run_train2(base_dir, seed=315):
    x_train, x_test = build_data(base_dir)
    y_train = x_train["T9_CASE1"]

    x_train_true = x_train.loc[x_train["T9_CASE1"] == True]
    x_train_false = x_train.loc[x_train["T9_CASE1"] == False]
    print(x_train_true["mean_diff"].mean(), x_train_false["mean_diff"].mean())

    x_test["T9_CASE1"] = False
    x_test.sort_values(["mean_diff"], ascending=False, inplace=True)
    x_test.iloc[:2]["T9_CASE1"] = True  # 特殊情况
    x_test[["SignalFileName", "mean_diff", "T9_CASE1"]].to_csv(
        "../../data/result/t9c1_res.csv", index=False
    )


if __name__ == "__main__":
    base_dir = "../../data/feature"
    run_train2(base_dir)
