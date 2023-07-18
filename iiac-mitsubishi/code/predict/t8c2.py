"""
类似台阶, t2c3
同理取最大值作为判断依据

"""

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
    train = pd.read_csv(base_dir + "/t8_fea.csv")
    test = pd.read_csv(base_dir + "/t8_fea_test.csv")
    print("Data build finished", train.shape, test.shape)
    return train, test


def run_train2(base_dir, seed=315):
    x_train, x_test = build_data(base_dir)
    y_train = x_train["T8_CASE2"]

    x_train_true = x_train.loc[x_train["T8_CASE2"] == True]
    x_train_false = x_train.loc[x_train["T8_CASE2"] == False]
    print(x_train_true["min_count"].mean(), x_train_false["min_count"].mean())

    x_test["T8_CASE2"] = False
    x_test.sort_values(["min_count"], ascending=True, inplace=True)
    x_test.iloc[:1]["T8_CASE2"] = True
    x_test[["SignalFileName", "min_count", "T8_CASE2"]].to_csv(
        "../../data/result/t8c2_res.csv", index=False
    )


if __name__ == "__main__":
    base_dir = "../../data/feature"
    run_train2(base_dir)
