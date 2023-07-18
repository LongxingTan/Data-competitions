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


def run_train(base_dir, seed=315):
    lgb_params = {
        "objective": "binary",
        "learning_rate": 0.05,
        "n_estimators": 10000,
        "num_leaves": 2**3 - 1,
        "bagging_fraction": 0.75,
        "feature_fraction": 0.75,
        "seed": 2020,
    }

    fit_params = {
        "eval_metric": ["binary_error"],
        "verbose": 100,
        "early_stopping_rounds": 100,
    }

    x_train, x_test = build_data(base_dir)
    y_train = x_train["T8_CASE1"]
    feature_cols = [
        i
        for i in list(x_train.columns)
        if i not in ["SignalFileName", "T8_CASE1", "T8_CASE2"]
    ]

    data_split = KFold(n_splits=4, random_state=seed, shuffle=True)
    model = LGBMClassifier(**lgb_params)
    trainer = Trainer(model)

    trainer.train(
        x_train[feature_cols], y_train, categorical_feature=None, fit_params=fit_params
    )

    cv = CV(trainer, data_split)
    valid_oof, pred = cv.run(
        x_train,
        y_train,
        x_test=x_test,
        split_groups="label",
        feature_cols=feature_cols,
        categorical_feature=None,
        fit_params=fit_params,
        final_eval_metric=[
            get_precision_score,
            get_recall_score,
            get_f1_score,
            roc_auc_score,
        ],
        predict_method="predict_proba_positive",
    )
    cv.get_feature_importance(columns=feature_cols, save=True, save_dir="../../data")

    submit = pd.DataFrame()
    submit["SignalFileName"] = x_test["SignalFileName"]
    submit["T8_CASE1"] = pred
    submit.to_csv("../../data/t8c1_submit.csv", index=False)
    print(np.mean(pred), len(pred))
    return


def run_train2(base_dir, seed=315):
    x_train, x_test = build_data(base_dir)

    x_train_true = x_train.loc[x_train["T8_CASE1"] == True]
    x_train_false = x_train.loc[x_train["T8_CASE1"] == False]
    print(x_train_true["diff_mean"].mean(), x_train_false["diff_mean"].mean())

    x_test["T8_CASE1"] = False
    x_test.sort_values(["diff_mean"], ascending=False, inplace=True)
    x_test.iloc[:3]["T8_CASE1"] = True
    x_test[["SignalFileName", "diff_mean", "T8_CASE1"]].to_csv(
        "../../data/result/t8c1_res.csv", index=False
    )


def get_f1_score(y_true, y_pred):
    y_pred = [1 if i > 0.5 else 0 for i in y_pred]
    return f1_score(y_true, y_pred)


def get_precision_score(y_true, y_pred):
    y_pred = [1 if i > 0.5 else 0 for i in y_pred]
    return precision_score(y_true, y_pred)


def get_recall_score(y_true, y_pred):
    y_pred = [1 if i > 0.5 else 0 for i in y_pred]
    return recall_score(y_true, y_pred)


if __name__ == "__main__":
    base_dir = "../../data/feature"
    run_train2(base_dir)
