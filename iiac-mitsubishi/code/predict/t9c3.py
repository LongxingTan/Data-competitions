import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

"""
T9 Case3 负载过高
训练集： 298：8
验证集： 21： 2
测试集： 1～2
"""


def build_data(base_dir):
    train = pd.read_csv(base_dir + "/t9_fea.csv")
    test = pd.read_csv(base_dir + "/t9_fea_test.csv")
    print("Data build finished", train.shape, test.shape)
    return train, test


def build_data2(base_dir):
    t9 = pd.read_csv(base_dir + "/t9.csv")
    t9c3_label = pd.read_csv(base_dir + "/1 训练用/Training set.csv", usecols=range(1, 29))
    t9c3_label = t9c3_label[["SignalFileName", "T9_CASE3"]]
    t9c3_label["SignalFileName"] = t9c3_label["SignalFileName"].apply(lambda x: "/" + x)
    print(t9c3_label)

    t9_count = (
        t9.groupby(["SignalFileName", "cat"])
        .agg(count=("主轴负载", "count"), median=("主轴负载", "median"))
        .reset_index()
    )
    t9_count = t9_count.loc[(t9_count["count"] > 55) & (t9_count["median"] > 20)]

    t9c3 = (
        t9_count.groupby(["SignalFileName"])
        .agg(medianmean=("median", "mean"), median_max=("median", "max"))
        .reset_index()
    )
    t9c3 = t9c3.merge(t9c3_label, on="SignalFileName", how="left").reset_index()
    print(t9c3)

    # t9c3.to_csv('t9c3.csv', index=False)

    # plt.scatter(range(len(t9_count)), t9_count['median'])
    # plt.show()

    sns.scatterplot(data=t9c3, x="index", y="medianmean", hue="T9_CASE3", sizes=5)
    plt.show()


def run_train2(base_dir, seed=315):
    x_train, x_test = build_data(base_dir)
    y_train = x_train["T9_CASE3"]

    x_train_true = x_train.loc[x_train["T9_CASE3"] == True]
    x_train_false = x_train.loc[x_train["T9_CASE3"] == False]
    print(
        x_train_true["count_medianmean"].mean(),
        x_train_false["count_medianmean"].mean(),
    )

    x_test["T9_CASE3"] = False
    x_test.sort_values(["count_medianmean"], ascending=True, inplace=True)
    x_test.iloc[-1:]["T9_CASE3"] = True  # 特殊情况
    x_test[["SignalFileName", "count_medianmean", "T9_CASE3"]].to_csv(
        "../../data/result/t9c3_res.csv", index=False
    )


if __name__ == "__main__":
    base_dir = "../../data/feature"
    run_train2(base_dir)
