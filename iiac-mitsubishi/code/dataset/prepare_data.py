import sys

sys.path.append("../feature")

import os

from filter_data import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stage import *

plt.style.use("ggplot")

# 进行数据的清洗


def prepare_train(base_dir="../../data/1 训练用", t2=False, t8=False, t9=False):
    files = sorted(os.listdir(base_dir + "/Signal Package"))
    labels = pd.read_csv(base_dir + "/Training set.csv", usecols=range(1, 29))
    data = []
    label = []

    for file in files:  # ['20210601-part_18.pkl']:
        file_name = "/Signal Package/{}".format(file)

        example = pd.read_pickle(base_dir + "/Signal Package/" + file)
        example["SignalFileName"] = file_name[1:]  # 无slash
        example_label = labels.loc[
            labels["SignalFileName"] == "Signal Package/{}".format(file)
        ]

        # 数据清洗(去除负载大于50， 去除停机）(第二种类型停机只有两个，手动修改)
        example = filter_data(example, plot=False)
        if file == "20210416-part_12.pkl":  # 训练
            example = example.iloc[
                list(range(4000)) + list(range(12500, example.shape[0]))
            ].reset_index(drop=True)
        if file == "20210517-part_18.pkl":  # 训练
            example = example.iloc[
                list(range(2000)) + list(range(16000, example.shape[0]))
            ].reset_index(drop=True)
        if file == "20210601-part_18.pkl":  # 训练
            example = example.iloc[
                list(range(2000)) + list(range(8000, example.shape[0]))
            ].reset_index(drop=True)
        if file == "20210528-part_9.pkl":  # 训练
            example = example.iloc[
                list(range(4500)) + list(range(8500, example.shape[0]))
            ].reset_index(drop=True)
        if file == "20210426-part_14.pkl":  # 验证
            example = example.iloc[
                list(range(4100)) + list(range(18800, example.shape[0]))
            ].reset_index(drop=True)
        if file == "20210514-part_18.pkl":  # 验证
            example = example.iloc[
                list(range(2000)) + list(range(10100, example.shape[0]))
            ].reset_index(drop=True)
        if file == "20210425-part_20.pkl":  # 测试
            example = example.iloc[
                list(range(4200)) + list(range(14000, example.shape[0]))
            ].reset_index(drop=True)

        # 得到T2、T8、T9刀的时间
        # Done @Yue: 后期T9的结束时间是不对的，调整参数或手动修正
        load = example["主轴负载"].values
        speed = example["主轴转速"].values
        if t2:
            start, end = find_t2_stage(load, speed)
            label.append(
                example_label[
                    [
                        "SignalFileName",
                        "T2_CASE0",
                        "T2_CASE1",
                        "T2_CASE2",
                        "T2_CASE3",
                        "T2_CASE4",
                    ]
                ]
            )
        if t8:
            print(file)
            start, end = find_t8_stage(load, speed)
            label.append(
                example_label[["SignalFileName", "T8_CASE0", "T8_CASE1", "T8_CASE2"]]
            )
        if t9:
            start, end = find_t9_stage(load, speed)
            label.append(
                example_label[
                    [
                        "SignalFileName",
                        "T9_CASE0",
                        "T9_CASE1",
                        "T9_CASE2",
                        "T9_CASE3",
                        "T9_CASE4",
                    ]
                ]
            )

        example_filter = example[start:end]
        data.append(example_filter)

    return data, label


def prepare_valid():
    files = sorted(os.listdir("../../data/2 测试用/Signal Package"))
    label = pd.read_csv("../../data/2 测试用/Online Test set.csv")
    return


def prepare_test(base_dir="../../data/3 正式赛题", t2=False, t8=False, t9=False):
    files = sorted(os.listdir(base_dir + "/Signal Package"))
    data = []

    for file in files:  # ['20210601-part_18.pkl']:
        file_name = "/Signal Package/{}".format(file)

        example = pd.read_pickle(base_dir + "/Signal Package/" + file)
        example["SignalFileName"] = file_name[1:]

        # 数据清洗(去除负载大于50， 去除停机）(第二种类型停机只有两个，手动修改)
        example = filter_data(example, plot=False)
        if file == "20210416-part_12.pkl":  # 训练
            example = example.iloc[
                list(range(4000)) + list(range(12500, example.shape[0]))
            ].reset_index(drop=True)
        if file == "20210517-part_18.pkl":  # 训练
            example = example.iloc[
                list(range(2000)) + list(range(16000, example.shape[0]))
            ].reset_index(drop=True)
        if file == "20210601-part_18.pkl":  # 训练
            example = example.iloc[
                list(range(2000)) + list(range(8000, example.shape[0]))
            ].reset_index(drop=True)
        if file == "20210528-part_9.pkl":  # 训练
            example = example.iloc[
                list(range(4500)) + list(range(8500, example.shape[0]))
            ].reset_index(drop=True)
        if file == "20210426-part_14.pkl":  # 验证
            example = example.iloc[
                list(range(4100)) + list(range(18800, example.shape[0]))
            ].reset_index(drop=True)
        if file == "20210514-part_18.pkl":  # 验证
            example = example.iloc[
                list(range(2000)) + list(range(10100, example.shape[0]))
            ].reset_index(drop=True)
        if file == "20210425-part_20.pkl":  # 测试
            example = example.iloc[
                list(range(4200)) + list(range(14000, example.shape[0]))
            ].reset_index(drop=True)

        # 得到T2、T8、T9刀的时间
        # Done @Yue: 后期T9的结束时间是不对的，调整参数或手动修正
        load = example["主轴负载"].values
        speed = example["主轴转速"].values
        if t2:
            start, end = find_t2_stage(load, speed)
        if t8:
            print(file)
            start, end = find_t8_stage(load, speed)
        if t9:
            start, end = find_t9_stage(load, speed)

        example_filter = example[start:end]
        data.append(example_filter)

    return data
