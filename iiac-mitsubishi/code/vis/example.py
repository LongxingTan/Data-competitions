import sys

sys.path.append("../feature")
sys.path.append("../dataset")

import os

from filter_data import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stage import *

plt.style.use("ggplot")


def vis_example(file, label=None, plot=False):
    # 增加一些划分
    try:
        example = pd.read_pickle("../../data/1 训练用/Signal Package/" + file)
    except:
        # example = pd.read_pickle('../../data/2 测试用/Signal Package/' + file)
        example = pd.read_pickle("../../data/3 正式赛题/Signal Package/" + file)
    example = filter_data(example, plot=False)

    if file == "20210416-part_12.pkl":  # 训练
        example = example.iloc[
            list(range(4000)) + list(range(12500, example.shape[0]))
        ].reset_index()
    if file == "20210517-part_18.pkl":  # 训练
        example = example.iloc[
            list(range(2000)) + list(range(16000, example.shape[0]))
        ].reset_index()
    if file == "20210601-part_18.pkl":  # 训练
        example = example.iloc[
            list(range(2000)) + list(range(8000, example.shape[0]))
        ].reset_index()
    if file == "20210528-part_9.pkl":  # 训练
        example = example.iloc[
            list(range(4500)) + list(range(8500, example.shape[0]))
        ].reset_index()
    if file == "20210426-part_14.pkl":  # 验证
        example = example.iloc[
            list(range(4100)) + list(range(18800, example.shape[0]))
        ].reset_index()
    if file == "20210514-part_18.pkl":  # 验证
        example = example.iloc[
            list(range(2000)) + list(range(10100, example.shape[0]))
        ].reset_index()
    if file == "20210425-part_20.pkl":  # 测试
        example = example.iloc[
            list(range(4200)) + list(range(14000, example.shape[0]))
        ].reset_index()

    print(file, ":")
    load = example["主轴负载"].values
    speed = example["主轴转速"].values
    t2_start, t2_end = find_t2_stage(load, speed)
    # t11_start, t11_end = find_t11_stage(load, speed)
    t8_start, t8_end = find_t8_stage(load, speed)
    t9_start, t9_end = find_t9_stage(load, speed)

    # 标签
    if label is not None:
        print(label[["T2_CASE0", "T2_CASE1", "T2_CASE2", "T2_CASE3", "T2_CASE4"]])
        print(label[["T8_CASE0", "T8_CASE1", "T8_CASE2"]])
        print(label[["T9_CASE0", "T9_CASE1", "T9_CASE2", "T9_CASE3", "T9_CASE4"]])
        print("\n")

    if plot:
        # if label['T9_CASE3'].values[0]:
        if True:
            plt.figure(figsize=(12, 5))
            plt.subplot(221)
            plt.plot(example["主轴负载"], alpha=0.9)
            plt.scatter(t2_start, example["主轴负载"][t2_start], s=20, color="green")
            plt.scatter(t2_end, example["主轴负载"][t2_end], s=20, color="green")
            # plt.scatter(t11_start, example['主轴负载'][t11_start], s=20, color='green')
            # plt.scatter(t11_end, example['主轴负载'][t11_end], s=20, color='green')
            plt.scatter(t8_start, example["主轴负载"][t8_start], s=20, color="black")
            plt.scatter(t8_end, example["主轴负载"][t8_end], s=20, color="black")
            plt.scatter(t9_start, example["主轴负载"][t9_start], s=20, color="green")
            plt.scatter(t9_end, example["主轴负载"][t9_end], s=20, color="green")

            plt.subplot(222)
            plt.plot(example["主轴转速"])
            plt.subplot(223)
            plt.plot(example["X负载"])
            plt.subplot(224)
            plt.plot(example["Z负载"])

            plt.show()


if __name__ == "__main__":
    # files = ['20210421-part_8.pkl']
    # files = sorted(os.listdir('../../data/1 训练用/Signal Package'))
    # label = pd.read_csv('../../data/1 训练用/Training set.csv', usecols=range(1, 29))

    # files = sorted(os.listdir('../../data/2 测试用/Signal Package'))
    # label = pd.read_csv('../../data/2 测试用/Online Test set.csv')

    files = sorted(os.listdir("../../data/3 正式赛题/Signal Package"))
    label = None

    for file in files:
        if label is not None:
            example_label = label.loc[
                label["SignalFileName"] == "Signal Package/{}".format(file)
            ]
        else:
            example_label = None
        vis_example(file, example_label, plot=True)
