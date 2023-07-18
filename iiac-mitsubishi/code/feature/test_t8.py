# 测试 与 画图
import sys

sys.path.append("../dataset")

import os

from filter_data_stage import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

"""

"""


def build_data(base_dir):
    t8 = pd.read_csv(base_dir + "/user_data/t8.csv")

    t8_label = pd.read_csv(base_dir + "/1 训练用/Training set.csv", usecols=range(1, 29))
    t8_label = t8_label[["SignalFileName", "T8_CASE1", "T8_CASE2"]]
    t8_label["SignalFileName"] = t8_label["SignalFileName"].apply(lambda x: "/" + x)

    # for id in np.unique(t8['SignalFileName']):
    for id in [
        "Signal Package/" + i
        for i in [
            "20210409-part_6.pkl",
            "20210419-part_1.pkl",
            "20210514-part_7.pkl",
            "20210517-part_16.pkl",
            "20210517-part_15.pkl",
            "20210517-part_5.pkl",
            "20210601-part_17.pkl",
            "20210601-part_9.pkl",
        ]
    ]:
        temp = t8.loc[t8["SignalFileName"] == id]

        print(t8_label.loc[t8_label["SignalFileName"] == id].iloc[:, 1:])
        ax1 = plt.subplot(2, 1, 1)
        ax2 = plt.subplot(2, 1, 2)

        for cat in range(6):
            temptemp = temp.loc[temp["cat"] == cat, ["主轴负载", "X负载"]]
            temptemp = filter_t8_stage2(temptemp)
            # temptemp = temptemp.iloc[30: -30]
            ax1.plot(temptemp["主轴负载"])
            ax2.plot(temptemp["X负载"])
        plt.show()


if __name__ == "__main__":
    base_dir = "../../data"
    build_data(base_dir)