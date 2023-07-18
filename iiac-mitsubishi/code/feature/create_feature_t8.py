# 想到一个特征，实现一个特征，检查一个特征
import sys

sys.path.append("../dataset")
from filter_data_stage import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_main_load(data, training=False):
    feature = []

    for id in np.unique(data["SignalFileName"]):
        temp = data.loc[data["SignalFileName"] == id]
        temp_median = []
        temp_40 = []
        temp_count = []

        scope = max(6, np.max(temp["cat"]))
        for cat in range(scope):
            try:
                temptemp = temp.loc[temp["cat"] == cat, ["主轴负载", "X负载"]]  # 注意两列
                temptemp = filter_t8_stage2(temptemp)
            except:
                print(temptemp)

            temp_count.append(len(temptemp))
            if len(temptemp) > 30:
                temp_median.append(np.median(temptemp))
                temp_40.append(np.quantile(temptemp, 0.4))

        if len(temp_median) > 4:
            diff_mean = np.mean(np.diff(temp_median))
            diff_median = np.median(np.diff(temp_median))
            diff_max = np.max(np.diff(temp_median))
            max_min = np.max(temp_median) - np.min(temp_median)
            diff_mean_40 = np.mean(np.diff(temp_40))
            diff01_abs = np.abs(temp_median[0] - temp_median[1])
            diff12_abs = np.abs(temp_median[1] - temp_median[2])
            diff23_abs = np.abs(temp_median[2] - temp_median[3])
            diff34_abs = np.abs(temp_median[3] - temp_median[4])
            min_count = np.min([i for i in temp_count if i > 10])
            min2_count = np.partition([i for i in temp_count if i > 10], -2)[-2]
            max_count = np.max(temp_count)
            var_count = np.var([i for i in temp_count if i > 10])

            feature.append(
                [
                    id,
                    diff_mean,
                    diff_median,
                    diff_max,
                    max_min,
                    diff_mean_40,
                    diff01_abs,
                    diff12_abs,
                    diff23_abs,
                    diff34_abs,
                    min_count,
                    min2_count,
                    max_count,
                    var_count,
                ]
            )

    feature = pd.DataFrame(
        np.array(feature),
        columns=[
            "SignalFileName",
            "diff_mean",
            "diff_median",
            "diff_max",
            "max_xin",
            "diff_mean_40",
            "diff01_abs",
            "diff12_abs",
            "diff23_abs",
            "diff34_abs",
            "min_count",
            "min2_count",
            "max_count",
            "var_count",
        ],
    )
    feature["large1"] = feature.loc[
        :, ["diff01_abs", "diff12_abs", "diff23_abs", "diff34_abs"]
    ].apply(lambda x: np.max(x), axis=1)
    feature["large2"] = feature.loc[
        :, ["diff01_abs", "diff12_abs", "diff23_abs", "diff34_abs"]
    ].apply(lambda x: np.partition(x, -2)[-2], axis=1)
    feature["large12_diff"] = feature["large1"].astype(float) - feature[
        "large2"
    ].astype(float)

    if training:
        t8_label = pd.read_csv(
            "../../data" + "/1 训练用/Training set.csv", usecols=range(1, 29)
        )
        t8_label = t8_label[["SignalFileName", "T8_CASE1", "T8_CASE2"]]
        # t8_label['SignalFileName'] = t8_label['SignalFileName'].apply(lambda x: '/' + x)
        feature = feature.merge(t8_label, on=["SignalFileName"], how="left")
        print(feature)
        feature.to_csv("../../data/feature/t8_fea.csv", index=False)
    else:
        feature.to_csv("../../data/feature/t8_fea_test.csv", index=False)


if __name__ == "__main__":
    base_dir = "../../data/user_data"
    t8 = pd.read_csv(base_dir + "/t8.csv")
    get_main_load(t8, training=True)

    t8_test = pd.read_csv(base_dir + "/t8_test.csv")
    get_main_load(t8_test, training=False)
