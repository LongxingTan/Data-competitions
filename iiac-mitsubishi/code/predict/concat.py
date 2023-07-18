import os

import numpy as np
import pandas as pd

base = pd.read_csv("../../data/3 正式赛题/Background Test set v2.csv", usecols=range(15))
ori_columns = base.columns

base_dir = "../../data/result"
for f in sorted(os.listdir(base_dir)):
    if f.endswith("csv"):
        res_t = pd.read_csv(os.path.join(base_dir, f))
        base = base.merge(res_t.iloc[:, [0, -1]], on=["SignalFileName"], how="left")

new_columns = base.columns
target_columns = [i for i in new_columns if i not in ori_columns]
base[target_columns] = base[target_columns].astype(int)
base["T2_CASE0"] = base[["T2_CASE1", "T2_CASE2", "T2_CASE3", "T2_CASE4"]].apply(
    np.sum, axis=1
)
base["T8_CASE0"] = base[["T8_CASE1", "T8_CASE2"]].apply(np.sum, axis=1)
base["T9_CASE0"] = base[["T9_CASE1", "T9_CASE2", "T9_CASE3", "T9_CASE4"]].apply(
    np.sum, axis=1
)
base["ProcessingResultIsOK"] = True
base.loc[
    base["SignalFileName"] == "Signal Package/20210421-part_8.pkl",
    "ProcessingResultIsOK",
] = False

base = base[
    list(ori_columns)
    + [
        "T2_CASE0",
        "T2_CASE1",
        "T2_CASE2",
        "T2_CASE3",
        "T2_CASE4",
        "T8_CASE0",
        "T8_CASE1",
        "T8_CASE2",
        "T9_CASE0",
        "T9_CASE1",
        "T9_CASE2",
        "T9_CASE3",
        "T9_CASE4",
        "ProcessingResultIsOK",
    ]
]
base[list(target_columns) + ["T2_CASE0", "T8_CASE0", "T9_CASE0"]] = base[
    list(target_columns) + ["T2_CASE0", "T8_CASE0", "T9_CASE0"]
].astype(bool)
base["T2_CASE0"] = base["T2_CASE0"].apply(lambda x: not x)
base["T8_CASE0"] = base["T8_CASE0"].apply(lambda x: not x)
base["T9_CASE0"] = base["T9_CASE0"].apply(lambda x: not x)
base.to_csv("../../data/submit.csv", index=False)
