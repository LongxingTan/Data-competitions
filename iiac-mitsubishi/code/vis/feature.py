import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("ggplot")


base_dir = "../../data/feature"
x = pd.read_csv(base_dir + "/t9_fea.csv")

target_col = "T9_CASE4"
feature_col = "max_min"
x_true = x.loc[x[target_col] == True]
x_false = x.loc[x[target_col] == False]
print(x_true[feature_col].mean(), x_false[feature_col].mean())

res = pd.DataFrame()
res["type"] = ["True", "False"]
res["value"] = [x_true[feature_col].mean(), x_false[feature_col].mean()]

print(res)

plt.bar(x=res["type"], height=res["value"])
plt.show()
