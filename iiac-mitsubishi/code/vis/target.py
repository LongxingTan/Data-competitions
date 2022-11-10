
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


train_label = pd.read_csv('../../data/1 训练用/Training set.csv', usecols=range(1, 29))
valid_label = pd.read_csv('../../data/2 测试用/Online Test set.csv', usecols=range(1, 29))
pred_label = pd.read_csv('../../data/20210808.csv', usecols=range(1, 29))

for col in ['T2_CASE0', 'T2_CASE1', 'T2_CASE2', 'T2_CASE3', 'T2_CASE4', 'T8_CASE0', 'T8_CASE1', 'T8_CASE2', 'T9_CASE0', 'T9_CASE1', 'T9_CASE2', 'T9_CASE3', 'T9_CASE4', 'ProcessingResultIsOK']:
    print('\n', col)
    print(train_label[col].value_counts())
    print(valid_label[col].value_counts())
    print(pred_label[col].value_counts())

#
# print(valid_label)


