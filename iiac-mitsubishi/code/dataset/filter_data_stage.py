# 划分substage时去除前后因为划分不准确导致数据
from datetime import date
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("ggplot")


def filter_t2(example):
    """T2刀
    首先起点选择X负载小于2的作为起点

    :param example:
    :return:
    """
    start = np.min(example[example["X负载"] < 1.9].index)
    return example.iloc[start:].reset_index(drop=True)


def filter_t2_stage1(example):
    """
    T2:
    首先根据转速，把前6个波峰和最后波峰进行区分: 转速大于-135的最小值
    然后根据X负载在前6个波段区分开来，注意阈值可能要动态选取

    其实，也有的是5段，和7段
    """
    end = np.min(example[example["主轴转速"] > -140].index)
    temp = example.iloc[:end]
    # print(example.shape, temp.shape)
    # end2 = np.max(temp.loc[temp['X负载'] < 1.9].index)
    return temp.reset_index(drop=True)  # .iloc[:end2]


def filter_t8_stage2(example):
    p1 = np.argmax(example["主轴负载"])
    temp = example[p1:]
    l1 = np.argmin(temp["主轴负载"])
    temp2 = temp[l1:]
    return temp2[25:]
