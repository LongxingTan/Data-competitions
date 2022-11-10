# 准备数据，将t2\t8\t9的数据单独保存
import sys
sys.path.append('../dataset')

import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prepare_data import *
from filter_data_stage import *
from substage import *


def test_train_t2(base_dir):
    data_t2, label_t2 = prepare_train(base_dir, t2=True)
    data_new = []

    for example, label in zip(data_t2, label_t2):
        example = filter_t2(example)
        example = filter_t2_stage1(example)
        example = find_t2_stage1(example)
        # example = clean_stage1(example)
        print(example.shape)  # 1. 检查形状是否符合要求

        # for col in np.unique(example['cat']):
        #     plt.plot(example.loc[example['cat'] == col, '主轴负载'])
        # plt.show()

        data_new.append(example)

    data_new = pd.concat(data_new, axis=0)
    data_new.to_csv('../../data/user_data/t2s1.csv', index=False)


def test_test_t2(base_dir):
    data_t2 = prepare_test(base_dir, t2=True)
    data_new = []

    for example in data_t2:
        example = filter_t2(example)
        example = filter_t2_stage1(example)
        example = find_t2_stage1(example)
        data_new.append(example)

    data_new = pd.concat(data_new, axis=0)
    data_new.to_csv('../../data/user_data/t2s1_test.csv', index=False)


def test_train_t8(base_dir):
    data_t8, label_t8 = prepare_train(base_dir, t8=True)
    data_new = []

    for example, label in zip(data_t8, label_t8):
        # example = filter_t8(example)
        # example = filter_t8_stage(example)
        example = find_t8_stage(example)
        # example = clean_stage1(example)
        print(example.shape)  # 1. 检查形状是否符合要求

        # for col in np.unique(example['cat']):
        #     plt.plot(example.loc[example['cat'] == col, '主轴负载'])
        # plt.show()

        data_new.append(example)

    data_new = pd.concat(data_new, axis=0)
    data_new.to_csv('../../data/user_data/t8.csv', index=False)


def test_test_t8(base_dir):
    data_t8 = prepare_test(base_dir, t8=True)
    data_new = []

    for example in data_t8:
        example = find_t8_stage(example)
        # example = clean_stage1(example)
        print(example.shape)  # 1. 检查形状是否符合要求

        # for col in np.unique(example['cat']):
        #     plt.plot(example.loc[example['cat'] == col, '主轴负载'])
        # plt.show()

        data_new.append(example)

    data_new = pd.concat(data_new, axis=0)
    data_new.to_csv('../../data/user_data/t8_test.csv', index=False)


def test_train_t9(base_dir):
    """ 测试训练集T9刀工作，顺便保存为中间数据
    :param base_dir:
    :return:
    """
    data_t9, label_t9 = prepare_train(base_dir, t9=True)
    data_new = []

    for example, label in zip(data_t9, label_t9):
        example = find_t9_stage(example)
        # example = clean_stage1(example)
        print(example.shape)  # 1. 检查形状是否符合要求

        # for col in np.unique(example['cat']):
        #     plt.plot(example.loc[example['cat'] == col, '主轴负载'])
        # plt.show()

        data_new.append(example)

    data_new = pd.concat(data_new, axis=0)
    data_new.to_csv('../../data/user_data/t9.csv', index=False)


def test_test_t9(base_dir):
    data_t9 = prepare_test(base_dir, t9=True)
    data_new = []

    for example in data_t9:
        example = find_t9_stage(example)
        # example = clean_stage1(example)
        print(example.shape)  # 1. 检查形状是否符合要求

        # for col in np.unique(example['cat']):
        #     plt.plot(example.loc[example['cat'] == col, '主轴负载'])
        # plt.show()

        data_new.append(example)

    data_new = pd.concat(data_new, axis=0)
    data_new.to_csv('../../data/user_data/t9_test.csv', index=False)


if __name__ == '__main__':
    base_dir = '../../data/1 训练用'
    test_train_t2(base_dir)
    test_train_t8(base_dir)
    test_train_t9(base_dir)

    base_dir = '../../data/3 正式赛题'
    test_test_t2(base_dir)
    test_test_t8(base_dir)
    test_test_t9(base_dir)
