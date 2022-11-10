# 准备数据
import sys
sys.path.append('../feature')

import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prepare_data import *
from filter_data_stage import *
from substage import *


def test_train_t2(base_dir):
    data_t2, label_t2 = prepare_train(base_dir, t2=True)

    for example, label in zip(data_t2, label_t2):
        example = filter_t2(example)
        example = filter_t2_stage1(example)
        example = find_t2_stage1(example)
        print(example.shape)  # 1. 检查形状是否符合要求
        print(label.iloc[:, 1:])
        print('\n')

        plt.figure(figsize=(12, 5))
        plt.subplot(221)
        plt.plot(example['主轴负载'], alpha=0.9)
        plt.subplot(222)
        plt.plot(example['主轴转速'])
        plt.subplot(223)
        plt.plot(example['X负载'])
        plt.subplot(224)
        # plt.plot(example['Z负载'])
        plt.hist(example['X负载'], bins=50)

        plt.show()


def test_train_t8(base_dir):
    data_t8, label_t8 = prepare_train(base_dir, t8=True)

    for example, label in zip(data_t8, label_t8):
        # example = filter_t8(example)
        # example = filter_t8_stage(example)
        example = find_t8_stage(example)
        print(example.shape)  # 1. 检查形状是否符合要求
        print(label.iloc[:, 1:])
        print('\n')

        plt.figure(figsize=(12, 5))
        plt.subplot(221)
        plt.plot(example['主轴负载'], alpha=0.9)
        plt.subplot(222)
        plt.plot(example['主轴转速'])
        plt.subplot(223)
        plt.plot(example['X负载'])
        plt.subplot(224)
        plt.plot(example['Z负载'])

        plt.show()


def test_train_t9(base_dir):
    data_t9, label_t9 = prepare_train(base_dir, t9=True)

    for example, label in zip(data_t9, label_t9):
        # example = filter_t8(example)
        # example = filter_t2_stage1(example)
        # example = find_t2_stage1(example)
        print(example.shape)  # 1. 检查形状是否符合要求
        print(label.iloc[:, 1:])
        print('\n')

        plt.figure(figsize=(12, 5))
        plt.subplot(221)
        plt.plot(example['主轴负载'], alpha=0.9)
        plt.subplot(222)
        plt.plot(example['主轴转速'])
        plt.subplot(223)
        plt.plot(example['X负载'])
        plt.subplot(224)
        plt.plot(example['Z负载'])

        plt.show()


if __name__ == '__main__':
    base_dir = '../../data/1 训练用'
    # test_train_t2(base_dir)
    # test_train_t8(base_dir)
    test_train_t9(base_dir)
