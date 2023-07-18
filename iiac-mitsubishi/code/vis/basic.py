import sys

sys.path.append("../feature")
sys.path.append("../dataset")

import os

from filter_data import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def explore_data(base_dir):
    data = []
    file_name = []
    for f in ["20210416-part_10.pkl"]:  # sorted(os.listdir(base_dir)):
        example = pd.read_pickle(os.path.join(base_dir, f))
        example = filter_data(example)
        # data.append(example)
        # file_name += ['/Signal Package/{}'.format(f)] * len(data)
        print(f, example.shape)

        plt.subplot(211)
        plt.plot(example["主轴负载"])
        # plt.plot(example['X负载'])
        plt.subplot(212)
        plt.plot(example["主轴转速"])
        # plt.plot(example['Z负载'])
        plt.show()


if __name__ == "__main__":
    base_dir = "../../data/1 训练用/Signal Package"
    # base_dir = '../../data/2 测试用/Signal Package'
    explore_data(base_dir)
