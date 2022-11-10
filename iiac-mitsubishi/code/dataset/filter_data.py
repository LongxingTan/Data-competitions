# 去除停机与异常数据
import os
import numpy as np
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
plt.style.use("ggplot")


def filter_data(example, verbose=1, plot=False):
    """
    思路：
    首先检查是否存在停机数据：根据日期和长度即可判断
    1. 首先去除主负载大于50的数据
    2. 对于个数大于40000的样本，检查其停机。但是很多7万的样本也是正常的
    3. 主负载rolling mean 窗口足够大，平均值小于5。找到对应转速，将对应转速能延伸的两边确定为边界。去除停机数据

    思路二：
    检查停机的转速是否都是91附近
    :return:
    """
    example = example.loc[(example['主轴负载'] < 50) & (example['X负载'] < 15)]

    # if example['Time'][0] > date(2021, 5, 10):
    #     if example.shape[0] < 40000:
    #         return example
    # else:
    #     if example.shape[0] < 80000:
    #         return example

    example_diff10 = example['主轴负载'].diff(periods=10)
    example_diff1000 = example['主轴负载'].diff(periods=1000)
    example_filtered = example.loc[(example_diff10 > 0.9) | (example_diff1000 > 0.9)].reset_index()

    # if str(example['Time'][0]) == '2021-05-17 13:33:30.020000':  # 手动去掉 '20210517-part_18.pkl'
    #     example_filtered = example_filtered.iloc[list(range(2000)) + list(range(16000, 33336))].reset_index()
    # if str(example['Time'][0] == '2021-06-01 19:16:00.043000'):  # 手动去掉 '20210601-part_18.pkl
    #     example_filtered = example_filtered.iloc[list(range(2000)) + list(range(8000, 25387))].reset_index()

    if verbose > 0:
        print('Data shape from {} to {}'.format(example.shape[0], example_filtered.shape[0]))

    if plot:
        plt.subplot(211)
        plt.plot(example['主轴负载'])
        plt.subplot(212)
        plt.plot(example_filtered['主轴负载'])
        plt.show()

    return example_filtered


if __name__ == '__main__':
    file = '20210601-part_18.pkl'
    # example = pd.read_pickle('../../data/2 测试用/Signal Package/' + file)
    example = pd.read_pickle('../../data/1 训练用/Signal Package/' + file)
    print(str(example['Time'][0]))
    filter_data(example, plot=True)

    # files = sorted(os.listdir('../../data/1 训练用/Signal Package'))
    # for file in files:
    #     example = pd.read_pickle('../../data/1 训练用/Signal Package/' + file)
    #     filter_data(example, plot=True)

