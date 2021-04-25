import sys
sys.path.append('.')  # dataset.read
from dataset.read_data import prepare_data

import os
import json
import numpy as np
import pandas as pd
import joblib
import zipfile
import shutil
import itertools
import tensorflow as tf


def transform2train(sample, start_month=1, only_roi=False):
    if only_roi:
        sample = sample[:, 10: 13, 38: 49, :]  # filter roi
        # first transform to 2d long type, same as SODA_train_roi, (12, 3, 11, 4)
        cols = list(itertools.product(range(start_month + 11, start_month - 1, -1),
                                    [-5, 0, 5],
                                    [190., 195., 200., 205., 210., 215., 220., 225., 230., 235., 240.],
                                    ))
    else:
        cols = list(itertools.product(range(start_month, start_month + 12),
                                    range(-55, 65, 5),
                                    range(0, 360, 5),
                                    ))

    data = pd.DataFrame(cols, columns=['month', 'lat', 'lon'])
    data['year'] = 1
    data['sst'] = sample[..., 0].reshape(-1)
    data['t300'] = sample[..., 1].reshape(-1)
    data['ua'] = sample[..., 2].reshape(-1)
    data['va'] = sample[..., 3].reshape(-1)
    return data


def predict_single(data_dir, file, model):
    data = np.load(os.path.join(data_dir, file))
    start_month = int(file.split('_')[2])
    if start_month <= 0 or start_month >= 13:
        print("month Error")

    data = transform2train(data, start_month=start_month)
    sst, t300, ua, va, month= prepare_data(data)
    month = month - 1 ## 与训练特征对齐，注意35分那个，这个忘记-1
    
    x = tuple([i[np.newaxis, ...].astype(np.float32) for i in [sst, t300, ua, va, month]])
    y = model(x)

    out = y.numpy().reshape(-1)  + 0.01
    return out


def predict(data_dir='../tcdata/enso_round1_test_20210201', model_dir='../user_data/fine'):  # 提交时： '../tcdata/enso_round1_test_20210201'
    if os.path.exists('../result'):
        shutil.rmtree('../result', ignore_errors=True)
    os.makedirs('../result')

    model = tf.saved_model.load(model_dir)

    for file in os.listdir(data_dir):        
        res = predict_single(data_dir, file, model)    
        np.save('../result/{}'.format(file), res)
    return


def compress(res_dir='../result', output_dir='result.zip'):
    z = zipfile.ZipFile(output_dir, 'w')
    for d in os.listdir(res_dir):
        z.write(res_dir + os.sep + d)
    z.close()


def local_test():
    model = tf.saved_model.load('../user_data/nn')
    y = predict_single(data_dir='../tcdata/enso_round1_test_20210201', file='test_0144_01_12.npy', model=model)
    y = y + 0.01
    print(y)


if __name__ == '__main__':   
    model_dir = '../user_data/nn'
    # predict(model_dir=model_dir, data_dir='../tcdata/enso_final_test_data_B')
    # compress()
    local_test()
