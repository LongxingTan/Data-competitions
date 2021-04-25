
import torch
import os
import json
import numpy as np
import pandas as pd
import joblib
import zipfile
import shutil
import itertools


def predict_single(data_dir, file, model):
    data = np.load(os.path.join(data_dir, file))

    sst = data[..., 0]
    t300 = data[..., 1]
    ua = data[..., 2]
    va = data[..., 3]
    sst, t300, ua, va = tuple([torch.Tensor(np.expand_dims(i, 0)).float() for i in [sst, t300, ua, va]])
    y = model(sst, t300, ua, va)
    y = y.detach().numpy().reshape(-1)
    return y


def predict(data_dir='../tcdata/enso_round1_test_20210201'):  # 提交时： '../tcdata/enso_round1_test_20210201'
    if os.path.exists('../result'):
        shutil.rmtree('../result', ignore_errors=True)
    os.makedirs('../result')

    model = torch.load('../user_data/ref.pkl', map_location='cpu')
    model.eval()
    model.to('cpu')

    # model = simpleSpatailTimeNN()
    # model.load_state_dict(torch.load('../user_data/ref.pkl', map_location='cpu'))
    # model.eval()
    # model.to('cpu')

    for file in os.listdir(data_dir):
        res = predict_single(data_dir, file, model)
        np.save('../result/{}'.format(file), res)
    return


def compress(res_dir='../result', output_dir='result.zip'):
    z = zipfile.ZipFile(output_dir, 'w')
    for d in os.listdir(res_dir):
        z.write(res_dir + os.sep + d)
    z.close()


if __name__ == '__main__':
    predict()
    compress()

    # model = simpleSpatailTimeNN()
    # model.load_state_dict(torch.load('../user_data/ref.pkl', map_location='cpu'))

    # model = torch.load('../user_data/ref.pkl', map_location='cpu')
    # model.eval()
    # model.to('cpu')
    # y = predict_single(data_dir='../data/enso_round1_test_20210201', file='test_0144_01_12.npy', model=model)
    # print(y)
