# -*- coding: utf-8 -*-

import os
import re
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from resnets import *
    from rnn import *
except:
    from .resnets import *
    from .rnn import *


path = os.path.dirname(os.path.abspath((__file__)))
print(path)


class CFG:
    use_model = "resnet34"

    if use_model == "resnet18":
        model_dir = "/resnet18_fold0_seed3150.pth"  # './rnnmodel1_fold0_seed3150.pth'
    elif use_model == "resnet34":
        model_dir = "/resnet34_fold0_seed3150.pth"

    num_classes = 8
    device = "cpu"
    test_batch_size = 1024


def build_data(d_path, labeled=False):
    data = []
    # files = os.listdir(d_path)
    # files = [f for f in os.listdir(d_path) if re.match(r'[0-9]+.txt', f)]
    files = [
        f
        for f in os.listdir(d_path)
        if str(f).split(".")[0].isdigit() and f.endswith("txt")
    ]
    files.sort(key=lambda x: int(str(x).split(".")[0]))

    for file in files:
        try:
            sample = np.loadtxt(open(os.path.join(d_path, file)), encoding="iso-8859-1")
        except:
            raise ValueError(file)
        data.append(sample)
    data = np.stack(data, axis=0)  # local test: 2000 * 4096
    print(data.shape)
    return data


class TestMobileDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.data[idx]
        features = np.expand_dims(features, 0)  # channel first, 2000 * 1 * 4096
        return {"features": torch.tensor(features, dtype=torch.float)}


def build_model(model_dir):
    torch.cuda.empty_cache()
    # model = RNNModel1(num_classes=CFG.num_classes)
    if CFG.use_model == "resnet18":
        model = resnet18()
    elif CFG.use_model == "resnet34":
        model = resnet34()

    model.load_state_dict(torch.load(model_dir))
    model.to(CFG.device)
    model.eval()
    return model


def predict_fn(model, dataloader):
    preds = []
    with torch.no_grad():
        for sample in dataloader:
            sample = sample["features"].to(CFG.device)
            out = model(sample)  # batch * 8
            out1 = out.detach().cpu().numpy()
            preds.append(out1)
        preds = np.concatenate(preds)  # n_test * 8
    return preds


class predictor:
    def __init__(self, test_path, model_path=None):
        self.test_dir = test_path

    def predict(self):
        print("Loading test data...")
        data_X = build_data(
            d_path=os.path.join(self.test_dir, "samples"), labeled=False
        )

        test_dataset = TestMobileDataset(data_X)
        test_loader = DataLoader(
            test_dataset, batch_size=CFG.test_batch_size, shuffle=False
        )

        # load model
        model = build_model(path + CFG.model_dir)

        print("Testing...")
        y_pred = predict_fn(model, test_loader)
        label_pre = np.array([np.argmax(item) for item in y_pred])
        np.savetxt(os.path.join(".", "preLabel.txt"), label_pre, fmt="%d")


# main
if __name__ == "__main__":
    test_path = os.path.join(".", "..", "..", "Data_version", "testing")
    pre = predictor(test_path)
    pre.predict()
