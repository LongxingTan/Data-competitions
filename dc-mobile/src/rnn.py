import torch
import torch.nn as nn
import torch.nn.functional as F

params = {"input_dim": 1, "lstm_dim": 32, "dense_dim": 32, "logit_dim": 16}


class RNNModel1(nn.Module):
    def __init__(self, num_classes=2, seq_len=4096, custom_model_params=None) -> None:
        super().__init__()
        if custom_model_params:
            params.update(custom_model_params)

        self.mlp = nn.Sequential(
            nn.Linear(params["input_dim"], params["dense_dim"] // 2),
            nn.ReLU(),
            nn.Linear(params["dense_dim"] // 2, params["dense_dim"]),
            nn.ReLU(),
        )
        self.rnn = nn.LSTM(
            params["dense_dim"],
            params["lstm_dim"],
            batch_first=True,
            bidirectional=True,
        )
        self.logits = nn.Sequential(
            nn.Linear(params["lstm_dim"] * 2 * seq_len, params["logit_dim"]),
            nn.ReLU(),
            nn.Linear(params["logit_dim"], num_classes),
        )

    def forward(self, x):
        features = self.mlp(x)
        features, _ = self.rnn(features)
        features = torch.flatten(features, start_dim=1)

        out = self.logits(features)
        return out


if __name__ == "__main__":
    net = RNNModel1(num_classes=8)
    x = torch.randn(1, 4096, 1)
    y = net(x)
    print(y.shape)
