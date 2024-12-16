import os

import torch
import torch.nn as nn


torch.manual_seed(int(os.environ["EXPERIMENT_SEED"]))


class CNN3d(nn.Module):
    def __init__(self, num_input_variables, hidden_size):
        super(CNN3d, self).__init__()
        self.block0 = nn.Sequential(
            torch.nn.Conv3d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=5,
                stride=1,
                padding=(5 - 1) // 2,
            ),
            torch.nn.BatchNorm3d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            torch.nn.Conv3d(
                in_channels=hidden_size,
                out_channels=hidden_size * 2,
                kernel_size=5,
                stride=1,
                padding=(5 - 1) // 2,
            ),
            torch.nn.BatchNorm3d(hidden_size * 2),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            torch.nn.Conv3d(
                in_channels=hidden_size * 2,
                out_channels=hidden_size,
                kernel_size=5,
                stride=1,
                padding=(5 - 1) // 2,
            ),
            torch.nn.BatchNorm3d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
        )
        self.proj = nn.Linear(hidden_size, 1)

        self.x_emb = nn.Linear(num_input_variables, hidden_size)

    def forward(self, x):
        out = [x[:, 0, ..., 7:8]]

        t_dim = x.shape[1]
        for t in range(t_dim):
            x_t = torch.concat([x[:, t, ..., :7], out[-1], x[:, t, ..., 8:]], -1)
            x_t = self.x_emb(x_t)
            x_t = x_t.permute(0, 4, 1, 2, 3)
            x_t = self.block0(x_t)
            x_t = x_t.permute(0, 2, 3, 4, 1)
            x_t = self.proj(x_t)
            out.append(x_t)
        return torch.stack(out[1:], 1)
