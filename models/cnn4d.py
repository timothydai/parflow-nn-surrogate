import os

import torch
import torch.nn as nn

from layers import conv4d, batchnorm4d

torch.manual_seed(int(os.environ["EXPERIMENT_SEED"]))


class CNN4d(nn.Module):
    def __init__(self, num_input_variables, hidden_size):
        super(CNN4d, self).__init__()
        self.x_emb = nn.Linear(num_input_variables, hidden_size)
        self.block0 = nn.Sequential(
            conv4d.Conv4d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=5,
                stride=1,
                padding=(5 - 1) // 2,
            ),
            batchnorm4d.BatchNorm4d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            conv4d.Conv4d(
                in_channels=hidden_size,
                out_channels=2 * hidden_size,
                kernel_size=5,
                stride=1,
                padding=(5 - 1) // 2,
            ),
            batchnorm4d.BatchNorm4d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            conv4d.Conv4d(
                in_channels=2 * hidden_size,
                out_channels=hidden_size,
                kernel_size=5,
                stride=1,
                padding=(5 - 1) // 2,
            ),
            batchnorm4d.BatchNorm4d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
        )
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.x_emb(x)

        x = x.permute(0, 5, 1, 2, 3, 4)
        x = self.block0(x)
        x = x.permute(0, 2, 3, 4, 5, 1)

        x = self.proj(x)
        return x
