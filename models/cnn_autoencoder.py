import os
import torch
import torch.nn as nn


torch.manual_seed(int(os.environ["EXPERIMENT_SEED"]))


class CNNAutoencoder(nn.Module):
    def __init__(self):
        super(CNNAutoencoder, self).__init__()
        k_x = 7
        k_y = 7
        s_x = 2
        s_y = 2

        self.enc = nn.Sequential(
            nn.Conv2d(25, 128, kernel_size=(k_x, k_y), stride=(s_x, s_y)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 25, kernel_size=(k_x, k_y), stride=(s_x, s_y)),
            nn.BatchNorm2d(25),
            nn.Sigmoid(),
        )

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(25, 128, kernel_size=(k_x, k_y), stride=(s_x, s_y)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(
                128, 25, kernel_size=(k_x + 1, k_y + 1), stride=(s_x, s_y)
            ),
        )

    def forward(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        has_t_dim = False
        if x.dim() == 5:
            _b, _x, _y, _z, _c = x.shape
        elif x.dim() == 6:
            has_t_dim = True
            _b, _t, _x, _y, _z, _c = x.shape
        assert _c == 1
        x = x.reshape(-1, _x, _y, _z)
        x = x.permute(0, -1, 1, 2)
        x = self.enc(x)
        x = x.permute(0, 2, 3, 1)
        _, _x, _y, _z = x.shape
        if not has_t_dim:
            x = x.reshape(_b, _x, _y, _z, _c)
        else:
            x = x.reshape(_b, _t, _x, _y, _z, _c)
        return x

    def decode(self, x):
        has_t_dim = False
        if x.dim() == 5:
            _b, _x, _y, _z, _c = x.shape
        elif x.dim() == 6:
            has_t_dim = True
            _b, _t, _x, _y, _z, _c = x.shape
        assert _c == 1
        x = x.reshape(-1, _x, _y, _z)
        x = x.permute(0, -1, 1, 2)
        x = self.dec(x)
        x = x.permute(0, 2, 3, 1)
        _, _x, _y, _z = x.shape
        if not has_t_dim:
            x = x.reshape(_b, _x, _y, _z, _c)
        else:
            x = x.reshape(_b, _t, _x, _y, _z, _c)
        return x

