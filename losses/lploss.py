import torch
import torch.nn as nn


class NormLpLoss(nn.Module):
    def __init__(self, p=2):
        super().__init__()
        self._p = p

    def forward(self, x, y):
        b = x.shape[0]
        error = torch.norm(x.reshape(b, -1) - y.reshape(b, -1), p=self._p, dim=1)
        y_norms = torch.norm(y.reshape(b, -1), p=self._p, dim=1)
        return torch.sum(error / y_norms)
