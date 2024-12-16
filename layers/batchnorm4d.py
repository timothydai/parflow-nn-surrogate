import os

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(int(os.environ["EXPERIMENT_SEED"]))


class BatchNorm4d(torch.nn.Module):
    def __init__(self, output_channels):
        super().__init__()
        self.output_channels = output_channels
        self.batchnorm1d = torch.nn.BatchNorm1d(output_channels)

    def forward(self, x):
        orig_shape = x.shape
        x = x.view(orig_shape[0], self.output_channels, -1)
        x = self.batchnorm1d(x)
        x = x.view(orig_shape)
        return x
