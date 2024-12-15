import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


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


if __name__ == "__main__":
    import torchmetrics as tm

    class BatchNorm3d(torch.nn.Module):
        def __init__(self, output_channels):
            super().__init__()
            self.output_channels = output_channels

            gamma = nn.Parameter(torch.rand([16]))
            beta = nn.Parameter(torch.rand([16]))

            self.batchnorm1d = torch.nn.BatchNorm1d(output_channels)

            self.official_batchnorm3d = torch.nn.BatchNorm3d(output_channels)

            self.batchnorm1d.weight = gamma
            self.batchnorm1d.bias = beta

            self.official_batchnorm3d.weight = gamma
            self.official_batchnorm3d.bias = beta

        def forward(self, x):
            official_out = self.official_batchnorm3d(x)

            orig_shape = x.shape
            x = x.view(orig_shape[0], self.output_channels, -1)
            x = self.batchnorm1d(x)
            x = x.view(orig_shape)

            mape = tm.functional.mean_absolute_percentage_error(x, official_out)

            return mape

    for _ in range(100):
        input = torch.randn(4, 16, 120, 80, 25).cuda()
        batchnorm = BatchNorm3d(16).cuda()
        mape = batchnorm(input)
        assert mape < 5e-3, "error"

    print(
        "100 random experiments of batchnorm3d have been completed, and the errors are all within a reasonable range!"
    )
