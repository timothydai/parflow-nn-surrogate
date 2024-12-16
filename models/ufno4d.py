import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import batchnorm4d, conv4d, convtranspose4d


torch.manual_seed(int(os.environ["EXPERIMENT_SEED"]))


class SpectralConv4d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, modes4):
        super(SpectralConv4d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.modes4 = modes4

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                dtype=torch.cfloat,
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                dtype=torch.cfloat,
            )
        )
        self.weights3 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                dtype=torch.cfloat,
            )
        )
        self.weights4 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                dtype=torch.cfloat,
            )
        )

    def compl_mul3d(self, input, weights):
        return torch.einsum("bitxyz,iotxyz->botxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x)

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-4),
            x.size(-3),
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2, : self.modes3, : self.modes4] = (
            self.compl_mul3d(
                x_ft[:, :, : self.modes1, : self.modes2, : self.modes3, : self.modes4],
                self.weights1,
            )
        )
        out_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3, : self.modes4] = (
            self.compl_mul3d(
                x_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3, : self.modes4],
                self.weights2,
            )
        )
        out_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3, : self.modes4] = (
            self.compl_mul3d(
                x_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3, : self.modes4],
                self.weights3,
            )
        )
        out_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3, : self.modes4] = (
            self.compl_mul3d(
                x_ft[
                    :, :, -self.modes1 :, -self.modes2 :, : self.modes3, : self.modes4
                ],
                self.weights4,
            )
        )

        x = torch.fft.irfftn(
            out_ft,
            s=(x.size(-4), x.size(-3), x.size(-2), x.size(-1)),
        )
        return x


class U_net(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, dropout_rate):
        super(U_net, self).__init__()
        self.input_channels = input_channels
        self.conv1 = self.conv(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=2,
            dropout_rate=dropout_rate,
        )
        self.conv2 = self.conv(
            output_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=2,
            dropout_rate=dropout_rate,
        )
        self.conv2_1 = self.conv(
            output_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout_rate=dropout_rate,
        )
        self.conv3 = self.conv(
            output_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=2,
            dropout_rate=dropout_rate,
        )
        self.conv3_1 = self.conv(
            output_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout_rate=dropout_rate,
        )

        self.deconv2 = self.deconv(output_channels, output_channels)
        self.deconv1 = self.deconv(output_channels * 2, output_channels)
        self.deconv0 = self.deconv(output_channels * 2, output_channels)

        self.output_layer = self.output(
            output_channels * 2,
            output_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout_rate=dropout_rate,
        )

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))

        out_deconv2 = self.deconv2(out_conv3)
        concat2 = torch.cat((out_conv2, out_deconv2), 1)

        out_deconv1 = self.deconv1(concat2)
        concat1 = torch.cat((out_conv1, out_deconv1), 1)
        out_deconv0 = self.deconv0(concat1)

        concat0 = torch.cat((x, out_deconv0), 1)
        out = self.output_layer(concat0)

        return out

    def conv(self, in_planes, output_channels, kernel_size, stride, dropout_rate):
        return nn.Sequential(
            conv4d.Conv4d(
                in_channels=in_planes,
                out_channels=output_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                bias=False,
            ),
            batchnorm4d.BatchNorm4d(output_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),
        )

    def deconv(self, input_channels, output_channels):
        return nn.Sequential(
            convtranspose4d.ConvTranspose4d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def output(
        self, input_channels, output_channels, kernel_size, stride, dropout_rate
    ):
        return conv4d.Conv4d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
        )


class FourierLayer(nn.Module):
    def __init__(self, modes1, modes2, modes3, modes4, width):
        super(FourierLayer, self).__init__()
        self.conv = SpectralConv4d(width, width, modes1, modes2, modes3, modes4)
        self.w = nn.Conv1d(width, width, 1)

        self.width = width

    def forward(self, x):
        b_dim, _, t_dim, x_dim, y_dim, z_dim = x.shape
        x1 = self.conv(x)
        x2 = self.w(x.view(b_dim, self.width, -1)).view(
            b_dim, self.width, t_dim, x_dim, y_dim, z_dim
        )
        x = x1 + x2
        x = F.relu(x)
        return x


class UFourierLayer(nn.Module):
    def __init__(self, modes1, modes2, modes3, modes4, width):
        super(UFourierLayer, self).__init__()
        self.conv = SpectralConv4d(width, width, modes1, modes2, modes3, modes4)
        self.w = nn.Conv1d(width, width, 1)
        self.unet = U_net(width, width, 3, 0)

        self.width = width

    def forward(self, x):
        b_dim, _, t_dim, x_dim, y_dim, z_dim = x.shape
        x1 = self.conv(x)
        x2 = self.w(x.view(b_dim, self.width, -1)).view(
            b_dim, self.width, t_dim, x_dim, y_dim, z_dim
        )
        x3 = self.unet(x)
        x = x1 + x2 + x3
        x = F.relu(x)
        return x


class SimpleBlock4d(nn.Module):
    def __init__(
        self, num_F_layers, num_UF_layers, modes1, modes2, modes3, modes4, width
    ):
        super(SimpleBlock4d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.modes4 = modes4
        self.width = width

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        self.F_layers = nn.Sequential(
            *[
                FourierLayer(modes1, modes2, modes3, modes4, width)
                for _ in range(num_F_layers)
            ]
        )
        self.UF_layers = nn.Sequential(
            *[
                UFourierLayer(modes1, modes2, modes3, modes4, width)
                for _ in range(num_UF_layers)
            ]
        )

    def forward(self, x):
        x = self.F_layers(x)
        x = self.UF_layers(x)

        x = x.permute(0, 2, 3, 4, 5, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x


class UFNO4d(nn.Module):
    def __init__(
        self,
        num_input_variables,
        num_F_layers,
        num_UF_layers,
        modes1,
        modes2,
        modes3,
        modes4,
        width,
        is_stage_3=False,
    ):
        super(UFNO4d, self).__init__()

        self.width = width
        self.space_embed = nn.Linear(num_input_variables, self.width, bias=False)

        self.conv1 = SimpleBlock4d(
            num_F_layers, num_UF_layers, modes1, modes2, modes3, modes4, width
        )
        self.is_stage_3 = is_stage_3

    def forward(self, x):
        b_dim, t_dim, x_dim, y_dim, z_dim, c_dim = x.shape

        x = self.space_embed(x)
        x = x.permute(0, 5, 1, 2, 3, 4)
        if self.is_stage_3:
            x = F.pad(x, (0, 7, 0, 0, 0, 6, 0, 3), "constant", 0)
        else:
            x = F.pad(x, (0, 7), "constant", 0)
        output = self.conv1(x)
        if self.is_stage_3:
            output = output.view(b_dim, t_dim + 3, x_dim + 6, y_dim, z_dim + 7, 1)[
                ..., :-3, :-6, :, :-7, :
            ]
        else:
            output = output.view(b_dim, t_dim, x_dim, y_dim, z_dim + 7, 1)[..., :-7, :]

        return output
