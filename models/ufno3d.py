import os

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(int(os.environ["EXPERIMENT_SEED"]))


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
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
                dtype=torch.cfloat,
            )
        )

    def compl_mul3d(self, input, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-3),
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, : self.modes2, : self.modes3], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3], self.weights2
        )
        out_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3], self.weights3
        )
        out_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3], self.weights4
        )

        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
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
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=2,
            dropout_rate=dropout_rate,
        )
        self.conv2_1 = self.conv(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout_rate=dropout_rate,
        )
        self.conv3 = self.conv(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=2,
            dropout_rate=dropout_rate,
        )
        self.conv3_1 = self.conv(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout_rate=dropout_rate,
        )

        self.deconv2 = self.deconv(input_channels, output_channels)
        self.deconv1 = self.deconv(input_channels * 2, output_channels)
        self.deconv0 = self.deconv(input_channels * 2, output_channels)

        self.output_layer = self.output(
            input_channels * 2,
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
            nn.Conv3d(
                in_planes,
                output_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                bias=False,
            ),
            nn.BatchNorm3d(output_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),
        )

    def deconv(self, input_channels, output_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(
                input_channels, output_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def output(
        self, input_channels, output_channels, kernel_size, stride, dropout_rate
    ):
        return nn.Conv3d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
        )


class SimpleBlock3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(SimpleBlock3d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.fc0 = nn.Linear(32, self.width)
        self.conv0 = SpectralConv3d(
            self.width, self.width, self.modes1, self.modes2, self.modes3
        )
        self.conv1 = SpectralConv3d(
            self.width, self.width, self.modes1, self.modes2, self.modes3
        )
        self.conv2 = SpectralConv3d(
            self.width, self.width, self.modes1, self.modes2, self.modes3
        )
        self.conv3 = SpectralConv3d(
            self.width, self.width, self.modes1, self.modes2, self.modes3
        )
        self.conv4 = SpectralConv3d(
            self.width, self.width, self.modes1, self.modes2, self.modes3
        )
        self.conv5 = SpectralConv3d(
            self.width, self.width, self.modes1, self.modes2, self.modes3
        )
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.w4 = nn.Conv1d(self.width, self.width, 1)
        self.w5 = nn.Conv1d(self.width, self.width, 1)
        self.unet3 = U_net(self.width, self.width, 3, 0)
        self.unet4 = U_net(self.width, self.width, 3, 0)
        self.unet5 = U_net(self.width, self.width, 3, 0)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        batchsize, _, size_x, size_y, size_z = x.shape

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(
            batchsize, self.width, size_x, size_y, size_z
        )
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(
            batchsize, self.width, size_x, size_y, size_z
        )
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(
            batchsize, self.width, size_x, size_y, size_z
        )
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(
            batchsize, self.width, size_x, size_y, size_z
        )
        x3 = self.unet3(x)
        x = x1 + x2 + x3
        x = F.relu(x)

        x1 = self.conv4(x)
        x2 = self.w4(x.view(batchsize, self.width, -1)).view(
            batchsize, self.width, size_x, size_y, size_z
        )
        x3 = self.unet4(x)
        x = x1 + x2 + x3
        x = F.relu(x)

        x1 = self.conv5(x)
        x2 = self.w5(x.view(batchsize, self.width, -1)).view(
            batchsize, self.width, size_x, size_y, size_z
        )
        x3 = self.unet5(x)
        x = x1 + x2 + x3
        x = F.relu(x)

        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x


class UFNO3d(nn.Module):
    def __init__(
        self, num_input_variables, modes1, modes2, modes3, width, is_stage_3=False
    ):
        super(UFNO3d, self).__init__()

        self.space_embed = nn.Linear(num_input_variables, width, bias=False)
        self.conv1 = SimpleBlock3d(modes1, modes2, modes3, width)
        self.is_stage_3 = is_stage_3

    def forward(self, x):
        t_dim = x.shape[1]
        out = [x[:, 0, ..., 7:8]]

        for t in range(t_dim):
            x_t = torch.concat([x[:, t, ..., :7], out[-1], x[:, t, ..., 8:]], -1)
            b_dim, x_dim, y_dim, z_dim, c_dim = x_t.shape
            x_t = self.space_embed(x_t)
            x_t = x_t.permute(0, 4, 1, 2, 3)
            if self.is_stage_3:
                x_t = F.pad(x_t, (0, 7, 0, 0, 0, 6), "constant", 0)
            else:
                x_t = F.pad(x_t, (0, 7), "constant", 0)
            x_t = self.conv1(x_t)
            if self.is_stage_3:
                x_t = x_t.view(b_dim, x_dim + 6, y_dim, z_dim + 7, 1)[
                    ..., :-6, :, :-7, :
                ]
            else:
                x_t = x_t.view(b_dim, x_dim, y_dim, z_dim + 7, 1)[..., :-7, :]
            out.append(x_t)

        return torch.stack(out[1:], 1)
