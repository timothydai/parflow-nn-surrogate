import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _quadruple

torch.manual_seed(int(os.environ["EXPERIMENT_SEED"]))


class ConvTranspose4d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: [int, tuple],
        stride: [int, tuple] = (1, 1, 1, 1),
        padding: [int, tuple] = (0, 0, 0, 0),
        dilation: [int, tuple] = (1, 1, 1, 1),
        groups: int = 1,
        bias=True,
        padding_mode: str = "zeros",
    ):
        super(ConvTranspose4d, self).__init__()
        kernel_size = _quadruple(kernel_size)
        stride = _quadruple(stride)
        padding = _quadruple(padding)
        dilation = _quadruple(dilation)

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        valid_padding_modes = {"zeros"}
        if padding_mode not in valid_padding_modes:
            raise ValueError(
                "padding_mode must be one of {}, but got padding_mode='{}'".format(
                    valid_padding_modes, padding_mode
                )
            )

        # Assertions for constructor arguments.
        assert len(kernel_size) == 4, "4D kernel size expected!"
        assert len(stride) == 4, "4D Stride size expected!!"
        assert len(padding) == 4, "4D Padding size expected!!"
        assert len(dilation) == 4, "4D dilation size expected!"
        assert groups == 1, "Groups other than 1 not yet implemented!"

        # Store constructor arguments.
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.groups = groups
        self.padding_mode = padding_mode

        # Construct weight and bias of 4D convolution.
        self.weight = nn.Parameter(
            torch.Tensor(in_channels // groups, out_channels, *kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
        self.reset_parameters()

        # Use a ModuleList to store layers to make the Conv4d layer trainable.
        self.conv3d_layers = torch.nn.ModuleList()

        for i in range(self.kernel_size[0]):
            # Initialize a Conv3D layer.
            conv3d_layer = nn.ConvTranspose3d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size[1::],
                padding=self.padding[1::],
                dilation=self.dilation[1::],
                stride=self.stride[1::],
                bias=False,
            )
            conv3d_layer.weight = nn.Parameter(self.weight[:, :, i])

            # Store the layer.
            self.conv3d_layers.append(conv3d_layer)

        del self.weight

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # Define shortcut names for dimensions of input and kernel.
        (Batch, _, l_i, d_i, h_i, w_i) = tuple(input.shape)
        (l_k, d_k, h_k, w_k) = self.kernel_size
        (l_p, d_p, h_p, w_p) = self.padding
        (l_d, d_d, h_d, w_d) = self.dilation
        (l_s, d_s, h_s, w_s) = self.stride

        # Compute the size of the output tensor based on the zero padding.
        l_o = (l_i - 1) * l_s - 2 * l_p + l_d * (l_k - 1) + 1
        d_o = (d_i - 1) * d_s - 2 * d_p + d_d * (d_k - 1) + 1
        h_o = (h_i - 1) * h_s - 2 * h_p + h_d * (h_k - 1) + 1
        w_o = (w_i - 1) * w_s - 2 * w_p + w_d * (w_k - 1) + 1

        # Pre-define output tensors.
        out = torch.zeros(Batch, self.out_channels, l_o, d_o, h_o, w_o).to(input.device)

        for i in range(l_k):
            for j in range(l_i):
                out_frame = i + j * l_s - l_p
                if out_frame < 0 or out_frame >= l_o:
                    continue
                out[:, :, out_frame] += self.conv3d_layers[i](input[:, :, j])

        # Add bias to output.
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1, 1, 1)

        return out
