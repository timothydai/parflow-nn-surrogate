import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _quadruple

torch.manual_seed(0)


class Conv4d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: [int, tuple],
        stride: [int, tuple] = (1, 1, 1, 1),
        padding: [int, tuple] = (0, 0, 0, 0),
        dilation: [int, tuple] = (1, 1, 1, 1),
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ):
        super(Conv4d, self).__init__()
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

        # Assertions for constructor arguments
        assert len(kernel_size) == 4, "4D kernel size expected!"
        assert len(stride) == 4, "4D Stride size expected!!"
        assert len(padding) == 4, "4D Padding size expected!!"
        assert len(dilation) == 4, "4D dilation size expected!"
        assert groups == 1, "Groups other than 1 not yet implemented!"

        # Store constructor arguments
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.groups = groups
        self.padding_mode = padding_mode

        # Construct weight and bias of 4D convolution
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
        self.reset_parameters()

        # Use a ModuleList to store layers to make the Conv4d layer trainable
        self.conv3d_layers = torch.nn.ModuleList()

        for i in range(self.kernel_size[0]):
            # Initialize a Conv3D layer
            conv3d_layer = nn.Conv3d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size[1::],
                padding=self.padding[1::],
                dilation=self.dilation[1::],
                stride=self.stride[1::],
                bias=False,
            )
            conv3d_layer.weight = nn.Parameter(self.weight[:, :, i])

            # Store the layer
            self.conv3d_layers.append(conv3d_layer)

        del self.weight

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # Define shortcut names for dimensions of input and kernel
        (Batch, _, l_i, d_i, h_i, w_i) = tuple(input.shape)
        (l_k, d_k, h_k, w_k) = self.kernel_size
        (l_p, d_p, h_p, w_p) = self.padding
        (l_d, d_d, h_d, w_d) = self.dilation
        (l_s, d_s, h_s, w_s) = self.stride

        # Compute the size of the output tensor based on the zero padding
        l_o = (l_i + 2 * l_p - (l_k) - (l_k - 1) * (l_d - 1)) // l_s + 1
        d_o = (d_i + 2 * d_p - (d_k) - (d_k - 1) * (d_d - 1)) // d_s + 1
        h_o = (h_i + 2 * h_p - (h_k) - (h_k - 1) * (h_d - 1)) // h_s + 1
        w_o = (w_i + 2 * w_p - (w_k) - (w_k - 1) * (w_d - 1)) // w_s + 1

        # Pre-define output tensors
        out = torch.zeros(Batch, self.out_channels, l_o, d_o, h_o, w_o).to(input.device)

        # Convolve each kernel frame i with each input frame j
        for i in range(l_k):
            # Calculate the zero-offset of kernel frame i
            zero_offset = -l_p + (i * l_d)
            # Calculate the range of input frame j corresponding to kernel frame i
            j_start = max(zero_offset % l_s, zero_offset)
            j_end = min(l_i, l_i + l_p - (l_k - i - 1) * l_d)
            # Convolve each kernel frame i with corresponding input frame j
            for j in range(j_start, j_end, l_s):
                # Calculate the output frame
                out_frame = (j - zero_offset) // l_s
                # Add results to this output frame
                out[:, :, out_frame] += self.conv3d_layers[i](input[:, :, j])

        # Add bias to output
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1, 1, 1)

        return out


if __name__ == "__main__":
    import numpy as np
    import torchmetrics as tm
    from torch.nn.modules.utils import _triple

    # Validate implementation by copying above implementation and modifying slightly
    # to make it Conv3d. Then, compare results of forward pass to official PyTorch Conv3d.
    class Conv3d(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: [int, tuple],
            stride: [int, tuple] = (1, 1, 1),
            padding: [int, tuple] = (0, 0, 0),
            dilation: [int, tuple] = (1, 1, 1),
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = "zeros",
        ):
            super(Conv3d, self).__init__()
            kernel_size = _triple(kernel_size)
            stride = _triple(stride)
            padding = _triple(padding)
            dilation = _triple(dilation)

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

            # Assertions for constructor arguments
            assert len(kernel_size) == 3, "3D kernel size expected!"
            assert len(stride) == 3, "3D Stride size expected!!"
            assert len(padding) == 3, "3D Padding size expected!!"
            assert len(dilation) == 3, "3D dilation size expected!"
            assert groups == 1, "Groups other than 1 not yet implemented!"

            # Store constructor arguments
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
            self.dilation = dilation

            self.groups = groups
            self.padding_mode = padding_mode

            # Construct weight and bias of 3D convolution
            self.weight = nn.Parameter(
                torch.Tensor(out_channels, in_channels // groups, *kernel_size)
            )
            if bias:
                self.bias = nn.Parameter(torch.Tensor(out_channels))
            else:
                self.bias = None

            self.reset_parameters()

            # Use a ModuleList to store layers to make the Conv4d layer trainable
            self.conv2d_layers = torch.nn.ModuleList()

            for i in range(self.kernel_size[0]):
                # Initialize a Conv3D layer
                conv2d_layer = nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size[1::],
                    padding=self.padding[1::],
                    dilation=self.dilation[1::],
                    stride=self.stride[1::],
                    bias=False,
                )
                conv2d_layer.weight = nn.Parameter(self.weight[:, :, i])

                # Store the layer
                self.conv2d_layers.append(conv2d_layer)

            self.official_conv3d = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias,
            )
            self.official_conv3d.weight = self.weight
            self.official_conv3d.bias = self.bias

        def reset_parameters(self) -> None:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

        def forward(self, input):
            # Define shortcut names for dimensions of input and kernel
            (Batch, _, l_i, h_i, w_i) = tuple(input.shape)
            (l_k, h_k, w_k) = self.kernel_size
            (l_p, h_p, w_p) = self.padding
            (l_d, h_d, w_d) = self.dilation
            (l_s, h_s, w_s) = self.stride

            # Compute the size of the output tensor based on the zero padding
            l_o = (l_i + 2 * l_p - (l_k) - (l_k - 1) * (l_d - 1)) // l_s + 1
            h_o = (h_i + 2 * h_p - (h_k) - (h_k - 1) * (h_d - 1)) // h_s + 1
            w_o = (w_i + 2 * w_p - (w_k) - (w_k - 1) * (w_d - 1)) // w_s + 1

            # Pre-define output tensors
            out = torch.zeros(Batch, self.out_channels, l_o, h_o, w_o).to(input.device)

            # Convolve each kernel frame i with each input frame j
            for i in range(l_k):
                # Calculate the zero-offset of kernel frame i
                zero_offset = -l_p + (i * l_d)
                # Calculate the range of input frame j corresponding to kernel frame i
                j_start = max(zero_offset % l_s, zero_offset)
                j_end = min(l_i, l_i + l_p - (l_k - i - 1) * l_d)
                # Convolve each kernel frame i with corresponding input frame j
                for j in range(j_start, j_end, l_s):
                    # Calculate the output frame
                    out_frame = (j - zero_offset) // l_s
                    # Add results to this output frame
                    out[:, :, out_frame] += self.conv2d_layers[i](input[:, :, j])

            # Add bias to output
            if self.bias is not None:
                out = out + self.bias.view(1, -1, 1, 1, 1)

            ################## Validation ##################
            out_official = self.official_conv3d(input)

            mape = tm.functional.mean_absolute_percentage_error(out, out_official)

            return mape

    for _ in range(100):
        input = torch.randn(4, 16, 120, 80, 25).cuda()
        kernel_size = (
            np.random.randint(1, 5),
            np.random.randint(1, 5),
            np.random.randint(1, 5),
        )
        padding = (
            np.random.randint(0, 5),
            np.random.randint(0, 5),
            np.random.randint(0, 5),
        )
        stride = (
            np.random.randint(1, 5),
            np.random.randint(1, 5),
            np.random.randint(1, 5),
        )
        dilation = 1  # (np.random.randint(1, 3), np.random.randint(1, 3), np.random.randint(1, 3))
        net = Conv3d(
            16,
            32,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=1,
            bias=True,
        ).cuda()
        mape = net(input)
        assert mape < 5e-3, "error"

    print(
        "100 random experiments of conv3d have been completed, and the errors are all within a reasonable range!"
    )
