import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from layers.conv4d import Conv4d

import operator
from functools import reduce
from functools import partial


import os

torch.manual_seed(int(os.environ["EXPERIMENT_SEED"]))


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=16):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe
        return self.dropout(x)


class ViT4d(nn.Module):
    def __init__(
        self,
        num_input_variables,
        hidden_size=128,
        num_heads=8,
        num_layers=6,
        is_infil=False,
    ):
        super(ViT4d, self).__init__()
        t_tsl, x_tsl, y_tsl, z_tsl = 2, 24, 16, 5
        t_dim, x_dim, y_dim, z_dim = 16, 120, 80, 25
        if is_infil:
            t_tsl = 16
            x_tsl = 3
            y_tsl = 3
            z_tsl = 4
            t_dim = 366
            x_dim = 26
            y_dim = 16
            z_dim = 25
        max_pe_len = (
            (t_dim // t_tsl) * (x_dim // x_tsl) * (y_dim // y_tsl) * (z_dim // z_tsl)
        )

        hidden_size = 64

        self.space_embed = nn.Linear(num_input_variables, hidden_size, bias=False)
        # self.time_embed = nn.Linear(6, hidden_size, bias=False)

        self.embed = Conv4d(
            hidden_size,
            hidden_size,
            kernel_size=(t_tsl, x_tsl, y_tsl, z_tsl),
            stride=(t_tsl, x_tsl, y_tsl, z_tsl),
        )
        self.pe = PositionalEncoding(hidden_size, max_len=max_pe_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.decoder = nn.Linear(
            hidden_size, t_tsl * x_tsl * y_tsl * z_tsl if not is_infil else 1314
        )
        self.is_infil = is_infil

    def register_hooks(self):
        for i in range(8):
            self.transformer_encoder.layers[i].self_attn.register_forward_pre_hook(
                self.self_attn_input_change, with_kwargs=True
            )
            self.transformer_encoder.layers[i].self_attn.register_forward_hook(
                self.self_attn_hook
            )

    def get_tracked(self):
        return self.self_attns

    def self_attn_input_change(self, module, input, kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False

    def self_attn_hook(self, module, args, output):
        self.self_attns.append(output[1].detach().cpu())

    def forward(self, x):
        self.self_attns = []
        b_dim, t_dim, x_dim, y_dim, z_dim, _ = x.shape

        x = self.space_embed(x)
        x = x.permute(0, -1, 1, 2, 3, 4)

        x = self.embed(x)
        x = x.flatten(2, -1).permute(2, 0, 1)  # seq_len, B, embed_dim

        x = self.pe(x)

        x = self.transformer_encoder(x)

        x = x.permute(1, 0, 2)  # B, seq_len, embed_dim
        x = self.decoder(x)

        if t_dim > 100:
            x = x.view(b_dim, 365, 26 + 6, 16 + 6, 25 + 2, 1)
            x = x[..., :26, :16, :25, :]
        else:
            x = x.view(b_dim, t_dim, x_dim, y_dim, z_dim, 1)
        return x
