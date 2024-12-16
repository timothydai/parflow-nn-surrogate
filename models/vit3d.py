import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        x = x + self.pe
        return self.dropout(x)


class ViT3d(nn.Module):
    def __init__(
        self,
        num_input_variables,
        hidden_size=512,
        num_heads=8,
        num_layers=6,
        is_stage_3=False,
    ):
        super(ViT3d, self).__init__()
        hidden_size = 64
        self.x_tsl, self.y_tsl, self.z_tsl = (24, 16, 5)
        x_dim, y_dim, z_dim = 120, 80, 25
        if is_stage_3:
            self.x_tsl, self.y_tsl, self.z_tsl = (3, 3, 4)
            x_dim, y_dim, z_dim = 26, 16, 25
        self.is_stage_3 = is_stage_3
        max_pe_len = (
            (x_dim // self.x_tsl) * (y_dim // self.y_tsl) * (z_dim // self.z_tsl)
        )

        self.space_embed = nn.Linear(num_input_variables, hidden_size, bias=False)
        self.patch_embedding = nn.Conv3d(
            hidden_size,
            hidden_size,
            kernel_size=(self.x_tsl, self.y_tsl, self.z_tsl),
            stride=(self.x_tsl, self.y_tsl, self.z_tsl),
            bias=False,
        )

        self.pe = PositionalEncoding(hidden_size, max_len=max_pe_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.decoder = nn.Linear(
            hidden_size, self.x_tsl * self.y_tsl * self.z_tsl if not is_stage_3 else 45
        )

    def forward(self, x):
        t_dim = x.shape[1]

        out = [x[:, 0, ..., 7:8]]

        for t in range(t_dim):
            x_t = torch.concat([x[:, t, ..., :7], out[-1], x[:, t, ..., 8:]], -1)

            b_dim, x_dim, y_dim, z_dim, c_dim = x_t.shape

            x_t = self.space_embed(x_t)

            x_t = x_t.permute(0, 4, 1, 2, 3)
            x_t = self.patch_embedding(x_t)
            x_t = x_t.flatten(2, -1).permute(2, 0, 1)

            x_t = self.pe(x_t)

            x_t = self.transformer_encoder(x_t)

            x_t = x_t.permute(1, 0, 2)
            x_t = self.decoder(x_t)

            if self.is_stage_3:
                x_t = x_t.view(b_dim, 26 + 1, 16, 25, 1)
                x_t = x_t[:, :26, :, :, :]
            else:
                x_t = x_t.view(b_dim, x_dim, y_dim, z_dim, 1)

            out.append(x_t)

        return torch.stack(out[1:], 1)
