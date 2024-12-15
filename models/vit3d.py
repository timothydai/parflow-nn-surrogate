import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import math

import operator
from functools import reduce
from functools import partial

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


class ViT3d(nn.Module):
    def __init__(
        self,
        num_input_variables,
        hidden_size=512,
        num_heads=8,
        num_layers=6,
        is_infil=False,
    ):
        super(ViT3d, self).__init__()
        hidden_size = 64
        self.x_tsl, self.y_tsl, self.z_tsl = (24, 16, 5)  # Tile side length
        x_dim, y_dim, z_dim = 120, 80, 25
        is_infil = False
        if is_infil:
            self.x_tsl, self.y_tsl, self.z_tsl = (3, 3, 4)
            x_dim, y_dim, z_dim = 26, 16, 25
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
            hidden_size, self.x_tsl * self.y_tsl * self.z_tsl if not is_infil else 45
        )

        self.self_attns = None

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
        t_dim = x.shape[1]

        out = [x[:, 0, ..., -1:]]

        for t in range(t_dim):
            x_t = torch.concat([x[:, t, ..., :-1], out[-1]], -1)

            b_dim, x_dim, y_dim, z_dim, c_dim = x_t.shape

            x_t = self.space_embed(x_t)

            x_t = x_t.permute(0, 4, 1, 2, 3)
            x_t = self.patch_embedding(x_t)
            x_t = x_t.flatten(2, -1).permute(2, 0, 1)  # seq_len, B, embed_dim

            x_t = self.pe(x_t)

            x_t = self.transformer_encoder(x_t)

            x_t = x_t.permute(1, 0, 2)  # B, seq_len, embed_dim
            x_t = self.decoder(x_t)

            # if t_dim > 100: RESTORE!!
            #    x_t = x_t.view(b_dim, 26+1, 16, 25, 1)
            #    x_t = x_t[:, :26, :, :, :]
            # else:
            x_t = x_t.view(b_dim, x_dim, y_dim, z_dim, 1)

            out.append(x_t)

        return torch.stack(out[1:], 1)


class ViTCNN3d(nn.Module):
    def __init__(self, num_input_variables, hidden_size=512, num_heads=8, num_layers=6):
        super(ViTCNN3d, self).__init__()
        self.tsl = 16  # Tile side length

        self.space_embed = nn.Linear(num_input_variables, hidden_size, bias=False)
        self.time_embed = nn.Linear(6, hidden_size, bias=False)

        self.patch_embedding = nn.Sequential(
            torch.nn.Conv3d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=5,
                stride=1,
                padding=(5 - 1) // 2,
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
            torch.nn.Conv3d(
                in_channels=hidden_size,
                out_channels=hidden_size * 2,
                kernel_size=5,
                stride=1,
                padding=(5 - 1) // 2,
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
            torch.nn.Conv3d(
                in_channels=hidden_size * 2,
                out_channels=hidden_size,
                kernel_size=5,
                stride=1,
                padding=(5 - 1) // 2,
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv3d(
                hidden_size,
                hidden_size,
                kernel_size=(24, 16, 5),
                stride=(24, 16, 5),
                bias=False,
            ),
        )

        self.pe = PositionalEncoding(
            hidden_size, max_len=(120 // 24) * (80 // 16) * (25 // 5)
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.decoder = nn.Linear(hidden_size, 24 * 16 * 5)

        self.self_attns = None

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
        x, x_time = x

        x = self.space_embed(x) + self.time_embed(x_time)

        b_dim, x_dim, y_dim, z_dim, c_dim = x.shape

        x = x.permute(0, -1, 1, 2, 3)
        x = self.patch_embedding(x)
        x = x.flatten(2, -1).permute(0, 2, 1)  # B, seq_len, embed_dim

        x = x.permute(1, 0, 2)  # seq_len, B, embed_dim
        x = self.pe(x)

        x = self.transformer_encoder(x)

        x = x.permute(1, 0, 2)  # B, seq_len, embed_dim
        x = self.decoder(x)
        x = x.view(
            b_dim, x_dim // 24, y_dim // 16, z_dim // 5, 24, 16, 5
        )  # (B, t, x/tsl, y/tsl, z/tsl, tsl, tsl, tsl)
        x = x.permute(0, 1, 4, 2, 5, 3, 6)
        x = x.flatten(1, 2).flatten(2, 3).flatten(3, 4).unsqueeze(-1)
        return x
