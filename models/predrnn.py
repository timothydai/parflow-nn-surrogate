import os

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(int(os.environ["EXPERIMENT_SEED"]))


class GHU(nn.Module):
    def __init__(self, hidden_size):
        super(GHU, self).__init__()
        self.Wx = nn.Conv3d(hidden_size, hidden_size * 2, 5, 1, 2)
        self.Wz = nn.Conv3d(hidden_size, hidden_size * 2, 5, 1, 2)

        self.hidden_size = hidden_size

        self.switch = None

    def get_switch(self):
        return self.switch

    def forward(self, x, z):
        p, s = torch.split(self.Wx(x) + self.Wz(z), self.hidden_size, 1)
        p = torch.tanh(p)
        s = torch.sigmoid(s)

        self.switch = s.detach().cpu().mean()

        z = s * p + (1 - s) * z
        return z


class CausalLSTMCell(nn.Module):
    def __init__(self, hidden_size_in, hidden_size_out):
        super(CausalLSTMCell, self).__init__()
        self.Wx = nn.Conv3d(hidden_size_in, hidden_size_out * 7, 5, 1, 2)
        self.Wh = nn.Conv3d(hidden_size_out, hidden_size_out * 3, 5, 1, 2)
        self.Wc = nn.Conv3d(hidden_size_out, hidden_size_out * 3, 5, 1, 2)
        self.Wc1 = nn.Conv3d(hidden_size_out, hidden_size_out * 4, 5, 1, 2)
        self.Wm = nn.Conv3d(hidden_size_out, hidden_size_out * 3, 5, 1, 2)
        self.Wmo = nn.Conv3d(hidden_size_out, hidden_size_out, 5, 1, 2)

        self.W11 = nn.Conv3d(hidden_size_out * 2, hidden_size_out, 1)

        self.hidden_size = hidden_size_out

        self.forget = None
        self.forget_prime = None

    def get_forget(self):
        return (self.forget, self.forget_prime)

    def forward(self, x, h, c, m):
        xg, xi, xf, xgprime, xiprime, xfprime, xo = torch.split(
            self.Wx(x), self.hidden_size, 1
        )
        hg, hi, hf = torch.split(self.Wh(h), self.hidden_size, 1)
        cg, ci, cf = torch.split(self.Wc(c), self.hidden_size, 1)

        g = torch.tanh(xg + hg + cg)
        i = torch.sigmoid(xi + hi + ci)
        f = torch.sigmoid(xf + hf + cf)

        self.forget = f.detach().cpu().mean()

        c = f * c + i * g

        cgprime, ciprime, cfprime, co = torch.split(self.Wc1(c), self.hidden_size, 1)
        mgprime, miprime, mfprime = torch.split(self.Wm(m), self.hidden_size, 1)

        gprime = torch.tanh(xgprime + cgprime + mgprime)
        iprime = torch.sigmoid(xiprime + ciprime + miprime)
        fprime = torch.sigmoid(xfprime + cfprime + mfprime)

        self.forget_prime = fprime.detach().cpu().mean()

        m = fprime * torch.tanh(m) + iprime * gprime

        o = torch.tanh(xo + co + self.Wmo(m))

        h = o * torch.tanh(self.W11(torch.cat([c, m], dim=1)))
        return h, c, m


class PredRNN(nn.Module):
    def __init__(self, num_input_variables, hidden_size, num_layers):
        super(PredRNN, self).__init__()
        self.cells = nn.ModuleList()

        for i in range(num_layers):
            self.cells.append(CausalLSTMCell(hidden_size, hidden_size))

        self.ghu = GHU(hidden_size)
        self.proj = nn.Conv3d(hidden_size, 1, 1)

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.space_embed = nn.Linear(num_input_variables, self.hidden_size, bias=False)

    def forward(self, x):
        b_dim, t_dim, x_dim, y_dim, z_dim, c_dim = x.shape

        x = self.space_embed(x)
        x = x.permute(0, 1, 5, 2, 3, 4)

        h_t = [
            torch.zeros(
                (1, self.hidden_size, x_dim, y_dim, z_dim), dtype=torch.float32
            ).to(x.device)
            for _ in range(self.num_layers)
        ]
        c_t = [
            torch.zeros(
                (1, self.hidden_size, x_dim, y_dim, z_dim), dtype=torch.float32
            ).to(x.device)
            for _ in range(self.num_layers)
        ]
        z = torch.zeros(
            (1, self.hidden_size, x_dim, y_dim, z_dim), dtype=torch.float32
        ).to(x.device)
        m = torch.zeros(
            (1, self.hidden_size, x_dim, y_dim, z_dim), dtype=torch.float32
        ).to(x.device)
        out = []
        for t in range(t_dim):
            x_t = x[:, t, ...]
            h_t[0], c_t[0], m = self.cells[0](x_t, h_t[0], c_t[0], m)
            z = self.ghu(h_t[0], z)
            h_t[1], c_t[1], m = self.cells[1](z, h_t[1], c_t[1], m)
            for i in range(2, self.num_layers):
                h_t[i], c_t[i], m = self.cells[i](h_t[i - 1], h_t[i], c_t[i], m)
            out.append(self.proj(h_t[self.num_layers - 1]))

        out = torch.stack(out, dim=1)
        out = out.permute(0, 1, 3, 4, 5, 2)
        return out
