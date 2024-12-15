import os
import torch
import torch.nn as nn


torch.manual_seed(int(os.environ["EXPERIMENT_SEED"]))


class CNN3d(nn.Module):
    def __init__(self, num_input_variables, hidden_size):
        super(CNN3d, self).__init__()
        self.block0 = nn.Sequential(
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
        )
        self.proj = nn.Linear(hidden_size, 1)

        self.x_emb = nn.Linear(num_input_variables, hidden_size)
        # self.time_emb = nn.Linear(6, hidden_size)

        self.x_gradients = None
        self.x_time_gradients = None

    def forward(self, x):
        out = [x[:, 0, ..., -1:]]

        t_dim = x.shape[1]
        for t in range(t_dim):
            x_t = torch.concat([x[:, t, ..., :-1], out[-1]], -1)
            x_t = self.x_emb(x_t)

            x_t = x_t.permute(0, 4, 1, 2, 3)

            x_t = self.block0(x_t)

            x_t = x_t.permute(0, 2, 3, 4, 1)

            x_t = self.proj(x_t)
            out.append(x_t)
        return torch.stack(out[1:], 1)

    def x_grad_hook(self, grad):
        self.x_gradients = grad.detach().cpu()

    def x_time_grad_hook(self, grad):
        self.x_time_gradients = grad.detach().cpu()

    def get_tracked(self):
        return [self.x_gradients, self.x_time_gradients]


class CNN3dBN(nn.Module):
    def __init__(self, num_input_variables, hidden_size):
        super(CNN3dBN, self).__init__()
        self.block0 = nn.Sequential(
            torch.nn.Conv3d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=5,
                stride=1,
                padding=(5 - 1) // 2,
            ),
            torch.nn.BatchNorm3d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            torch.nn.Conv3d(
                in_channels=hidden_size,
                out_channels=hidden_size * 2,
                kernel_size=5,
                stride=1,
                padding=(5 - 1) // 2,
            ),
            torch.nn.BatchNorm3d(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            torch.nn.Conv3d(
                in_channels=hidden_size * 2,
                out_channels=hidden_size,
                kernel_size=5,
                stride=1,
                padding=(5 - 1) // 2,
            ),
            torch.nn.BatchNorm3d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.proj = nn.Linear(hidden_size, 1)

        self.x_emb = nn.Linear(num_input_variables, hidden_size)
        # self.time_emb = nn.Linear(6, hidden_size)

        self.x_gradients = None
        self.x_time_gradients = None

    def forward(self, x):
        out = [x[:, 0, ..., -1:]]

        t_dim = x.shape[1]
        for t in range(t_dim):
            x_t = torch.concat([x[:, t, ..., :-1], out[-1]], -1)
            x_t = self.x_emb(x_t)

            x_t = x_t.permute(0, 4, 1, 2, 3)

            x_t = self.block0(x_t)

            x_t = x_t.permute(0, 2, 3, 4, 1)

            x_t = self.proj(x_t)
            out.append(x_t)
        return torch.stack(out[1:], 1)

    def x_grad_hook(self, grad):
        self.x_gradients = grad.detach().cpu()

    def x_time_grad_hook(self, grad):
        self.x_time_gradients = grad.detach().cpu()

    def get_tracked(self):
        return [self.x_gradients, self.x_time_gradients]


class CNN3dSE(nn.Module):
    def __init__(self, num_input_variables, hidden_size):
        super(CNN3dSE, self).__init__()
        self.block0 = nn.Sequential(
            torch.nn.Conv3d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=5,
                stride=1,
                padding=(5 - 1) // 2,
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.SEblock0 = nn.Sequential(
            nn.Conv3d(hidden_size, hidden_size // 4, 1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv3d(hidden_size // 4, hidden_size, 1),
            nn.Sigmoid(),
        )

        self.block1 = nn.Sequential(
            torch.nn.Conv3d(
                in_channels=hidden_size,
                out_channels=hidden_size * 2,
                kernel_size=5,
                stride=1,
                padding=(5 - 1) // 2,
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.SEblock1 = nn.Sequential(
            nn.Conv3d(hidden_size * 2, hidden_size // 2, 1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv3d(hidden_size // 2, hidden_size * 2, 1),
            nn.Sigmoid(),
        )

        self.block2 = nn.Sequential(
            torch.nn.Conv3d(
                in_channels=hidden_size * 2,
                out_channels=hidden_size,
                kernel_size=5,
                stride=1,
                padding=(5 - 1) // 2,
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.SEblock2 = nn.Sequential(
            nn.Conv3d(hidden_size, hidden_size // 4, 1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv3d(hidden_size // 4, hidden_size, 1),
            nn.Sigmoid(),
        )
        self.proj = nn.Linear(hidden_size, 1)

        self.x_emb = nn.Linear(num_input_variables, hidden_size)
        self.time_emb = nn.Linear(6, hidden_size)

        self.x_gradients = None
        self.x_time_gradients = None
        self.excites = []

    def forward(self, x):
        self.excites = []
        x, time_enc = x

        if x.requires_grad:
            x.register_hook(self.x_grad_hook)
            time_enc.register_hook(self.x_time_grad_hook)

        x = self.x_emb(x) + self.time_emb(time_enc)

        x = x.permute(0, 4, 1, 2, 3)

        x = self.block0(x)
        x_squeeze = x.mean([2, 3, 4], keepdim=True)
        x_excite = self.SEblock0(x_squeeze)
        x = x * x_excite
        self.excites.append(x_excite)

        x = self.block1(x)
        x_squeeze = x.mean([2, 3, 4], keepdim=True)
        x_excite = self.SEblock1(x_squeeze)
        x = x * x_excite
        self.excites.append(x_excite)

        x = self.block2(x)
        x_squeeze = x.mean([2, 3, 4], keepdim=True)
        x_excite = self.SEblock2(x_squeeze)
        x = x * x_excite
        self.excites.append(x_excite)

        x = x.permute(0, 2, 3, 4, 1)

        x = self.proj(x)
        return x

    def x_grad_hook(self, grad):
        self.x_gradients = grad.detach().cpu()

    def x_time_grad_hook(self, grad):
        self.x_time_gradients = grad.detach().cpu()

    def get_excites(self):
        return self.excites

    def get_tracked(self):
        return [self.x_gradients, self.x_time_gradients]


class CNN3dSERes(nn.Module):
    def __init__(self, num_input_variables, hidden_size):
        super(CNN3dSERes, self).__init__()
        self.block0 = nn.Sequential(
            torch.nn.Conv3d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=5,
                stride=1,
                padding=(5 - 1) // 2,
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.SEblock0 = nn.Sequential(
            nn.Conv3d(hidden_size, hidden_size // 4, 1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv3d(hidden_size // 4, hidden_size, 1),
            nn.Sigmoid(),
        )

        self.block1 = nn.Sequential(
            torch.nn.Conv3d(
                in_channels=hidden_size,
                out_channels=hidden_size * 2,
                kernel_size=5,
                stride=1,
                padding=(5 - 1) // 2,
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.SEblock1 = nn.Sequential(
            nn.Conv3d(hidden_size * 2, hidden_size // 2, 1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv3d(hidden_size // 2, hidden_size * 2, 1),
            nn.Sigmoid(),
        )

        self.block2 = nn.Sequential(
            torch.nn.Conv3d(
                in_channels=hidden_size * 2,
                out_channels=hidden_size,
                kernel_size=5,
                stride=1,
                padding=(5 - 1) // 2,
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.SEblock2 = nn.Sequential(
            nn.Conv3d(hidden_size, hidden_size // 4, 1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv3d(hidden_size // 4, hidden_size, 1),
            nn.Sigmoid(),
        )

        self.res = nn.Conv3d(hidden_size, hidden_size, 1)

        self.proj = nn.Linear(hidden_size, 1)

        self.x_emb = nn.Linear(num_input_variables, hidden_size)
        self.time_emb = nn.Linear(6, hidden_size)

        self.x_gradients = None
        self.x_time_gradients = None
        self.excites = []

    def forward(self, x):
        self.excites = []
        x, time_enc = x

        if x.requires_grad:
            x.register_hook(self.x_grad_hook)
            time_enc.register_hook(self.x_time_grad_hook)

        x = self.x_emb(x) + self.time_emb(time_enc)

        x = x.permute(0, 4, 1, 2, 3)

        res = self.res(x)

        x = self.block0(x)
        x_squeeze = x.mean([2, 3, 4], keepdim=True)
        x_excite = self.SEblock0(x_squeeze)
        x = x * x_excite
        self.excites.append(x_excite)

        x = self.block1(x)
        x_squeeze = x.mean([2, 3, 4], keepdim=True)
        x_excite = self.SEblock1(x_squeeze)
        x = x * x_excite
        self.excites.append(x_excite)

        x = self.block2(x)
        x_squeeze = x.mean([2, 3, 4], keepdim=True)
        x_excite = self.SEblock2(x_squeeze)
        x = x * x_excite
        self.excites.append(x_excite)

        x = x + res

        x = x.permute(0, 2, 3, 4, 1)

        x = self.proj(x)
        return x

    def x_grad_hook(self, grad):
        self.x_gradients = grad.detach().cpu()

    def x_time_grad_hook(self, grad):
        self.x_time_gradients = grad.detach().cpu()

    def get_excites(self):
        return self.excites

    def get_tracked(self):
        return [self.x_gradients, self.x_time_gradients]


class CNN3dSEResTime(nn.Module):
    def __init__(self, num_input_variables, hidden_size):
        super(CNN3dSEResTime, self).__init__()
        self.block0 = nn.Sequential(
            torch.nn.Conv3d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=5,
                stride=1,
                padding=(5 - 1) // 2,
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.SEblock0 = nn.Sequential(
            nn.Conv3d(hidden_size, hidden_size // 4, 1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv3d(hidden_size // 4, hidden_size, 1),
            nn.Sigmoid(),
        )

        self.block1 = nn.Sequential(
            torch.nn.Conv3d(
                in_channels=hidden_size,
                out_channels=hidden_size * 2,
                kernel_size=5,
                stride=1,
                padding=(5 - 1) // 2,
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.SEblock1 = nn.Sequential(
            nn.Conv3d(hidden_size * 2, hidden_size // 2, 1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv3d(hidden_size // 2, hidden_size * 2, 1),
            nn.Sigmoid(),
        )

        self.block2 = nn.Sequential(
            torch.nn.Conv3d(
                in_channels=hidden_size * 2,
                out_channels=hidden_size,
                kernel_size=5,
                stride=1,
                padding=(5 - 1) // 2,
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.SEblock2 = nn.Sequential(
            nn.Conv3d(hidden_size, hidden_size // 4, 1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv3d(hidden_size // 4, hidden_size, 1),
            nn.Sigmoid(),
        )

        self.res_gate = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())

        self.proj = nn.Linear(hidden_size, 1)
        self.proj1 = nn.Linear(1, 1)

        self.x_emb = nn.Linear(num_input_variables, hidden_size)
        self.time_emb = nn.Linear(6, hidden_size)

        self.x_gradients = None
        self.x_time_gradients = None
        self.excites = []

    def forward(self, x):
        self.excites = []
        x, time_enc = x

        if x.requires_grad:
            x.register_hook(self.x_grad_hook)
            time_enc.register_hook(self.x_time_grad_hook)

        res = x[..., -1:]
        res_gate = self.res_gate(time_enc[..., -1:])

        x = self.x_emb(x) + self.time_emb(time_enc)

        x = x.permute(0, 4, 1, 2, 3)

        x = self.block0(x)
        x_squeeze = x.mean([2, 3, 4], keepdim=True)
        x_excite = self.SEblock0(x_squeeze)
        x = x * x_excite
        self.excites.append(x_excite)

        x = self.block1(x)
        x_squeeze = x.mean([2, 3, 4], keepdim=True)
        x_excite = self.SEblock1(x_squeeze)
        x = x * x_excite
        self.excites.append(x_excite)

        x = self.block2(x)
        x_squeeze = x.mean([2, 3, 4], keepdim=True)
        x_excite = self.SEblock2(x_squeeze)
        x = x * x_excite
        self.excites.append(x_excite)

        x = x.permute(0, 2, 3, 4, 1)

        x = self.proj(x)
        x = (1 - res_gate) * x + (res_gate) * res
        x = self.proj1(x)
        return x

    def x_grad_hook(self, grad):
        self.x_gradients = grad.detach().cpu()

    def x_time_grad_hook(self, grad):
        self.x_time_gradients = grad.detach().cpu()

    def get_excites(self):
        return self.excites

    def get_tracked(self):
        return [self.x_gradients, self.x_time_gradients]


class CNN3dResTime(nn.Module):
    def __init__(self, num_input_variables, hidden_size):
        super(CNN3dResTime, self).__init__()

        self.block0 = nn.Sequential(
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
        )
        self.res_gate = nn.Sequential(nn.Linear(6, hidden_size), nn.Sigmoid())

        self.proj = nn.Linear(hidden_size, 1)

        self.x_emb = nn.Linear(num_input_variables, hidden_size)
        self.time_emb = nn.Linear(6, hidden_size)

        self.x_gradients = None
        self.x_time_gradients = None
        self.excites = []

    def forward(self, x):
        self.excites = []
        x, time_enc = x

        if x.requires_grad:
            x.register_hook(self.x_grad_hook)
            time_enc.register_hook(self.x_time_grad_hook)

        x = self.x_emb(x) + self.time_emb(time_enc)

        res_gate = self.res_gate(time_enc)
        res = x

        x = x.permute(0, 4, 1, 2, 3)

        x = self.block0(x)

        x = x.permute(0, 2, 3, 4, 1)

        x = res * res_gate + x * (1 - res_gate)

        x = self.proj(x)
        return x

    def x_grad_hook(self, grad):
        self.x_gradients = grad.detach().cpu()

    def x_time_grad_hook(self, grad):
        self.x_time_gradients = grad.detach().cpu()

    def get_excites(self):
        return self.excites

    def get_tracked(self):
        return [self.x_gradients, self.x_time_gradients]
