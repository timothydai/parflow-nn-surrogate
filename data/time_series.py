import os
import pathlib

import numpy as np
import pandas as pd
from sklearn import model_selection
import torch
import torchvision
import tqdm

from data import utils


def get_xs_ys_from_data_dir(data_dir, mode):
    num_ts_to_sample = 365 if mode == "stage3" else 16
    simulation_dirs = list(data_dir.glob("*"))
    print("Collecting data...")
    xs, ys = [], []
    for simulation_dir in tqdm.tqdm(simulation_dirs):
        y, initial_pressure, ts = utils.collect_targets_from_one_simulation(
            simulation_dir, num_ts_to_sample=num_ts_to_sample
        )
        x = utils.collect_static_inputs_from_one_simulation(
            simulation_dir, initial_pressure
        )
        # [x, y, z, c] --> [t, x, y, z, c].
        x = x.unsqueeze(0)
        x = x.tile([y.shape[0], 1, 1, 1, 1])

        if mode in ["stage2", "stage3"]:
            x_dynamic = utils.collect_dynamic_inputs_from_one_simulation(
                simulation_dir, ts, mode
            )
            # [t, c] --> [t, x, y, z, c].
            x_dynamic = x_dynamic[:, None, None, None, :]
            _, x_dim, y_dim, z_dim, _ = x.shape
            x_dynamic = x_dynamic.tile([1, x_dim, y_dim, z_dim, 1])
            x = torch.concat([x, x_dynamic], axis=-1)

        # x: [t, x, y, z, c].
        # y: [t, x, y, z, c].
        xs.append(x)
        ys.append(y)

    # List[t, x, y, z, c] --> [b, t, x, y, z, c].
    xs = torch.stack(xs, axis=0)
    ys = torch.stack(ys, axis=0)
    return xs.float(), ys.float()


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        xs,
        ys,
        device,
        dataset_mode="train",
        normalization="none",
        train_set_frac=1.0,
        encode_inputs=False,
        encode_targets=False,
        autoencoder_ckpt_path=None,
    ):
        assert xs.shape[0] == ys.shape[0]
        sample_bs = range(len(xs))
        train_bs, test_bs = model_selection.train_test_split(
            sample_bs, test_size=0.1, random_state=0
        )
        if train_set_frac < 1.0:
            _, train_bs = model_selection.train_test_split(
                train_bs,
                test_size=train_set_frac,
                random_state=int(os.environ["EXPERIMENT_SEED"]),
            )
            print(f"Training set reduced to {len(train_bs)}.")

        train_bs, val_bs = model_selection.train_test_split(
            train_bs,
            test_size=1 / 9,
            random_state=int(os.environ["EXPERIMENT_SEED"]),
        )

        self._xs, self._ys = None, None
        if dataset_mode == "train":
            self._xs = xs[train_bs]
            self._ys = ys[train_bs]
        elif dataset_mode == "val":
            self._xs = xs[val_bs]
            self._ys = ys[val_bs]
        elif dataset_mode == "test":
            self._xs = xs[test_bs]
            self._ys = ys[test_bs]

        if normalization == "z_score":
            train_xs = xs[train_bs]
            train_ys = ys[train_bs]

            train_xs_mean = train_xs.mean(dim=[0, 1, 2, 3, 4], keepdim=True)
            train_xs_std = train_xs.std(dim=[0, 1, 2, 3, 4], keepdim=True)

            train_ys_mean = train_ys.mean(dim=[0, 1, 2, 3, 4], keepdim=True)
            train_ys_std = train_ys.std(dim=[0, 1, 2, 3, 4], keepdim=True)

            self._xs = (self._xs - train_xs_mean) / train_xs_std
            self._ys = (self._ys - train_ys_mean) / train_ys_std
        elif normalization == "min_max":
            train_xs = xs[train_bs]
            train_ys = ys[train_bs]

            train_xs_min = torch.amin(train_xs, dim=[0, 1, 2, 3, 4], keepdim=True)
            train_xs_max = torch.amax(train_xs, dim=[0, 1, 2, 3, 4], keepdim=True)

            train_ys_min = torch.amin(train_ys, dim=[0, 1, 2, 3, 4], keepdim=True)
            train_ys_max = torch.amax(train_ys, dim=[0, 1, 2, 3, 4], keepdim=True)

            self._xs = (self._xs - train_xs_min) / (train_xs_max - train_xs_min)
            self._ys = (self._ys - train_ys_min) / (train_ys_max - train_ys_min)

        self._device = device

    def __len__(self):
        return len(self._xs)

    def __getitem__(self, idx):
        x = self._xs[idx].to(self._device)
        y = self._ys[idx].to(self._device)
        return x, y


def shrink_input(x, avg_pool_3d):
    _b, _t, _x, _y, _z, _c = x.shape
    x = x.reshape(-1, _x, _y, _z, _c)
    x = x.permute(0, -1, 1, 2, 3)
    x = avg_pool_3d(x)
    x = x.permute(0, 2, 3, 4, 1)
    _, _x, _y, _z, _ = x.shape
    x = x.reshape(_b, _t, _x, _y, _z, _c)
    return x


class DummyTimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, mode, device):
        if mode == "stage1":
            self._x = torch.rand(1, 16, 120, 80, 25, 8)
            self._y = torch.rand(1, 16, 120, 80, 25, 1)
        elif mode == "stage2":
            self._x = torch.rand(1, 16, 120, 80, 25, 16)
            self._y = torch.rand(1, 16, 120, 80, 25, 1)
        elif mode == "stage3":
            self._x = torch.rand(1, 365, 120, 80, 25, 17)
            self._y = torch.rand(1, 365, 120, 80, 25, 1)
        self._device = device

    def __len__(self):
        return len(self._x)

    def __getitem__(self, idx):
        return self._x[idx].to(self._device), self._y[idx].to(self._device)
