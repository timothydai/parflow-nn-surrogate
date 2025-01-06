import os
import pathlib

import numpy as np
import pandas as pd
from sklearn import model_selection
import torch
import tqdm

from data import utils


def get_xs_ys_from_data_dir(data_dir, mode):
    simulation_dirs = [path for path in list(data_dir.glob("*")) if path.is_dir()]
    print("Collecting data...")
    ys = []
    for simulation_dir in tqdm.tqdm(simulation_dirs):
        y, initial_pressure, ts = utils.collect_targets_from_one_simulation(
            simulation_dir, mode, num_ts_to_sample=100
        )  # [t, x, y, z, c].
        y = torch.concat([initial_pressure[None, ..., None], y], axis=0)
        ys.append(y)

    # List[t, x, y, z, c] --> [b, t, x, y, z, c].
    ys = torch.concat(ys, axis=0)
    xs, ys = ys[:-1], ys[1:]
    return xs.float(), ys.float()


class SingleFrameDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        xs,
        ys,
        device,
        dataset_mode="train",
        normalization="none",
    ):
        assert xs.shape[0] == ys.shape[0]
        sample_bs = range(len(xs))
        train_bs, test_bs = model_selection.train_test_split(
            sample_bs, test_size=0.1, random_state=0
        )
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

            train_xs_mean = train_xs.mean(dim=[0, 1, 2, 3], keepdim=True)
            train_xs_std = train_xs.std(dim=[0, 1, 2, 3], keepdim=True)

            train_ys_mean = train_ys.mean(dim=[0, 1, 2, 3], keepdim=True)
            train_ys_std = train_ys.std(dim=[0, 1, 2, 3], keepdim=True)

            self._xs = (self._xs - train_xs_mean) / train_xs_std
            self._ys = (self._ys - train_ys_mean) / train_ys_std
        elif normalization == "min_max":
            train_xs = xs[train_bs]
            train_ys = ys[train_bs]

            train_xs_min = torch.amin(train_xs, dim=[0, 1, 2, 3], keepdim=True)
            train_xs_max = torch.amax(train_xs, dim=[0, 1, 2, 3], keepdim=True)

            train_ys_min = torch.amin(train_ys, dim=[0, 1, 2, 3], keepdim=True)
            train_ys_max = torch.amax(train_ys, dim=[0, 1, 2, 3], keepdim=True)

            self._xs = (self._xs - train_xs_min) / (train_xs_max - train_xs_min)
            self._ys = (self._ys - train_ys_min) / (train_ys_max - train_ys_min)

        self._device = device

    def __len__(self):
        return len(self._xs)

    def __getitem__(self, idx):
        x = self._xs[idx].to(self._device)
        y = self._ys[idx].to(self._device)
        return x, y


class DummySingleFrameDataset(torch.utils.data.Dataset):
    def __init__(self, device):
        self._x = torch.rand(1, 120, 80, 25, 1)
        self._y = torch.rand(1, 120, 80, 25, 1)
        self._device = device

    def __len__(self):
        return len(self._x)

    def __getitem__(self, idx):
        return self._x[idx].to(self._device), self._y[idx].to(self._device)
