import argparse
import glob
import os

os.environ["EXPERIMENT_SEED"] = "0"

import pathlib
import pandas as pd
import torch
import torchmetrics as tm
import tqdm

from data import time_series, single_frame
from models import cnn_autoencoder

import time


def test(args):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using GPU")
    else:
        device = torch.device("cpu")

    # Load model.
    assert args.ckpt is not None, "Must specify ckpt."
    print("Loading checkpoint from...", args.ckpt)
    model = torch.load(args.ckpt, map_location=device)["model"]
    model.eval()

    if args.use_dummy_dataset:
        if args.mode in ["stage1", "stage2", "stage3"]:
            test_dataset = time_series.DummyTimeSeriesDataset(args.mode, device)
        elif args.mode == "autoencoder":
            test_dataset = single_frame.DummySingleFrameDataset(device)
    else:
        if args.mode in ["stage1", "stage2", "stage3"]:
            xs, ys = time_series.get_xs_ys_from_data_dir(args.data_dir, args.mode)
            test_dataset = time_series.TimeSeriesDataset(
                xs,
                ys,
                device,
                dataset_mode="test",
                normalization=args.normalization,
            )
        if args.mode == "autoencoder":
            xs, ys = single_frame.get_xs_ys_from_data_dir(args.data_dir, args.mode)
            test_dataset = single_frame.SingleFrameDataset(
                xs,
                ys,
                device,
                dataset_mode="test",
                normalization=args.normalization,
            )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    avg_pool_3d = torch.nn.AvgPool3d((20, 20, 1), (4, 4, 1))
    autoencoder = None
    if args.mode == "stage3":
        if args.autoencoder_ckpt_path is None:
            # Use a dummy autoencoder with random weights.
            autoencoder = cnn_autoencoder.CNNAutoencoder()
            print("WARNING: Encoding targets with a random autoencoder.")
        else:
            autoencoder = torch.load(autoencoder_ckpt_path)["model"]
        autoencoder.eval()
        autoencoder.to(device)

    begin = time.time()
    print(f"Testing on {len(test_dataset)} sample(s).")
    with torch.no_grad():
        mape_by_t = [0 for _ in range(16 if args.mode != "stage3" else 365)]
        mape_by_b = [0 for _ in range(len(test_loader))]
        mae_by_b = [0 for _ in range(len(test_loader))]

        for i, (x, y) in enumerate(tqdm.tqdm(test_loader)):
            if args.mode == "stage3":
                x = time_series.shrink_input(x, avg_pool_3d)
            pred = model(x).contiguous()
            if args.mode == "stage3":
                pred = autoencoder.decode(pred)

            if args.mode != "autoencoder":
                # Autoencoders do not have a notion of time, since they are single-frame.
                for t in range(y.shape[1]):
                    mape_by_t[t] += tm.functional.mean_absolute_percentage_error(
                        pred[:, t], y[:, t]
                    ).item()

            mape_by_b[i] = tm.functional.mean_absolute_percentage_error(pred, y).item()
            mae_by_b[i] = tm.functional.mean_absolute_error(pred, y).item()

    runtime = time.time() - begin

    if args.mode != "autoencoder":
        mape_by_t = [m / len(test_loader) for m in mape_by_t]
        pd.DataFrame(mape_by_t, columns=["mape"]).to_csv(
            args.ckpt.parent / "mape_by_t.csv", index=False
        )
    pd.DataFrame(list(zip(mape_by_b, mae_by_b)), columns=["mape", "mae"]).to_csv(
        args.ckpt.parent / "mape_by_b.csv", index=False
    )
    pd.DataFrame(
        {
            "mape": [sum(mape_by_b) / len(mape_by_b)],
            "mae": [sum(mae_by_b) / len(mae_by_b)],
            "runtime": [runtime],
            "runtime_per_ex": [runtime / len(mape_by_b)],
        }
    ).to_csv(args.ckpt.parent / "test_stats.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=pathlib.Path, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["stage1", "stage2", "stage3", "autoencoder"],
        required=True,
    )
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        default="data/sample_data",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        choices=["none", "z_score", "min_max"],
        default="min_max",
    )
    parser.add_argument(
        "--autoencoder_ckpt_path",
        type=pathlib.Path,
    )
    parser.add_argument("--use_dummy_dataset", action="store_true")

    args = parser.parse_args()
    test(args)
