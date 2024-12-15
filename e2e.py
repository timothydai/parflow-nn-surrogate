import argparse
from datetime import datetime
import os
os.environ["EXPERIMENT_SEED"] = "0"

import pandas as pd
import pathlib
import torch
import torchmetrics as tm
import tqdm

from data import time_series
from models import cnn_autoencoder


def test_e2e(args):
    assert args.name or args.name_exact, "Name this experiment."
    experiment_name = (
        args.name_exact
        or f'{args.name}_e2e_{datetime.now().strftime("%m%d%Y_%H%M%S%f")}'
    )
    save_dir = args.save_dir / experiment_name
    save_dir.mkdir(exist_ok=True, parents=True)
    with open(save_dir / "args.txt", "w") as f:
        f.write(str(args))

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using GPU")
    else:
        device = torch.device("cpu")

    print(
        "Loading checkpoint from...",
        args.stage1_ckpt,
        args.stage2_ckpt,
        args.stage3_ckpt,
    )
    stage1_model = torch.load(args.stage1_ckpt, map_location=device)["model"]
    stage1_model.eval()
    stage2_model = torch.load(args.stage2_ckpt, map_location=device)["model"]
    stage2_model.eval()
    stage3_model = torch.load(args.stage3_ckpt, map_location=device)["model"]
    stage3_model.eval()

    if args.use_dummy_dataset:
        stage1_test_dataset = time_series.DummyTimeSeriesDataset("stage1", device)
        stage2_test_dataset = time_series.DummyTimeSeriesDataset("stage2", device)
        stage3_test_dataset = time_series.DummyTimeSeriesDataset("stage3", device)
    else:
        stage1_xs, stage1_ys = time_series.get_xs_ys_from_data_dir(
            args.stage1_data_dir, "stage1"
        )
        stage1_test_dataset = time_series.TimeSeriesDataset(
            stage1_xs,
            stage1_ys,
            device,
            dataset_mode="test",
            normalization=args.normalization,
        )
        stage2_xs, stage2_ys = time_series.get_xs_ys_from_data_dir(
            args.stage2_data_dir, "stage2"
        )
        stage2_test_dataset = time_series.TimeSeriesDataset(
            stage2_xs,
            stage2_ys,
            device,
            dataset_mode="test",
            normalization=args.normalization,
        )
        stage3_xs, stage3_ys = time_series.get_xs_ys_from_data_dir(
            args.stage3_data_dir, "stage3"
        )
        stage3_test_dataset = time_series.TimeSeriesDataset(
            stage3_xs,
            stage3_ys,
            device,
            dataset_mode="test",
            normalization=args.normalization,
        )

    stage1_test_loader = torch.utils.data.DataLoader(stage1_test_dataset, batch_size=1)
    stage2_test_loader = torch.utils.data.DataLoader(stage2_test_dataset, batch_size=1)
    stage3_test_loader = torch.utils.data.DataLoader(stage3_test_dataset, batch_size=1)

    avg_pool_3d = torch.nn.AvgPool3d((20, 20, 1), (4, 4, 1))
    autoencoder = None
    if args.autoencoder_ckpt_path is None:
        # Use a dummy autoencoder with random weights.
        autoencoder = cnn_autoencoder.CNNAutoencoder()
        print("WARNING: Encoding targets with a random autoencoder.")
    else:
        autoencoder = torch.load(autoencoder_ckpt_path)["model"]
    autoencoder.eval()
    autoencoder.to(device)

    assert len(stage1_test_dataset) == len(stage2_test_dataset) and len(stage2_test_dataset) == len(stage3_test_dataset)
    print(f"Testing on {len(stage1_test_dataset)} sample(s).")
    with torch.no_grad():
        mape_by_t = [0 for _ in range(16 + 16 + 366)]
        mape_by_b = [0 for _ in range(len(stage1_test_dataset))]
        mae_by_b = [0 for _ in range(len(stage1_test_dataset))]

        for i, (
            (stage1_x, stage1_y),
            (stage2_x, stage2_y),
            (stage3_x, stage3_y),
        ) in enumerate(
            tqdm.tqdm(zip(stage1_test_loader, stage2_test_loader, stage3_test_loader), total=len(stage1_test_dataset))
        ):
            preds = []
            targets = []

            pred = stage1_model(stage1_x).contiguous()

            preds.append(pred)
            targets.append(stage1_y)

            # Replace 'initial pressure' input with prediction from previous stage.
            stage2_x[..., 7] = pred[..., 0]
            pred = stage2_model(stage2_x).contiguous()

            preds.append(pred)
            targets.append(stage2_y)

            # Replace 'initial pressure' input with prediction from previous stage.
            stage3_x[..., 7] = pred[..., 0]
            stage3_x = time_series.shrink_input(stage3_x, avg_pool_3d)
            pred = stage3_model(stage3_x).contiguous()
            pred = autoencoder.decode(pred)

            preds.append(pred)
            targets.append(stage3_y)

            preds = torch.concat(preds, axis=1)
            targets = torch.concat(targets, axis=1)
            for t in range(targets.shape[1]):
                mape_by_t[t] += tm.functional.mean_absolute_percentage_error(
                    preds[:, t], targets[:, t]
                ).item()
            mape_by_b[i] = tm.functional.mean_absolute_percentage_error(preds, targets).item()
            mae_by_b[i] = tm.functional.mean_absolute_error(preds, targets).item()

    mape_by_t = [m / len(stage1_test_dataset) for m in mape_by_t]
    pd.DataFrame(mape_by_t, columns=["mape"]).to_csv(
        save_dir / "mape_by_t.csv", index=False
    )

    pd.DataFrame(list(zip(mape_by_b, mae_by_b)), columns=["mape", "mae"]).to_csv(
        save_dir / "mape_by_b.csv", index=False
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    parser.add_argument("--name_exact", type=str)
    parser.add_argument(
        "--save_dir",
        type=pathlib.Path,
        help="Path to directory for saving E2E results.",
        default="checkpoints",
    )
    parser.add_argument("--stage1_ckpt", type=pathlib.Path, required=True)
    parser.add_argument("--stage2_ckpt", type=pathlib.Path, required=True)
    parser.add_argument("--stage3_ckpt", type=pathlib.Path, required=True)
    parser.add_argument(
        "--stage1_data_dir",
        type=pathlib.Path,
        default="data/sample_data",
    )
    parser.add_argument(
        "--stage2_data_dir",
        type=pathlib.Path,
        default="data/sample_data",
    )
    parser.add_argument(
        "--stage3_data_dir",
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
    test_e2e(args)

