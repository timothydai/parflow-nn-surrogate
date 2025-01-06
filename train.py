import argparse
from datetime import datetime
import os
import pathlib
import time

import torch
import torchmetrics as tm
import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

# Set seed before any relative imports.
os.environ["EXPERIMENT_SEED"] = "0"
from data import single_frame, time_series
from losses.lploss import LpLoss
from models import (
    cnn_autoencoder,
    cnn3d,
    cnn4d,
    predrnn,
    ufno3d,
    ufno4d,
    vit3d,
    vit4d,
)
import test


def train(args):
    # Set experiment name.
    assert args.name or args.name_exact, "Name this experiment."
    experiment_name = (
        args.name_exact
        or f'{args.name}_{args.mode}_{datetime.now().strftime("%m%d%Y_%H%M%S%f")}'
    )

    is_dry_run = experiment_name.startswith("test")
    if is_dry_run:
        print("Dry run: not writing to Tensorboard or saving checkpoints.")
    else:
        # Set experiment checkpoint path.
        print("Starting experiment at", experiment_name)
        save_dir = args.checkpoint_dir / experiment_name
        save_dir.mkdir(exist_ok=True, parents=True)
        with open(save_dir / "args.txt", "w") as f:
            f.write(str(args))

    # Set device.
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Set model params dependent on simulation stage.
    if args.mode == "stage1":
        num_input_variables = 8
    elif args.mode == "stage2":
        num_input_variables = 16
    elif args.mode == "stage3":
        num_input_variables = 17

    if args.ckpt:
        print("Loading checkpoint from", args.ckpt)
        ckpt = torch.load(args.ckpt)
        model = ckpt["model"]
        epoch_start = ckpt["epoch"]
        optimizer = ckpt["optimizer"]
        scheduler = ckpt["scheduler"]
    else:
        if args.model == "CNN4d":
            model = cnn4d.CNN4d(
                num_input_variables=num_input_variables,
                hidden_size=args.hidden_size or 64,
            )
        elif args.model == "CNN3d":
            model = cnn3d.CNN3d(
                num_input_variables=num_input_variables,
                hidden_size=args.hidden_size or 64,
            )
        elif args.model == "PredRNN":
            model = predrnn.PredRNN(
                num_input_variables=num_input_variables,
                hidden_size=args.hidden_size or 32,
                num_layers=4,
            )
        elif args.model == "UFNO3d":
            model = ufno3d.UFNO3d(
                num_input_variables=num_input_variables,
                modes1=10,
                modes2=10,
                modes3=10,
                width=args.hidden_size or 32,
                is_stage_3=args.mode == "stage3",
            )
        elif args.model == "UFNO4d":
            model = ufno4d.UFNO4d(
                num_input_variables=num_input_variables,
                num_F_layers=3,
                num_UF_layers=3,
                modes1=10,
                modes2=10,
                modes3=10,
                modes4=10,
                width=args.hidden_size or 32,
                is_stage_3=args.mode == "stage3",
            )
        elif args.model == "ViT4d":
            model = vit4d.ViT4d(
                num_input_variables=num_input_variables,
                hidden_size=args.hidden_size or 32,
                num_heads=8,
                num_layers=8,
                is_stage_3=args.mode == "stage3",
            )
        elif args.model == "ViT3d":
            model = vit3d.ViT3d(
                num_input_variables=num_input_variables,
                hidden_size=args.hidden_size or 32,
                num_heads=8,
                num_layers=8,
                is_stage_3=args.mode == "stage3",
            )
        elif args.model == "CNNAutoencoder":
            model = cnn_autoencoder.CNNAutoencoder()
            assert args.mode == "autoencoder"
        else:
            exit(args.model + " is not a valid option.")
        print("Using", args.model)
        epoch_start = 0
        optimizer = None
        scheduler = None
    print(
        "Number of trainable parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    model = model.to(device)

    # Create data loaders.
    if args.use_dummy_dataset:
        if args.mode in ["stage1", "stage2", "stage3"]:
            train_dataset = time_series.DummyTimeSeriesDataset(
                args.mode,
                device,
            )
            val_dataset = time_series.DummyTimeSeriesDataset(
                args.mode,
                device,
            )
        elif args.mode == "autoencoder":
            train_dataset = single_frame.DummySingleFrameDataset(
                device,
            )
            val_dataset = single_frame.DummySingleFrameDataset(
                device,
            )
    elif args.data_dir:
        if args.mode in ["stage1", "stage2", "stage3"]:
            xs, ys = time_series.get_xs_ys_from_data_dir(args.data_dir, args.mode)
            train_dataset = time_series.TimeSeriesDataset(
                xs,
                ys,
                device,
                dataset_mode="train",
                normalization=args.normalization,
                train_set_frac=args.train_set_frac,
            )
            val_dataset = time_series.TimeSeriesDataset(
                xs,
                ys,
                device,
                dataset_mode="val",
                normalization=args.normalization,
                train_set_frac=args.train_set_frac,
            )
        if args.mode == "autoencoder":
            xs, ys = single_frame.get_xs_ys_from_data_dir(args.data_dir, args.mode)
            train_dataset = single_frame.SingleFrameDataset(
                xs,
                ys,
                device,
                dataset_mode="train",
                normalization=args.normalization,
            )
            val_dataset = single_frame.SingleFrameDataset(
                xs,
                ys,
                device,
                dataset_mode="val",
                normalization=args.normalization,
            )
    else:
        raise ValueError("Must either provide args.data_dir or args.use_dummy_dataset.")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)

    optimizer = optimizer or torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = scheduler or torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma
    )

    avg_pool_3d = torch.nn.AvgPool3d((20, 20, 1), (4, 4, 1))
    autoencoder = None
    if args.mode == "stage3":
        if args.autoencoder_ckpt_path is None:
            # Use a dummy autoencoder with random weights.
            autoencoder = cnn_autoencoder.CNNAutoencoder()
            print("WARNING: Encoding targets with a random autoencoder.")
        else:
            autoencoder = torch.load(args.autoencoder_ckpt_path)["model"]
        autoencoder.eval()
        autoencoder.to(device)

    # Define loss.
    if args.loss == "LpLoss":
        loss_fn = LpLoss(size_average=False)
    elif args.loss == "MSELoss":
        loss_fn = torch.nn.MSELoss()
    elif args.loss == "L1Loss":
        loss_fn = torch.nn.L1Loss()
    elif args.loss == "CrossEntropyLoss":
        loss_fn = torch.nn.CrossEntropyLoss()

    num_train_batches = len(train_loader)
    num_val_batches = len(val_loader)
    print(
        "Number of training and validation examples:",
        num_train_batches,
        num_val_batches,
    )
    print("Starting training...")
    writer = None
    if not is_dry_run:
        writer = SummaryWriter(log_dir=save_dir)
    best_val_loss = None
    start_time = time.time()
    for ep in range(epoch_start, args.epochs):
        t_counter = 0
        v_counter = 0

        t_loss = 0
        v_loss = 0

        t_mae = 0
        t_mape = 0
        t_mse = 0

        v_mae = 0
        v_mape = 0
        v_mse = 0

        model.train()
        train_bar = tqdm.tqdm(train_loader)
        for i, (x, y) in enumerate(train_bar):
            optimizer.zero_grad()

            if args.mode == "stage3":
                x = time_series.shrink_input(x, avg_pool_3d)
                with torch.no_grad():
                    y = autoencoder.encode(y)
            pred = model(x).contiguous()

            loss = loss_fn(pred, y)
            loss.backward()

            optimizer.step()
            t_loss += loss.item()

            t_mae += tm.functional.mean_absolute_error(pred, y).item()
            t_mape += tm.functional.mean_absolute_percentage_error(pred, y).item()
            t_mse += tm.functional.mean_squared_error(pred, y).item()

            t_counter += 1

            train_bar.set_description(
                f"Epoch: {ep}, Batch: {i + 1}/{num_train_batches}, Train Loss: {(t_loss / t_counter):.5f}"
            )

        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                if args.mode == "stage3":
                    x = time_series.shrink_input(x, avg_pool_3d)
                pred = model(x).contiguous()
                if args.mode == "stage3":
                    pred = autoencoder.decode(pred)

                v_loss += loss_fn(pred, y).item()

                v_mae += tm.functional.mean_absolute_error(pred, y).item()
                v_mape += tm.functional.mean_absolute_percentage_error(pred, y).item()
                v_mse += tm.functional.mean_squared_error(pred, y).item()

                v_counter += 1

        scheduler.step()

        if writer is not None:
            writer.add_scalar(
                "t_loss", t_loss / num_train_batches / args.batch_size, ep
            )
            writer.add_scalar("v_loss", v_loss / num_val_batches, ep)
            writer.add_scalar("t_mae", t_mae / t_counter, ep)
            writer.add_scalar("t_mape", t_mape / t_counter, ep)
            writer.add_scalar("t_mse", t_mse / t_counter, ep)
            writer.add_scalar("v_mae", v_mae / v_counter, ep)
            writer.add_scalar("v_mape", v_mape / v_counter, ep)
            writer.add_scalar("v_mse", v_mse / v_counter, ep)

        # Save checkpoint.
        val_message = ""
        if not is_dry_run:
            state_to_save = {
                "epoch": ep,
                "model": model,
                "optimizer": optimizer,
                "scheduler": scheduler,
            }
            torch.save(state_to_save, os.path.join(save_dir, f"ep_{ep}.pt"))
        if best_val_loss is None or v_loss / num_val_batches < best_val_loss:
            if best_val_loss is None:
                val_message = f"Validation loss improved from inf to {(v_loss / num_val_batches):.5f} "
            else:
                val_message = f"Validation loss improved from {best_val_loss:.5f} to {(v_loss / num_val_batches):.5f}"

            if args.alt_display_metric == "MAPE":
                val_message += f" (VMAPE: {(v_mape / v_counter):.5f})"
            elif args.alt_display_metric == "MAE":
                val_message += f" (VMAE: {(v_mae / v_counter):.5f})"

            if not is_dry_run:
                best_state_to_save = {
                    "epoch": ep,
                    "model": model,
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                }
                torch.save(best_state_to_save, save_dir / "best_checkpoint.pt")
            best_val_loss = v_loss / num_val_batches
            end_time = time.time()
        else:
            val_message = "Validation loss did not improve"
        print(val_message)

    if not is_dry_run:
        with open(save_dir / "train_time_for_best_ckpt.txt", "w") as f:
            f.write(f"{end_time - start_time} seconds")
        return save_dir / "best_checkpoint.pt"
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    parser.add_argument("--name_exact", type=str)
    parser.add_argument(
        "--checkpoint_dir",
        type=pathlib.Path,
        help="Path to directory for saving checkpoints.",
        default="checkpoints",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["stage1", "stage2", "stage3", "autoencoder"],
        required=True,
    )
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--train_only",
        action="store_true",
        help="If flag is on, do not test checkpoint after training",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "CNN3d",
            "CNN4d",
            "PredRNN",
            "UFNO3d",
            "UFNO4d",
            "ViT3d",
            "ViT4d",
            "CNNAutoencoder",
        ],
    )
    parser.add_argument(
        "--normalization",
        type=str,
        choices=["none", "z_score", "min_max"],
        default="min_max",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="LpLoss",
        choices=["LpLoss", "MSELoss", "L1Loss", "CrossEntropyLoss"],
    )
    parser.add_argument("--ckpt", type=pathlib.Path)
    parser.add_argument(
        "--train_set_frac",
        type=float,
        default=1.0,
        help="Fraction of training set to keep. Default 1.0 means that all training samples are kept.",
    )
    parser.add_argument("--hidden_size", type=int)
    parser.add_argument("--use_dummy_dataset", action="store_true")
    parser.add_argument(
        "--alt_display_metric",
        type=str,
        choices=["MAPE", "MAE"],
        default="MAPE",
        help="Which performance metric to display during training.",
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--scheduler_step", type=int, default=4)
    parser.add_argument("--scheduler_gamma", type=float, default=1)
    parser.add_argument(
        "--autoencoder_ckpt_path",
        type=pathlib.Path,
    )

    args = parser.parse_args()
    args.ckpt = train(args)
    if args.ckpt and not args.train_only:
        test.test(args)
