# Code and data for "Machine learning surrogates for efficient hydrologic modeling: Insights from stochastic simulations of managed aquifer recharge"

## Overview

This repository contains the code for the paper "Machine Learning Surrogates for Efficient Hydrologic Modeling: Insights from Stochastic Simulations of Managed Aquifer Recharge" by [Dai et al. (2025)](https://doi.org/10.1016/j.jhydrol.2024.132606) in the Journal of Hydrology.
The study evaluates a hybrid modeling framework that combines process-based hydrologic simulations (with the integrated hydrologic code ParFlow-CLM) and machine learning (ML) surrogates to efficiently simulate managed aquifer recharge.

This repository is organized as follows:

1. `data/sample_data` contains sample output for all three simulation stages and sample data for autoencoder training.
Instructions for unzipping the data are provided in the Installation section below.
2. `data` also contains PyTorch dataset modules and utility functions that construct PyTorch tensors from raw ParFlow-CLM outputs.
3. `models` contains PyTorch implementations of the 8 surrogate architectures used in the study (CNN3d, CNN4d, U-FNO3d, U-FNO4d, ViT3d, ViT4d, PredRNN++, and a CNN autoencoder).
4. `layers` contains custom PyTorch layers used in some of the surrogate architectures above.
5. `losses` contains a PyTorch implementation of the normalized $L^p$-norm used as a loss function in this study.
6. Finally, the base directory contains scripts to train and evaluate each surrogate architecture.

## Installation

Install all required modules with `pip install -r requirements.txt`.
For complete compatibility, create your virtual environment with Python 3.8.20.
Other versions of Python have not been tested but may also work.

### Get sample data

Unzip the sample data and set up the data directory hierarchy with `sh data/sample_data/unzip_all.sh`.
For users who wish to train on the complete dataset used in the paper, ParFlow output files are available to the public in a [separate repository](https://doi.org/10.25740/hj302gv2126).

Any external data, provided through the `--data_dir` option, must have its directory hierarchy structured similarly to the sample data.

## Training

All surrogate architectures described in the paper can be trained using the `train.py` script.
The script uses `argparse` to take in several command line arguments to specify the model, dataset and hyperparameters.
To view all command-line options, run `python train.py --help`.

### To train an autoencoder

```python
python train.py --name <name> \
    --mode autoencoder \
    --model CNNAutoencoder \
    --data_dir data/sample_data/autoencoder \
    [--OPTIONS]
```

where `<name>` is the name of the experiment (e.g., `my_first_autoencoder`) and `CNNAutoencoder` is the architecture to be used.

### To train a Stage 1 surrogate

```python
python train.py --name <name> \
    --mode stage1 \
    --model <model> \
    --data_dir data/sample_data/stage1 \
    [--OPTIONS]
```

where `<name>` is the name of the experiment and `<model>` is the name of the architecture to be used.
Note that the `--model` option must be one of the following: `CNN3d`, `CNN4d`, `PredRNN`, `UFNO3d`, `UFNO4d`, `ViT3d` or `ViT4d`.
All other options can be viewed with `python train.py --help`.

### To train a Stage 2 surrogate

```python
python train.py --name <name> \
    --mode stage2 \
    --model <model> \
    --data_dir data/sample_data/stage2 \
    [--OPTIONS]
```

### To train a Stage 3 surrogate

```python
python train.py --name <name> \
    --mode stage3 \
    --model <model> \
    --data_dir data/sample_data/stage3 \
    --autoencoder_ckpt_path <autoencoder_ckpt_path> \
    [--OPTIONS]
```

Instead of providing an autoencoder checkpoint in Stage 3 training, users can also use a randomly initialized autoencoder by omitting the `--autoencoder_ckpt_path` option.

Notable options:

* Start `--name` with "test" to run without saving checkpoints or tensorboard data.
* Use the `--use_dummy_dataset` flag to quickly load correctly sized but randomly initialized tensors.

## Evaluation

Testing occurs automatically at the end of training when the `--train_only` flag is not set.
However, testing can also be initiated separately with the commands below.

### To test an autoencoder

```python
python test.py \
    --mode autoencoder \
    --ckpt <ckpt> \
    --data_dir data/sample_data/autoencoder \
    [--OPTIONS]
```

### To test a Stage 1 surrogate

```python
python test.py \
    --mode stage1 \
    --ckpt <ckpt> \
    --data_dir data/sample_data/stage1 \
    [--OPTIONS]
```

### To test a Stage 2 surrogate

```python
python test.py \
    --mode stage2 \
    --ckpt <ckpt> \
    --data_dir data/sample_data/stage2 \
    [--OPTIONS]
```

### To test a Stage 3 surrogate

```python
python test.py \
    --mode stage3 \
    --ckpt <ckpt> \
    --data_dir data/sample_data/stage3 \
    [--OPTIONS]
```

### End-to-end (E2E) evaluation

Three checkpoints can be tested together in an end-to-end fashion using the following command:

```python
python e2e.py \
  --name <name> \
  --stage1_ckpt <stage1_ckpt> \
  --stage2_ckpt <stage2_ckpt> \
  --stage3_ckpt <stage3_ckpt> \
  --stage1_data_dir data/sample_data/stage1 \
  --stage2_data_dir data/sample_data/stage2 \
  --stage3_data_dir data/sample_data/stage3 \
  --autoencoder_ckpt_path <autoencoder_ckpt_path> \
  [--OPTIONS]
```
