# Machine learning surrogates for efficient hydrologic modeling: Insights from stochastic simulations of managed aquifer recharge

**Authors**: Timothy Dai, Kate Maher, Zach Perzan

**Paper**: Accepted for publication in *Journal of Hydrology*. DOI to be assigned.

## Installation

Install all required modules with `pip install -r requirements.txt`.

### Get sample data

In `data/sample_data/`, we provide sample simulations that can serve as sample data for any of the three stages.
Unzip the data with `sh data/sample_data/unzip_all.sh`.
Note that by default, `--data_dir` in all scripts point to `data/sample_data`.
Any external data, provided through the `--data_dir` option, must be formatted similarly to `data/sample_data`.

## Training commands

To train an autoencoder:
```python
python train.py --name <name> --mode autoencoder --model CNNAutoencoder [--OPTIONS]
```

To train a Stage 1 surrogate:
```python
python train.py --name <name> --mode stage1 --model <model> [--OPTIONS]
```

To train a Stage 2 surrogate:
```python
python train.py --name <name> --mode stage2 --model <model> [--OPTIONS]
```

To train a Stage 3 surrogate:
```python
python train.py --name <name> --mode stage3 --model <model> --autoencoder_ckpt_path <autoencoder_ckpt_path> [--OPTIONS]
```

Notable options:

* Start `--name` with "test" to run without saving checkpoints or tensorboard data.
* The `--use_dummy_dataset` flag is provided to quickly load correctly sized but randomly initialized tensors.
* Instead of providing an autoencoder checkpoint in Stage 3 training, users can also use a randomly initialized autoencoder by simply omitting the `--autoencoder_ckpt_path` option.

## Evaluation commands

Testing occurs automatically at the end of training when the `--train_only` flag is not set.
However, testing can also be initiated separately with the commands below.

To test an autoencoder:
```python
python test.py --mode autoencoder --ckpt <ckpt> [--OPTIONS]
```

To test a Stage 1 surrogate:
```python
python test.py --mode stage1 --ckpt <ckpt> [--OPTIONS]
```

To test a Stage 2 surrogate:
```python
python test.py --mode stage2 --ckpt <ckpt> [--OPTIONS]
```

To test a Stage 3 surrogate:
```python
python test.py --mode stage3 --ckpt <ckpt> [--OPTIONS]
```

### E2E evaluation command

Three checkpoints can be tested together in an end-to-end fashion using the following command:

```python
python e2e.py \
  --name <name> \
  --stage1_ckpt <stage1_ckpt> \
  --stage2_ckpt <stage2_ckpt> \
  --stage3_ckpt <stage3_ckpt> \
  --stage1_data_dir <stage1_data_dir> \
  --stage2_data_dir <stage2_data_dir> \
  --stage3_data_dir <stage3_data_dir> \
  --autoencoder_ckpt_path <autoencoder_ckpt_path> \
  [--OPTIONS]
```
