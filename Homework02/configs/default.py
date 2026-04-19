"""Default configuration for SVHN experiments."""

from __future__ import annotations

from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    """Return the base experiment configuration."""
    config = ConfigDict()

    config.experiment = ConfigDict()
    config.experiment.name = "default"
    config.experiment.run_name = ""
    config.experiment.notes = ""

    config.paths = ConfigDict()
    config.paths.data_dir = "data/raw"
    config.paths.train_mat = "data/raw/train_32x32.mat"
    config.paths.test_mat = "data/raw/test_32x32.mat"
    config.paths.split_dir = "data/splits"
    config.paths.output_root = "outputs"

    config.data = ConfigDict()
    config.data.image_size = 32
    config.data.in_channels = 3
    config.data.num_classes = 10
    config.data.val_ratio = 0.1
    config.data.num_workers = 0
    config.data.pin_memory = True
    config.data.normalize = True
    config.data.mean = [0.5, 0.5, 0.5]
    config.data.std = [0.5, 0.5, 0.5]

    config.model = ConfigDict()
    config.model.name = "cnn"
    config.model.num_classes = 10
    config.model.in_channels = 3
    config.model.base_channels = 32
    config.model.dropout = 0.3
    config.model.reduction = 16

    config.train = ConfigDict()
    config.train.seed = 42
    config.train.device = "cuda:2"
    config.train.batch_size = 128
    config.train.epochs = 20
    config.train.log_interval = 50
    config.train.save_best_metric = "val_acc"
    config.train.save_last = True

    config.optimizer = ConfigDict()
    config.optimizer.name = "sgd"
    config.optimizer.lr = 0.01
    config.optimizer.momentum = 0.9
    config.optimizer.weight_decay = 5e-4
    config.optimizer.nesterov = False

    config.scheduler = ConfigDict()
    config.scheduler.name = "step"
    config.scheduler.step_size = 10
    config.scheduler.gamma = 0.1

    config.eval = ConfigDict()
    config.eval.batch_size = 256

    config.plot = ConfigDict()
    config.plot.dpi = 150

    return config
