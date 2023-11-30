import os
import sys
import math
import pprint
import shutil
import random
import logging
import argparse
import numpy as np

import torch

import torchdrug
from torchdrug import core, datasets, tasks, models, layers
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import util
from gearnet import dataset, model, task
from gearnet.engine import MultiTaskEngine


def build_solver(cfg, logger):
    # build dataset and model for each task
    model = core.Configurable.load_config_dict(cfg.model)

    train_sets, valid_sets, test_sets = [], [], []
    tasks = []
    for dataset_config, task_config in zip(cfg.datasets, cfg.tasks):
        if "center" in dataset_config:
            is_center = dataset_config.pop("center")
        else:
            is_center = False
        if "test_split" in dataset_config:
            test_split = dataset_config.pop("test_split")
            _dataset = core.Configurable.load_config_dict(dataset_config)
            train_set, valid_set, test_set = _dataset.split(['train', 'valid', test_split])
        else:
            _dataset = core.Configurable.load_config_dict(dataset_config)
            train_set, valid_set, test_set = _dataset.split()
        if comm.get_rank() == 0:
            logger.warning(_dataset)
            logger.warning("#train: %d, #valid: %d, #test: %d" % (len(train_set), len(valid_set), len(test_set)))
        if is_center:
            train_sets = [train_set] + train_sets
            valid_sets = [valid_set] + valid_sets
            test_sets = [test_set] + test_sets
        else:
            train_sets.append(train_set)
            valid_sets.append(valid_set)
            test_sets.append(test_set)

        if task_config["class"] in ["PropertyPrediction", "PairReward"]:
            task_config.task = _dataset.tasks
        task_config.model = model
        task = core.Configurable.load_config_dict(task_config)
        if is_center:
            tasks = [task] + tasks
        else:
            tasks.append(task)

    # build solver
    cfg.optimizer.params = model.parameters()
    optimizer = core.Configurable.load_config_dict(cfg.optimizer)

    if not "scheduler" in cfg:
        scheduler = None
    else:
        cfg.scheduler.optimizer = optimizer
        scheduler = core.Configurable.load_config_dict(cfg.scheduler)

    solver = MultiTaskEngine(tasks, train_sets, valid_sets, test_sets, optimizer, scheduler, **cfg.engine)
    if "lr_ratio" in cfg:
        cfg.optimizer.params = [
            {'params': model.parameters(), 'lr': cfg.optimizer.lr * cfg.lr_ratio},
        ]
        for model in solver.models:
            cfg.optimizer.params.append({
                'params': model.mlp.parameters(), 'lr': cfg.optimizer.lr
            })
        optimizer = core.Configurable.load_config_dict(cfg.optimizer)
        solver.optimizer = optimizer
    elif "sequence_model_lr_ratio" in cfg:
        assert cfg.model["class"] == "FusionNetwork"
        cfg.optimizer.params = [
            {'params': model.sequence_model.parameters(), 'lr': cfg.optimizer.lr * cfg.sequence_model_lr_ratio},
            {'params': model.structure_model.parameters(), 'lr': cfg.optimizer.lr},
        ]
        for model in solver.models:
            cfg.optimizer.params.append({
                'params': model.mlp.parameters(), 'lr': cfg.optimizer.lr
            })
        optimizer = core.Configurable.load_config_dict(cfg.optimizer)
        solver.optimizer = optimizer

    if "checkpoint" in cfg:
        solver.load(cfg.checkpoint, load_optimizer=False)

    return solver


def train_and_validate(cfg, solver):
    step = math.ceil(cfg.train.num_epoch / 50)
    best_epoch = -1
    best_result = float("-inf")
    best_epoch = -1
    best_val_metric = {}
    best_metric = {}

    if not cfg.train.num_epoch > 0:
        return solver, best_epoch

    for i in range(0, cfg.train.num_epoch, step):
        kwargs = cfg.train.copy()
        kwargs["num_epoch"] = min(step, cfg.train.num_epoch - i)
        solver.train(**kwargs)
        # solver.save("model_epoch_%d.pth" % solver.epoch)
        if "test_batch_size" in cfg:
            solver.batch_size = cfg.test_batch_size
        metric = solver.evaluate("valid")
        test_metric = solver.evaluate("test")
        solver.batch_size = cfg.engine.batch_size

        result = metric[cfg.metric]
        if result > best_result:
            best_result = result
            best_epoch = solver.epoch
            best_val_metric = metric
            best_metric = test_metric

    # solver.load("model_epoch_%d.pth" % best_epoch)
    print(best_epoch)
    pprint.pprint(best_val_metric)
    pprint.pprint(best_metric)
    return solver


def test(cfg, solver):
    if "test_batch_size" in cfg:
        solver.batch_size = cfg.test_batch_size
    solver.evaluate("valid")
    solver.evaluate("test")


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory_mtl(cfg)

    seed = args.seed
    torch.manual_seed(seed + comm.get_rank())
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

    solver = build_solver(cfg, logger)
    solver = train_and_validate(cfg, solver)
    test(cfg, solver)