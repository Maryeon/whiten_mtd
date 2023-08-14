#!/usr/bin/env python
# coding=utf-8

import os
import yaml
import logging
import shutil

import torch
import torch.nn as nn
import torch.distributed as dist


def load_config(config_file):
    assert os.path.exists(config_file), f"Config file {config_file} not found!"

    with open(config_file, "r") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    return cfg


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


def setup_logger(log_path=None, log_level=logging.INFO):
    logger = logging.root
    logger.setLevel(log_level)

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)

    if log_path is not None:
        log_file = os.path.join(log_path, "log.log")
        os.makedirs(log_path, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="w")
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return


def load_config(config_file):
    assert os.path.exists(config_file), f"Config file {config_file} not found!"

    with open(config_file, "r") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    return cfg


def save_checkpoint(state, is_best, path, filename='checkpoint.pt'):
    assert path is not None, f"Checkpoint save path should not be None type."
    os.makedirs(path, exist_ok=True)
    torch.save(state, os.path.join(path, filename))
    if is_best:
        shutil.copyfile(os.path.join(path, filename), os.path.join(path, 'model_best.pt'))


class RGB2BGR(object):
    def __call__(self, x):
        return x[(2,1,0),]


class PCA(nn.Module):
    """
    Class to  compute and apply PCA.
    """
    def __init__(self, dim1=0, dim2=0, whit=0.5):
        super().__init__()
        self.register_buffer("dim1", torch.tensor(dim1, dtype=torch.long))
        self.register_buffer("dim2", torch.tensor(dim2, dtype=torch.long))
        self.register_buffer("whit", torch.tensor(whit, dtype=torch.float32))
        self.register_buffer("d", torch.zeros(self.dim1, dtype=torch.float32))
        self.register_buffer("v", torch.zeros(self.dim1, self.dim1, dtype=torch.float32))
        self.register_buffer("n_0", torch.tensor(0, dtype=torch.long))
        self.register_buffer("mean", torch.zeros(1, self.dim1, dtype=torch.float32))
        self.register_buffer("dvt", torch.zeros(self.dim2, self.dim1, dtype=torch.float32))

    def train_pca(self, x):
        """
        Takes a covariance matrix (torch.Tensor) as input.
        """
        x_mean = x.mean(dim=0, keepdim=True)
        self.mean = x_mean
        x -= x_mean
        cov = x.t().mm(x) / x.size(0)

        d, v = torch.linalg.eigh(cov)

        self.d.copy_(d)
        self.v.copy_(v)

        eps = d.max() * 1e-5
        n_0 = (d < eps).sum()
        if n_0 > 0:
            d[d < eps] = eps

        self.n_0 = n_0

        # total energy
        totenergy = d.sum()

        # sort eigenvectors with eigenvalues order
        idx = torch.argsort(d, descending=True)[:self.dim2]
        d = d[idx]
        v = v[:, idx]

        logger = logging.getLogger("pca")
        logger.info(f"keeping {d.sum() / totenergy * 100.0:.2f} % of the energy")
                
        # for the whitening
        d = torch.diag(1. / d**self.whit)

        # principal components
        self.dvt = d @ v.T

    def forward(self, x):
        x -= self.mean
        return torch.mm(self.dvt, x.transpose(0, 1)).transpose(0, 1)