#!/usr/bin/env python
# coding=utf-8

import argparse
import os
import sys
import math
import logging
import yaml
import random
import pandas as pd

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from models import distill_model
from loss import MultiTeacherDistillLoss
import utils
import metric


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Multi-teacher knowledge distill on Google Landmark V2 cleaned dataset.")
    parser.add_argument(
        "-a", "--arch", default="resnet18", type=str, metavar="ARCH",
        choices=list(distill_model.archs.keys()), help="architecture of student model"
    )
    parser.add_argument(
        "-ts", "--teachers", required=True, nargs="+", metavar="LIST",
        help="teacher models to distill"
    )
    parser.add_argument(
        "-c", "--ckpt_dir", required=True, type=str, metavar="DIR",
        help="directory to save checkpoint files"
    )
    parser.add_argument(
        "--gld_root_path", default="/path/to/gldv2", type=str, metavar="PATH",
        help="frame root path of GLDv2 dataset"
    )
    parser.add_argument(
        "-b", "--batch_size", default=64, type=int, metavar="N",
        help="batch size"
    )
    parser.add_argument(
        "--num_workers", default=8, type=int, metavar="N",
        help="number of data loader workers"
    )
    parser.add_argument(
        "-r", "--resume", default=None, type=str, metavar="DIR",
        help="checkpoint model to resume"
    )
    parser.add_argument(
        "--path_to_pretrained_weights", default="/path/to/pretrained_weights", type=str, metavar="DIR",
        help="path to pretrained teacher models and whitening weights"
    )
    parser.add_argument(
        "--world_size", default=8, type=int,
        help="number of workers"
    )
    parser.add_argument(
        "--dist_url", default="tcp://localhost:2023", type=str,
        help='url used to set up distributed training'
    )
    parser.add_argument(
        '--rank', default=0, type=int,
        help='node rank for distributed training'
    )
    parser.add_argument(
        "--start_epoch", type=int, metavar="N", default=0,
        help="start from epoch i"
    )
    parser.add_argument(
        "--epochs", default=200, type=int, metavar="N",
        help="number of training epochs"
    )
    parser.add_argument(
        "--lr", default=1e-3, type=float, metavar="N",
        help="initial learning rate"
    )
    parser.add_argument(
        '--wd', '--weight-decay', default=1e-6, type=float, metavar='W', dest='weight_decay',
        help='weight decay (default: 1e-6)'
    )
    parser.add_argument(
        "--snapshot_step", default=10, type=int, metavar="N",
        help="interval to dump checkpoint"
    )
    parser.add_argument(
        '--embed_dim', default=512, type=int,
        help='embedding dimension'
    )
    parser.add_argument(
        '-tt', default=0.05, type=float,
        help='teacher distill temperature'
    )
    parser.add_argument(
        '-st', default=0.05, type=float,
        help='student distill temperature'
    )
    parser.add_argument(
        '-p', default=3, type=float,
        help='power rate'
    )
    parser.add_argument(
        "--imsize", default=512, type=int,
        help="input image shape"
    )
    parser.add_argument(
        "-s", "--strategy", default="maxmin", type=str,
        help="similarity fusion strategy"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # create checkpoint directory and save configurations
    os.makedirs(args.ckpt_dir, exist_ok=True)
    with open(os.path.join(args.ckpt_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    utils.setup_logger()
    logger = logging.getLogger("gld_distill")

    logger.info(vars(args))

    ngpus_per_node = torch.cuda.device_count()

    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    cudnn.benchmark = True

    args.gpu = gpu

    if args.dist_url == "env://" and args.rank == -1:
        args.rank = int(os.environ["RANK"])
    args.rank = args.rank * ngpus_per_node + gpu

    if args.rank == 0:
        utils.setup_logger(args.ckpt_dir)
    else:
        utils.setup_logger(log_path=None)
    logger = logging.getLogger("worker " + str(args.rank))

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank
    )

    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda:"+str(torch.cuda.current_device()))
    logger.info(f"Using device cuda:{torch.cuda.current_device()}")

    # parse landmarks
    df = pd.read_csv(os.path.join(args.gld_root_path, "meta/train_clean.csv"))
    landmark_list = df[["images"]].values.tolist()
    landmark_list = list(map(lambda x: x[0].split(" "), landmark_list))
    landmark_list = list(group for group in landmark_list if len(group) >= 2)

    transform = T.Compose([
        T.RandomResizedCrop(args.imsize, scale=(0.4, 1.)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    train_dataset = GLDImageDataset(os.path.join(args.gld_root_path, "train"), transform)
    train_sampler = DistributedGroupSampler(
        args.rank, args.world_size, args.batch_size, landmark_list,
        shuffle=True, drop_last=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True
    )

    model = distill_model.MultiTeacherDistillModel(args)
    logger.info(f"Build model {model.__class__.__name__}")
    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    n_trainable_parameters = sum([p.data.nelement() for p in model.parameters() if p.requires_grad])
    logger.info(f"Number of parameters: {n_parameters}")
    logger.info(f"Number of trainable parameters: {n_trainable_parameters}")

    model.cuda(device)

    if utils.has_batchnorms(model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * len(train_loader), eta_min=1e-6
    )

    loss_fn = MultiTeacherDistillLoss(
        st=args.st, tt=args.tt,
        s=args.strategy, teachers=args.teachers
    ).to(device)

    if args.resume is not None:
        checkpoint_file = args.resume
        if os.path.isfile(checkpoint_file):
            logger.info(f"Loading checkpoint \"{checkpoint_file}\"...")
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
        else:
            logger.error(f"=> No checkpoint found at '{checkpoint_file}'.")
            sys.exit()

        model.load_state_dict(checkpoint["state_dict"], strict=False)
        args.start_epoch = checkpoint["epoch"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        loss_fn.load_state_dict(checkpoint["loss_fn"])
        logger.info(f"Loaded checkpoint.")
    
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(
            model,
            train_loader,
            optimizer,
            lr_scheduler,
            loss_fn,
            device,
            epoch,
            args.epochs
        )
        
        if (epoch + 1) % args.snapshot_step == 0 and args.rank == 0:
            utils.save_checkpoint(
                {
                    "epoch": epoch+1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "loss_fn": loss_fn.state_dict()
                },
                False,
                args.ckpt_dir,
                filename=f"checkpoint_{epoch+1}.pt"
            )


def train_one_epoch(model, train_loader, optimizer, lr_scheduler, loss_fn, device, epoch, total_epochs):
    model.train()

    rank = utils.get_rank()
    if rank == 0:
        logger = logging.getLogger("gld_distill_train")
    else:
        logger = None
    metric_logger = metric.MetricLogger(logger, delimiter="    ")
    header = f'Epoch: [{epoch+1}/{total_epochs}]'
    iters_per_epoch = len(train_loader)
    log_freq = iters_per_epoch // 16 if iters_per_epoch >= 16 else iters_per_epoch

    train_loader.batch_sampler.set_epoch(epoch)

    for batch_idx, batch in metric_logger.log_every(train_loader, log_freq, header=header, iterations=iters_per_epoch):
        x, img_ids = batch

        batch_size = x.size(0) // 2

        x1 = x[:batch_size]
        x2 = x[batch_size:]

        x1 = x1.to(device)
        x2 = x2.to(device)

        predicts = model(x1, x2)

        loss_dict = loss_fn(*predicts)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if rank == 0 and batch_idx % log_freq == 0:
            logger.info(f"learning rate: {lr_scheduler.get_last_lr()[0]:.8f}")
            logger.info(loss_fn)
        lr_scheduler.step()

        loss_dict = {k: v.item() for k, v in loss_dict.items()}
        metric_logger.update(**loss_dict)

    metric_logger.sync()
    if rank == 0:
        logger.info("Averaged stats: " + str(metric_logger))


class GLDImageDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, root_path, transform=None):
        super().__init__()
        self.root_path = root_path
        self.t = transform

    def __getitem__(self, img_id):
        img_path = os.path.join(
            self.root_path,
            img_id[0], img_id[1], img_id[2],
            img_id+".jpg"
        )
        
        img = Image.open(img_path)
        img = img.convert("RGB")
        if self.t is not None:
            img = self.t(img)
        
        return img, img_id

    def __len__(self):
        return len(self.img_id_list)


class DistributedSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, rank, num_replicas, batch_size, samples, seed=0, shuffle=True, drop_last=True):
        self._rank = rank
        self._num_replicas = num_replicas
        self._batch_size = batch_size
        self._samples = samples
        self._seed = seed
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._epoch = 0

        if self._drop_last and len(self._samples) % self._num_replicas != 0:
            self._num_samples = math.ceil(
                (len(self._samples) - self._num_replicas) / self._num_replicas
            )
        else:
            self._num_samples = math.ceil(len(self._samples) / self._num_replicas)
        self._total_size = self._num_samples * self._num_replicas

    def __len__(self):
        return math.ceil(self._num_samples/self._batch_size)

    def set_epoch(self, epoch):
        self._epoch = epoch

    def _get_subset(self):
        if self._shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self._seed + self._epoch)
            indices = torch.randperm(len(self._samples), generator=g).tolist()
        else:
            indices = list(range(len(self._samples)))

        if not self._drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self._total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self._total_size]
        assert len(indices) == self._total_size

        # subsample
        indices = indices[self._rank:self._total_size:self._num_replicas]
        assert len(indices) == self._num_samples

        return [self._samples[i] for i in indices]


class DistributedGroupSampler(DistributedSampler):
    def __init__(self, rank, num_replicas, batch_size, groups, seed=0, shuffle=True, drop_last=True):
        super().__init__(
            rank, num_replicas, batch_size, groups,
            seed=seed, shuffle=shuffle, drop_last=drop_last
        )
        random.seed(rank)

        logger = logging.getLogger("dist_group_sampler."+str(rank))
        logger.info(self)

    def __str__(self):
        return f"| Distributed Group Sampler | {self._num_samples} groups | iters {len(self)} | {self._batch_size} per batch"

    def __iter__(self):
        subset = self._get_subset()
        for i in range(0, len(subset), self._batch_size):
            groups = subset[i:i+self._batch_size]
            img_id_pairs = [random.sample(group, 2) for group in groups]
            yield [pair[0] for pair in img_id_pairs] + [pair[1] for pair in img_id_pairs]


if __name__ == "__main__":
    main()
