import argparse
import os
import sys
import logging
import yaml
import random

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torchvision.transforms as T
from PIL import ImageFilter, ImageOps

from models import distill_model
from datasets.svd.core import Frame, MetaData
from datasets.svd.loader import DistributedDistillLoader
from loss import CrossViewSimilarityDistillLoss
import utils
import metric


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Knowledge Distill on SVD dataset.")
    parser.add_argument(
        "-a", "--arch", default="mobilenet_v2", type=str, metavar="ARCH",
        choices=list(distill_model.archs.keys()), help="name of backbone model"
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
        "-dm", "--dataset_meta", default="config/svd.yaml", type=str, metavar="FILE",
        help="dataset meta file"
    )
    parser.add_argument(
        "-b", "--batch_size", default=64, type=int, metavar="N",
        help="training batch size"
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
        "--world_size", default=8, type=int,
        help="number of workers"
    )
    parser.add_argument(
        "--dist_url", default="tcp://localhost:1234", type=str,
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
        help="training epochs"
    )
    parser.add_argument(
        "--warmup_epochs", default=5, type=int, metavar="N",
        help="number of warmup epochs"
    )
    parser.add_argument(
        "--lr", default=0.3, type=float, metavar="N",
        help="initial learning rate"
    )
    parser.add_argument(
        "-t", default=0.05, type=float, metavar="N",
        help="temprature rate of distillation loss"
    )
    parser.add_argument(
        '--wd', '--weight-decay', default=1e-6, type=float, metavar='W', dest='weight_decay',
        help='weight decay (default: 1e-6)'
    )
    parser.add_argument(
        "--snapshot_step", default=5, type=int, metavar="N",
        help="interval to dump checkpoint"
    )
    parser.add_argument(
        '-p', default=1, type=float,
        help='power rate'
    )
    parser.add_argument(
        '--embed_dim', default=512, type=int,
        help='embedding dimension'
    )
    parser.add_argument(
        "--wopca", action="store_true",
        help="do not pca-whitening teacher models"
    )
    parser.add_argument(
        "-s", "--strategy", default=None, type=str,
        help="similarity fusion strategy"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    with open(os.path.join(args.ckpt_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    utils.setup_logger()
    logger = logging.getLogger("svd_distill")

    logger.info(vars(args))

    ngpus_per_node = torch.cuda.device_count()

    mp.spawn(main_worker, nprocs=8, args=(ngpus_per_node, args))


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
    logger = logging.getLogger("dist_worker " + str(args.rank))

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank
    )

    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda:"+str(torch.cuda.current_device()))
    logger.info(f"Using device cuda:{torch.cuda.current_device()}")

    dataset_meta_cfg = utils.load_config(args.dataset_meta)
    dataset_meta = MetaData(dataset_meta_cfg)

    train_dataset = Frame(
        dataset_meta.frm_root_path,
        dataset_meta.frm_cnt,
        TwoCropsTransform()
    )
    train_loader = DistributedDistillLoader.build(
        train_dataset, dataset_meta,
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = distill_model.CrossViewDistillModel(args)
    logger.info(f"Build model {model.__class__.__name__}")
    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    n_trainable_parameters = sum([p.data.nelement() for p in model.parameters() if p.requires_grad])
    logger.info(f"Number of parameters: {n_parameters}")
    logger.info(f"Number of trainable parameters: {n_trainable_parameters}")
    
    model.cuda(device)

    if utils.has_batchnorms(model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])

    # args.lr = args.lr * args.batch_size * args.world_size / 256

    if args.arch.startswith("vit"):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr
        )
    else:
        # optimizer = LARS(
        #     filter(lambda p: p.requires_grad, model.parameters()),
        #     lr=args.lr,
        #     weight_decay=args.weight_decay
        # )
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    # warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
    #     optimizer,
    #     start_factor=1e-6, end_factor=1, total_iters=args.warmup_epochs * len(train_loader)
    # )
    # cosine_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=(args.epochs - args.warmup_epochs) * len(train_loader), eta_min=1e-6
    # )
    # lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
    #     optimizer,
    #     [warmup_lr_scheduler, cosine_lr_scheduler],
    #     milestones=[args.warmup_epochs * len(train_loader)]
    # )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * len(train_loader), eta_min=1e-6
    )

    loss_fn = CrossViewSimilarityDistillLoss(
        dt=args.t, teachers=args.teachers, s=args.strategy
    ).to(device)

    if args.resume is not None:
        checkpoint_file = args.resume
        if os.path.isfile(checkpoint_file):
            logger.info(f"Loading checkpoint \"{checkpoint_file}\"...")
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
        else:
            logger.error(f"=> No checkpoint found at '{checkpoint_file}'.")
            sys.exit()

        model.load_state_dict(checkpoint["state_dict"], strict=True)
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
            epoch+1,
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
        logger = logging.getLogger("svd_distill_train")
    else:
        logger = None
    metric_logger = metric.MetricLogger(logger, delimiter="    ")
    header = f'Epoch: [{epoch}/{total_epochs}]'
    log_freq = len(train_loader) // 16 if len(train_loader) >= 16 else len(train_loader)

    train_loader.batch_sampler.set_epoch(epoch)

    for batch_idx, batch in metric_logger.log_every(train_loader, log_freq, header=header, iterations=len(train_loader)):
        x1, x2, _ = batch

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


class TwoCropsTransform:
    def __init__(self):
        self.transform1 = T.Compose([
            T.RandomResizedCrop(224, scale=(0.4, 1.)),
            T.RandomApply([
                T.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([GaussianBlur([.1, 2.])], p=1.),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.transform2 = T.Compose([
            T.RandomResizedCrop(224, scale=(0.4, 1.)),
            T.RandomApply([
                T.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
            T.RandomApply([Solarize()], p=0.2),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __call__(self, x):
        im1 = self.transform1(x)
        im2 = self.transform2(x)
        return im1, im2


class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)


class LARS(torch.optim.Optimizer):
    """
    LARS optimizer, no rate scaling or weight decay for parameters <= 1D.
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, trust_coefficient=0.001):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, trust_coefficient=trust_coefficient)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim > 1: # if not normalization gamma/beta or bias
                    dp = dp.add(p, alpha=g['weight_decay'])
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                    (g['trust_coefficient'] * param_norm / update_norm), one),
                                    one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)
                p.add_(mu, alpha=-g['lr'])



if __name__ == "__main__":
    main()
