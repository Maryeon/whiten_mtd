#!/usr/bin/env python
# coding=utf-8

import argparse
import os
import logging
import random

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from models.teachers import teacher_models
import utils
import metric


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="PCA training and saving.")
    parser.add_argument(
        "-t", "--teacher", default=None, type=str, metavar="ARCH",
        choices=list(teacher_models.keys()), help="name of teacher model"
    )
    parser.add_argument(
        "--imsize", default=None, type=int,
        help="input image shape"
    )
    parser.add_argument(
        "--num_workers", default=8, type=int, metavar="N",
        help="number of data loader workers"
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
        '--embed_dim', default=512, type=int,
        help='embedding dimension'
    )
    parser.add_argument(
        "-b", "--batch_size", default=256, type=int,
        help="batch size"
    )
    parser.add_argument(
        '-p', default=3, type=float,
        help='power rate'
    )
    parser.add_argument(
        '--num_samples', default=10000, type=int,
        help='number of samples to train PCA transform'
    )
    parser.add_argument(
        "--gld_root_path", default="/path/to/gldv2", type=str, metavar="PATH",
        help="frame root path of GLDv2 dataset"
    )
    parser.add_argument(
        "--dump_to", required=True, type=str,
        help="dump learned pca transformer to"
    )
    return parser.parse_args()


def main():
    random.seed(0)

    args = parse_args()

    utils.setup_logger()
    logger = logging.getLogger("pca training")

    logger.info(vars(args))

    ngpus_per_node = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    cudnn.benchmark = True

    args.gpu = gpu

    if args.dist_url == "env://" and args.rank == -1:
        args.rank = int(os.environ["RANK"])
    args.rank = args.rank * ngpus_per_node + gpu

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

    model = teacher_models[args.teacher](gem_p=args.p)
    model.cuda(device)
    if utils.has_batchnorms(model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])

    pca_wtn = learn_pca_whitening(args, model)

    if args.rank == 0:
        torch.save(pca_wtn.state_dict(), args.dump_to)


@torch.no_grad()
def learn_pca_whitening(args, model):
    model.eval()
    logger = logging.getLogger("pca training")

    image_id_list = list()
    with open(os.path.join(args.gld_root_path, "meta/train_clean_ids.txt"), "r") as f:
        for l in f:
            image_id_list.append(l.strip())

    logger.info(f"totally {len(image_id_list)} images.")
    
    num_samples = args.num_samples
    if num_samples > 0:
        logger.info(f"select {num_samples} samples to train pca whitening.")
        random.shuffle(image_id_list)
        image_id_list = image_id_list[:num_samples]
    else:
        logger.info(f"select all samples to train pca whitening.")
        image_id_list = image_id_list

    transform_list = [
        T.RandomResizedCrop(args.imsize, scale=(0.4, 1.)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
    if args.teacher.endswith("delg") or args.teacher.endswith("dolg"):
        transform_list.append(utils.RGB2BGR())
    transform = T.Compose(transform_list)
    train_dataset = GLDv2(
        os.path.join(args.gld_root_path, "train"), image_id_list,
        transform=transform
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False, drop_last=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True
    )

    rank = utils.get_rank()
    if rank == 0:
        pca_wtn = utils.PCA(dim1=model.module.embed_dim, dim2=args.embed_dim)

    logger.info("extracting pca training features")
    features = extract_features(model, train_loader)

    if rank == 0:
        pca_wtn.train_pca(features)
        return pca_wtn
    else:
        return None


@torch.no_grad()
def extract_features(model, data_loader):
    rank = utils.get_rank()
    if rank == 0:
        logger = logging.getLogger("extract_feature")
    else:
        logger = None
    metric_logger = metric.MetricLogger(logger, delimiter="    ")
    log_freq = len(data_loader) // 16 if len(data_loader) >= 16 else len(data_loader)
    if rank == 0:
        features = torch.zeros(len(data_loader.dataset), model.module.embed_dim)
    for batch_idx, (samples, index) in metric_logger.log_every(data_loader, log_freq):
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)

        feats = model(samples)

        feats = nn.functional.normalize(feats, p=2, dim=-1)

        index_all = gather_to_main(index)
        feats_all = gather_to_main(feats)

        if rank == 0:
            features.index_copy_(0, index_all.cpu(), feats_all.cpu())

    if rank == 0:
        return features
    else:
        return None


def gather_to_main(x):
    world_size = utils.get_world_size()
    rank = utils.get_rank()

    if rank == 0:
        gather_x = [torch.zeros_like(x) for _ in range(world_size)]
    else:
        gather_x = None
    dist.gather(x, gather_x if rank == 0 else None, dst=0)

    if rank == 0:
        gather_x = torch.cat(gather_x, dim=0)

    return gather_x


class GLDv2(torch.utils.data.dataset.Dataset):
    def __init__(self, root_path, img_id_list, transform=None):
        super().__init__()
        self.root_path = root_path
        self.img_id_list = img_id_list

        self.t = transform

    def __getitem__(self, i):
        img_id = self.img_id_list[i]
        img_path = os.path.join(
            self.root_path,
            img_id[0], img_id[1], img_id[2],
            img_id+".jpg"
        )
        
        img = Image.open(img_path)
        img = img.convert("RGB")
        if self.t is not None:
            img = self.t(img)
        
        return img, i

    def __len__(self):
        return len(self.img_id_list)


if __name__ == "__main__":
    main()
