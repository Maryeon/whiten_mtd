import argparse
import os
import sys
import logging
import random
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageFilter, ImageOps

from datasets.svd.core import MetaData
from models.teachers import teacher_models
from models.gem_pooling import GeneralizedMeanPooling
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
        '-p', default=1, type=float,
        help='power rate'
    )
    parser.add_argument(
        "--dump_to", required=True, type=str,
        help="dump learned pca transformer to"
    )
    parser.add_argument(
        "-dm", "--dataset_meta", default="config/svd.yaml", type=str, metavar="FILE",
        help="dataset meta file"
    )
    return parser.parse_args()


def main():
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


def sample_frames(video_ids, frm_cnt, n=60):
    random.seed(0)
    image_id_list = list()

    for video_id in video_ids:
        frame_ids = random.sample(list(range(frm_cnt[video_id])), min(n, frm_cnt[video_id]))
        image_ids = [f"{video_id}/{frame_id:04d}" for frame_id in frame_ids]
        image_id_list += image_ids

    return image_id_list


@torch.no_grad()
def learn_pca_whitening(args, model):
    model.eval()
    logger = logging.getLogger("pca training")

    dataset_meta_cfg = utils.load_config(args.dataset_meta)
    dataset_meta = MetaData(dataset_meta_cfg)


    # video_ids = dataset_meta.train_ids + dataset_meta.unlabeled_ids
    video_ids = dataset_meta.train_ids
    rank = utils.get_rank()
    if rank == 0:
        image_id_list = sample_frames(video_ids, dataset_meta.frm_cnt)
        objects = [image_id_list]
    else:
        objects = [None]
    dist.broadcast_object_list(
        objects, src=0, 
        device=torch.device("cuda:"+str(torch.cuda.current_device()))
    )
    image_id_list = objects[0]
    logger.info(f"totally {len(image_id_list)} images.")

    transform = T.Compose([
        T.RandomResizedCrop(224, scale=(0.4, 1.)),
        T.RandomApply([
            T.ColorJitter(0.4, 0.4, 0.2, 0.1)
        ], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        T.RandomApply([Solarize()], p=0.2),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    train_dataset = SVD(
        dataset_meta.frm_root_path, image_id_list, transform
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


class SVD(torch.utils.data.dataset.Dataset):
    def __init__(self, root_path, img_id_list, transform):
        super().__init__()
        self.root_path = root_path
        self.img_id_list = img_id_list
        self.t = transform

    def __getitem__(self, i):
        img_id = self.img_id_list[i]
        img_path = os.path.join(
            self.root_path, img_id+".jpg"
        )
        
        img = Image.open(img_path)
        img = img.convert("RGB")
        img = self.t(img)
        
        return img, i

    def __len__(self):
        return len(self.img_id_list)


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


if __name__ == "__main__":
    main()
