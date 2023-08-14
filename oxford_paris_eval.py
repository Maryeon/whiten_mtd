#!/usr/bin/env python
# coding=utf-8

import argparse
import os
import sys
import logging
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

import datasets.oxford_paris as oxford_paris
from models.distill_model import archs
from models.teachers import teacher_models
from models.gem_pooling import GeneralizedMeanPooling
from models.pca_layer import pca_layers
import utils
import metric


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Evaluation on roxford5k or rparis6k.")
    parser.add_argument(
        "-dp", "--data_path", default="/path/to/datasets", type=str,
        help="root path to dataset"
    )
    parser.add_argument(
        "-d", "--dataset", default="roxford5k", type=str,
        choices=["roxford5k", "rparis6k"], help="dataset name"
    )
    parser.add_argument(
        "-a", "--arch", default=None, type=str, metavar="ARCH",
        choices=list(archs.keys()), help="architecture of backbone model"
    )
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
        help="number of workers per node"
    )
    parser.add_argument(
        "--dist_url", default="tcp://localhost:2023", type=str,
        help='url used to set up distributed evaluation'
    )
    parser.add_argument(
        '--rank', default=0, type=int,
        help='node rank for distributed training'
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
        "-ms", "--multiscale", action="store_true",
        help="multiscale testing"
    )
    parser.add_argument(
        '--embed_dim', default=512, type=int,
        help='embedding dimension'
    )
    parser.add_argument(
        '-p', default=3, type=float,
        help='power rate'
    )
    parser.add_argument(
        "--plus1m", action="store_true",
        help="plus 1M distractors"
    )
    parser.add_argument(
        "--pca", action="store_true",
        help="whether to perform PCA-Whitening on teacher model"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    utils.setup_logger()
    logger = logging.getLogger("image retrieval")

    logger.info(vars(args))

    # by default use all available gpus
    ngpus_per_node = torch.cuda.device_count()

    # distributed evaluation
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

    # arguments arch and teacher are mutually exclusive
    assert args.arch is None or args.teacher is None

    # build student model
    if args.arch is not None:
        model = archs[args.arch](pretrained=False, num_classes=args.embed_dim)
        model.avgpool = GeneralizedMeanPooling(args.p)

        if args.resume is not None:
            checkpoint_file = args.resume
            if os.path.isfile(checkpoint_file):
                logger.info(f"Loading checkpoint \"{checkpoint_file}\"...")
                checkpoint = torch.load(checkpoint_file, map_location='cpu')
            else:
                logger.error(f"=> No checkpoint found at '{checkpoint_file}'.")
                sys.exit()

            state_dict = checkpoint["state_dict"]
            for k in list(state_dict.keys()):
                if k.startswith('module.base_encoder'):
                    state_dict[k[len("module.base_encoder."):]] = state_dict.pop(k)
                else:
                    state_dict.pop(k)
            model.load_state_dict(state_dict, strict=True)
            logger.info(f"Loaded checkpoint.")

    # build teacher model
    if args.teacher is not None:
        model = teacher_models[args.teacher](args.path_to_pretrained_weights, gem_p=args.p)

    model.cuda(device)

    if utils.has_batchnorms(model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])

    model.eval()
    
    # for teacher model, a pca-whitening layer is optional to be used
    pca_wtn = None
    if args.teacher is not None and args.pca:
        pca_wtn = pca_layers[args.teacher]().cuda(device)
        model.module.embed_dim = pca_wtn.dim2

    # no image scale
    transform_list = [
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
    # DELG and DOLG's input channel is BGR rather than default RGB
    if args.teacher is not None and (args.teacher.endswith("delg") or args.teacher.endswith("dolg")):
        transform_list.append(utils.RGB2BGR())
    transform = T.Compose(transform_list)
    dataset_query = oxford_paris.OxfordParisDataset(
        args.data_path, args.dataset, "query",
        transform=transform, imsize=args.imsize
    )
    dataset_database = oxford_paris.OxfordParisDataset(
        args.data_path, args.dataset, "database",
        transform=transform, imsize=args.imsize
    )
    sampler_query = torch.utils.data.distributed.DistributedSampler(
        dataset_query, num_replicas=args.world_size, rank=args.rank, shuffle=False, drop_last=False
    )
    sampler_database = torch.utils.data.distributed.DistributedSampler(
        dataset_database, num_replicas=args.world_size, rank=args.rank, shuffle=False, drop_last=False
    )
    data_loader_query = torch.utils.data.DataLoader(
        dataset_query, 
        batch_size=1, shuffle=False, sampler=sampler_query,
        pin_memory=True, num_workers=args.num_workers
    )
    data_loader_database = torch.utils.data.DataLoader(
        dataset_database,
        batch_size=1, shuffle=False, sampler=sampler_database,
        pin_memory=True, num_workers=args.num_workers
    )

    logger.info(f"database: {len(dataset_database)} imgs")
    logger.info(f"query: {len(dataset_query)} imgs")

    query_features = extract_features(
        model, data_loader_query, 
        pca_wtn=pca_wtn, 
        multiscale=args.multiscale
    )
    database_features = extract_features(
        model, data_loader_database, 
        pca_wtn=pca_wtn, 
        multiscale=args.multiscale
    )

    if args.plus1m:
        dataset_distractor = oxford_paris.OxfordParisDataset(
            args.data_path, "revisitop1m", "distractor",
            transform=transform, imsize=args.imsize
        )
        sampler_distractor = torch.utils.data.distributed.DistributedSampler(
            dataset_distractor, num_replicas=args.world_size, rank=args.rank, shuffle=False, drop_last=False
        )
        data_loader_distractor = torch.utils.data.DataLoader(
            dataset_distractor, 
            batch_size=1, shuffle=False, sampler=sampler_distractor,
            pin_memory=True, num_workers=args.num_workers
        )
        logger.info(f"distractor: {len(dataset_distractor)} imgs")
        distractor_features = extract_features(
            model, data_loader_distractor, 
            pca_wtn=pca_wtn, multiscale=args.multiscale
        )

    # calculate metrics on the main process
    if args.rank == 0:
        # Step 1: normalize features
        database_features = nn.functional.normalize(database_features, dim=1, p=2)
        query_features = nn.functional.normalize(query_features, dim=1, p=2)
        if args.plus1m:
            distractor_features = nn.functional.normalize(distractor_features, dim=1, p=2)
            database_features = torch.cat([database_features, distractor_features], dim=0)

        ############################################################################
        # Step 2: similarity
        sim = torch.mm(database_features, query_features.T)
        ranks = torch.argsort(-sim, dim=0).cpu().numpy()

        ############################################################################
        # Step 3: evaluate
        gnd = dataset_database.cfg['gnd']
        # evaluate ranks
        ks = [1, 5, 10]
        # search for easy & hard
        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
            g['junk'] = np.concatenate([gnd[i]['junk']])
            gnd_t.append(g)
        mapM, apsM, mprM, prsM = oxford_paris.compute_map(ranks, gnd_t, ks)
        # search for hard
        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['hard']])
            g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
            gnd_t.append(g)
        mapH, apsH, mprH, prsH = oxford_paris.compute_map(ranks, gnd_t, ks)
        logger.info('>> {}: mAP M: {}, H: {}'.format(args.dataset, np.around(mapM*100, decimals=2), np.around(mapH*100, decimals=2)))
        logger.info('>> {}: mP@k{} M: {}, H: {}'.format(args.dataset, np.array(ks), np.around(mprM*100, decimals=2), np.around(mprH*100, decimals=2)))

    dist.barrier()


@torch.no_grad()
def extract_features(model, data_loader, pca_wtn=None, multiscale=False):
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

        if multiscale:
            feats = multiscale_feature(samples, model, pca_wtn=pca_wtn)
        else:
            feats = model(samples)
            if pca_wtn is not None:
                feats = nn.functional.normalize(feats, p=2, dim=-1)
                feats = pca_wtn(feats)

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


def multiscale_feature(samples, model, pca_wtn=None):
    scale = [1, 1/2**(1/2), 1/2]
    v = None
    for s in scale:
        if s == 1:
            inp = samples
        else:
            inp = nn.functional.interpolate(samples, scale_factor=s, mode="bilinear", align_corners=False)
        feats = model(inp)
        feats = nn.functional.normalize(feats, p=2, dim=-1)
        if pca_wtn is not None:
            feats = pca_wtn(feats)
            feats = nn.functional.normalize(feats, p=2, dim=-1)

        if v is None:
            v = feats
        else:
            v += feats
    v /= len(scale)
    return v


if __name__ == "__main__":
    main()
