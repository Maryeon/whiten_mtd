import argparse
import os
import sys
import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torchvision.transforms as T

from models.distill_model import archs
from models.teachers import teacher_models
from models.gem_pooling import GeneralizedMeanPooling
from datasets.svd.core import Video, MetaData
from datasets.svd.loader import DistributedTestLoader
from datasets.svd.eval import DistEvaluator
import utils


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Knowledge Distill Evaluation on SVD dataset.")
    parser.add_argument(
        "-a", "--arch", default=None, type=str, metavar="ARCH",
        choices=list(archs.keys()), help="name of backbone model"
    )
    parser.add_argument(
        "-t", "--teacher", default=None, type=str, metavar="ARCH",
        choices=list(teacher_models.keys()), help="name of teacher model"
    )
    parser.add_argument(
        "-dm", "--dataset_meta", default="config/svd.yaml", type=str, metavar="FILE",
        help="dataset meta file"
    )
    parser.add_argument(
        "--max_frames", default=60, type=int, metavar="N",
        help="max number of frames to truncate"
    )
    parser.add_argument(
        "--stride", default=1, type=int, metavar="N",
        help="stride to sample frames",
    )
    parser.add_argument(
        "-b", "--batch_size", default=16, type=int, metavar="N",
        help="test batch size"
    )
    parser.add_argument(
        "--num_workers", default=8, type=int, metavar="N",
        help="number of data loader workers"
    )
    parser.add_argument(
        "--topk", default=[100], nargs="+", metavar="N",
        help="to calculate topk metric"
    )
    parser.add_argument(
        "--sim_fn", default="fmx", type=str,
        help="similarity function to use"
    )
    parser.add_argument(
        "--subset_eval", action="store_true",
        help="eval full or subset"
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
        '--embed_dim', default=512, type=int,
        help='embedding dimension'
    )
    parser.add_argument(
        '-p', default=1., type=float,
        help='power rate'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    utils.setup_logger()
    logger = logging.getLogger("svd_distill_eval")

    logger.info(vars(args))

    ngpus_per_node = torch.cuda.device_count()

    mp.spawn(main_worker, nprocs=8, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.dist_url == "env://" and args.rank == -1:
        args.rank = int(os.environ["RANK"])
    args.rank = args.rank * ngpus_per_node + gpu

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

    assert args.arch is None or args.teacher is None

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

    if args.teacher is not None:
        model = teacher_models[args.teacher](gem_p=args.p)

    model.cuda(device)

    if utils.has_batchnorms(model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])

    dataset_meta_cfg = utils.load_config(args.dataset_meta)
    dataset_meta = MetaData(dataset_meta_cfg)

    test_dataset = Video(
        dataset_meta.frm_root_path,
        dataset_meta.frm_cnt,
        T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor()
        ]),
        args.max_frames,
        args.stride
    )
    test_loader = DistributedTestLoader.build(
        test_dataset, dataset_meta,
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    test(model, test_loader, dataset_meta, device, args, not args.subset_eval)


def test(model, test_loader, dataset_meta, device, args, full_eval):
    model.eval()

    evaluator = DistEvaluator(
        device, dataset_meta.test_groundtruth, args.topk, args.max_frames
    )
    return evaluator(
        model,
        test_loader.query_loader,
        test_loader.labeled_loader,
        test_loader.unlabeled_loader,
        full_eval,
        args.sim_fn
    )



if __name__ == "__main__":
    main()
