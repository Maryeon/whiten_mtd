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

from models.distill_model import archs
from datasets.svd.core import Frame, Video, MetaData
from datasets.svd.loader import DistributedFinetuneLoader, DistributedTestLoader
from datasets.svd.eval import FinetuneEvaluator
import utils
import metric


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Finetune with video-level annotations on SVD dataset.")
    parser.add_argument(
        "-a", "--arch", default="mobilenet_v2", type=str, metavar="ARCH",
        choices=list(archs.keys()), help="name of backbone model"
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
        "-d", "--embed_dim", default=256, type=int, metavar="N",
        help="embedding dimension"
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
        "-trb", "--train_batch_size", default=64, type=int, metavar="N",
        help="training batch size"
    )
    parser.add_argument(
        "-trnb", "--train_negative_batch_size", default=1024, type=int, metavar="N",
        help="training negative batch size"
    )
    parser.add_argument(
        "-teb", "--test_batch_size", default=16, type=int, metavar="N",
        help="testing batch size"
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
        "-t", "--test", action="store_true",
        help="run test"
    )
    parser.add_argument(
        "-r", "--resume", default=None, type=str, metavar="DIR",
        help="checkpoint model to resume"
    )
    parser.add_argument(
        "--sim_fn", default="cf", type=str,
        help="similarity function to use"
    )
    parser.add_argument(
        "--pretrained", default=None, type=str, metavar="DIR",
        help="pretrained model to resume"
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
        "--epochs", default=50, type=int, metavar="N",
        help="training epochs"
    )
    parser.add_argument(
        "--lr", default=1e-4, type=float, metavar="N",
        help="initial learning rate"
    )
    parser.add_argument(
        "--temp", default=0.05, type=float, metavar="N",
        help="temprature rate of contrastive loss"
    )
    parser.add_argument(
        '--wd', '--weight-decay', default=1e-6, type=float, metavar='W', dest='weight_decay',
        help='weight decay (default: 1e-6)'
    )
    parser.add_argument(
        "--eval_step", default=10, type=int, metavar="N",
        help="interval to perform evaluation"
    )
    parser.add_argument(
        "--snapshot_step", default=5, type=int, metavar="N",
        help="interval to dump checkpoint"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    if not args.test:
        with open(os.path.join(args.ckpt_dir, "config.yaml"), "w") as f:
            yaml.dump(vars(args), f)

    utils.setup_logger()
    logger = logging.getLogger("main")

    logger.info(vars(args))

    ngpus_per_node = torch.cuda.device_count()

    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    cudnn.benchmark = True

    args.gpu = gpu

    if args.dist_url == "env://" and args.rank == -1:
        args.rank = int(os.environ["RANK"])
    args.rank = args.rank * ngpus_per_node + gpu

    if args.rank == 0 and not args.test:
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

    model = Finetune(args)
    logger.info(f"Build model Finetune")
    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    n_trainable_parameters = sum([p.data.nelement() for p in model.parameters() if p.requires_grad])
    logger.info(f"Number of parameters: {n_parameters}")
    logger.info(f"Number of trainable parameters: {n_trainable_parameters}")

    model.cuda(device)

    if utils.has_batchnorms(model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])

    if args.pretrained is not None:
        checkpoint_file = args.pretrained
        if os.path.isfile(checkpoint_file):
            logger.info(f"Loading checkpoint \"{checkpoint_file}\"...")
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
        else:
            logger.error(f"=> No checkpoint found at '{checkpoint_file}'.")
            sys.exit()

        model.load_state_dict(checkpoint["state_dict"], strict=False)
        logger.info(f"Loaded checkpoint.")

    dataset_meta_cfg = utils.load_config(args.dataset_meta)
    dataset_meta = MetaData(dataset_meta_cfg)

    dataset = Video(
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
    train_loader = DistributedFinetuneLoader.build(
        dataset, dataset_meta,
        batch_size=args.train_batch_size, 
        negative_batch_size=args.train_negative_batch_size, num_workers=args.num_workers
    )

    test_loader = DistributedTestLoader.build(
        dataset, dataset_meta,
        batch_size=args.test_batch_size, num_workers=args.num_workers
    )

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * len(train_loader), eta_min=1e-6
    )

    loss_fn = FinetuneLoss(
        temp=args.temp
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

    if args.test:
        test(model, test_loader, dataset_meta, device, args, True)
        return

    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(
            model,
            train_loader,
            optimizer,
            lr_scheduler,
            loss_fn,
            device,
            epoch+1,
            args.epochs,
            args.train_batch_size
        )

        performance = None
        if (epoch + 1) == args.epochs:
            performance = test(model, test_loader, dataset_meta, device, args, True)
        elif (epoch + 1) % args.eval_step == 0:
            performance = test(model, test_loader, dataset_meta, device, args, False)

        if (epoch + 1) % args.snapshot_step == 0 and args.rank == 0:
            utils.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "loss_fn": loss_fn.state_dict()
                },
                False,
                args.ckpt_dir,
                filename=f"checkpoint_{epoch+1}.pt"
            )


def train_one_epoch(model, train_loader, optimizer, lr_scheduler, loss_fn, device, epoch, total_epochs, n_pos):
    model.train()

    rank = utils.get_rank()
    if rank == 0:
        logger = logging.getLogger("svd_finetune")
    else:
        logger = None
    metric_logger = metric.MetricLogger(logger, delimiter="    ")
    header = f'Epoch: [{epoch}/{total_epochs}]'
    log_freq = len(train_loader) // 16 if len(train_loader) >= 16 else 1

    train_loader.batch_sampler.set_epoch(epoch)

    for batch_idx, batch in metric_logger.log_every(train_loader, log_freq, header=header, iterations=len(train_loader)):
        x, n_frames, video_ids = batch

        x = x.to(device)
        n_frames = n_frames.to(device)

        x = model(x, n_frames)

        x1 = x[:n_pos]
        x2 = x[n_pos:n_pos*2]
        n = x[n_pos*2:]

        loss_dict = loss_fn(x1, x2, n)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        loss_dict = {k: v.item() for k, v in loss_dict.items()}
        metric_logger.update(**loss_dict)

    metric_logger.sync()
    if rank == 0:
        logger.info("Averaged stats: " + str(metric_logger))


def test(model, test_loader, dataset_meta, device, args, full_eval):
    model.eval()

    evaluator = FinetuneEvaluator(
        device, dataset_meta.test_groundtruth, args.topk, args.max_frames
    )
    return evaluator(
        model.module.forward_test,
        test_loader.query_loader,
        test_loader.labeled_loader,
        test_loader.unlabeled_loader,
        full_eval,
        args.sim_fn
    )


class Finetune(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.base_encoder_batch_size = 256
        self.max_frames = args.max_frames
        self.embed_dim = args.embed_dim
        self.base_encoder = archs[args.arch](pretrained=False, num_classes=0)
        for param in self.base_encoder.parameters():
            param.requires_grad = False
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            dim_feedforward=self.embed_dim,
            batch_first=True
        )
        self.lin = nn.Linear(self.base_encoder.embed_dim, self.embed_dim, bias=False)
        self.head = nn.TransformerEncoder(encoder_layer, 1)

    def forward(self, x, n_frames):
        with torch.no_grad():
            _x = []
            for i in range(0, x.size(0), self.base_encoder_batch_size):
                _x.append(self.base_encoder(x[i:i+self.base_encoder_batch_size]))
            x = torch.cat(_x, dim=0)

        v = torch.zeros(len(n_frames), self.max_frames, x.size(1), dtype=x.dtype, device=x.device)
        s = 0
        for i, nf in enumerate(n_frames):
            v[i][:nf] = x[s:s+nf]
            s += nf

        key_padding_mask = torch.zeros(v.size(0), v.size(1), dtype=torch.bool, device=v.device)
        for i in range(v.size(0)):
            key_padding_mask[i,n_frames[i]:] = True

        v = self.lin(v)
        v = self.head(v, src_key_padding_mask=key_padding_mask)

        _v = []
        for i in range(v.size(0)):
            _v.append(v[i][:n_frames[i]].mean(dim=0))
        v = torch.stack(_v, dim=0)

        return v

    def forward_test(self, x, n_frames):
        with torch.no_grad():
            _x = []
            for i in range(0, x.size(0), self.base_encoder_batch_size):
                _x.append(self.base_encoder(x[i:i+self.base_encoder_batch_size]))
            x = torch.cat(_x, dim=0)
        
        v = torch.zeros(len(n_frames), self.max_frames, x.size(1), dtype=x.dtype, device=x.device)
        s = 0
        for i, nf in enumerate(n_frames):
            v[i][:nf] = x[s:s+nf]
            s += nf

        key_padding_mask = torch.zeros(v.size(0), v.size(1), dtype=torch.bool, device=v.device)
        for i in range(v.size(0)):
            key_padding_mask[i,n_frames[i]:] = True

        v = self.lin(v)
        v = self.head(v, src_key_padding_mask=key_padding_mask)

        return v

    def train(self, mode=True):
        self.base_encoder.train(False)
        self.lin.train(mode)
        self.head.train(mode)

        return self


class FinetuneLoss(nn.Module):
    def __init__(self, temp=0.1):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.t = temp

    def forward(self, x1, x2, n):
        x1 = nn.functional.normalize(x1, p=2, dim=-1)
        x2 = nn.functional.normalize(x2, p=2, dim=-1)
        n = nn.functional.normalize(n, p=2, dim=-1)

        x1 = self.gather_with_grad(x1)
        x2 = self.gather_with_grad(x2)
        n = self.gather_with_grad(n)
        pos_sim = x1.mm(x2.t())

        target = torch.arange(pos_sim.size(0), dtype=torch.long, device=pos_sim.device)
        sim1 = torch.cat([pos_sim, x1.mm(n.t())], dim=-1)
        sim2 = torch.cat([pos_sim.t(), x2.mm(n.t())], dim=-1)
        loss = self.loss_fn(sim1.div(self.t), target) + self.loss_fn(sim2.div(self.t), target)
        loss *= utils.get_world_size()

        return {"finetune loss": loss}

    def gather_with_grad(self, x):
        world_size = utils.get_world_size()
        rank = utils.get_rank()

        gather_x = [torch.zeros_like(x) for _ in range(world_size)]
        dist.all_gather(gather_x, x)
        gather_x[rank] = x

        gather_x = torch.cat(gather_x, dim=0)

        return gather_x



if __name__ == "__main__":
    main()
