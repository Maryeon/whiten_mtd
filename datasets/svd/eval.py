import os
import json
import torch
import logging
import torch.nn as nn
import torch.distributed as dist
from collections import defaultdict, OrderedDict
from metric import APScorer, SmoothedValue, MetricLogger
from utils import get_rank, get_world_size


__all__ = [
    "DistEvaluator",
    "MultiTeacherEvaluator"
]


class DistEvaluator(object):
    def __init__(self, device, test_groundtruth, topk, max_frames):
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.device = device

        self.gdtruth = test_groundtruth
        self.ranks = defaultdict(list)

        self.max_frames = max_frames

        if self.rank == 0:
            self.topk = topk
            self.scorer = APScorer()
            self.res = OrderedDict()
            for k in self.topk:
                self.res["top-"+str(k)] = SmoothedValue(window_size=None)
            self.res["top-inf"] = SmoothedValue(window_size=None)
            self.logger = logging.getLogger("svd.dist_eval."+str(self.rank))

    @torch.no_grad()
    def __call__(self, 
        model, query_loader, labeled_loader, unlabeled_loader,
        full_eval, sim_fn, dump_to=None
    ):
        
        if self.rank == 0:
            self.logger.info("Start evaluation.")
            self.logger.info("Processing Query Samples...")

        dist.barrier()

        metric_logger = MetricLogger(delimiter="  ", logger=self.logger if self.rank == 0 else None)

        qfeats = []
        qlens = []
        qids = []
        
        for idx, batch in metric_logger.log_every(query_loader, log_freq=16):
            feats, lens, ids = self.forward_query(model, batch, sim_fn)

            qfeats.append(feats)
            qlens.append(lens)
            qids += ids

        qfeats = torch.cat(qfeats, dim=0)
        qlens = torch.cat(qlens, dim=0)
        self.gather_query(qfeats, qlens, qids)

        if self.rank == 0:
            self.logger.info(f"Query gathered to all workers, totally {len(self.qfeats)} queries.")

            self.logger.info("Processing Labaled Samples.")

        for idx, batch in metric_logger.log_every(labeled_loader, log_freq=16):
            feats, lens, ids = self.forward_labeled(model, batch, sim_fn)
            sims = self.cal_sim(feats, lens, sim_fn)

            self.handle_labeled(sims, ids)

        if full_eval:

            if self.rank == 0:
                self.logger.info("Processing Features of UnLabaled Samples.")

            for idx, batch in metric_logger.log_every(unlabeled_loader, log_freq=1024):
                feats, lens, ids = self.forward_unlabeled(model, batch, sim_fn)
                sims = self.cal_sim(feats, lens, sim_fn)

                self.handle_unlabeled(sims, ids)

        self.sync_ranks()

        # eval result
        if self.rank == 0:
            evalr = self.score()
            self.logger.info(" | ".join([f"{k} mAP: {v:.4f}" for k, v in evalr.items()]))

            if dump_to is not None:
                # dump evaluation result to file
                self.dump(dump_to)

            return evalr
        else:
            return None

    def forward_batch(self, model, batch, sim_fn):
        frames, n_frames, ids = batch
        frames = frames.to(self.device)
        n_frames = n_frames.to(self.device)

        frames = model(frames)

        if sim_fn == "fme" or sim_fn == "fmx":
            _frames= []
            i = 0
            for nf in n_frames:
                if sim_fn == "fme":
                    _frames.append(frames[i:i+nf].mean(dim=0))
                elif sim_fn == "fmx":
                    _frames.append(frames[i:i+nf].max(dim=0)[0])
                i += nf
            frames = torch.stack(_frames, dim=0)
        elif sim_fn == "sme" or sim_fn == "smx" or sim_fn == "cf":
            _frames = torch.zeros(
                len(n_frames), self.max_frames, frames.size(1), 
                dtype=frames.dtype, device=frames.device
            )
            s = 0
            for i, nf in enumerate(n_frames):
                _frames[i][:nf] = frames[s:s+nf]
                s += nf
            frames = _frames
        else:
            raise NotImplementedError(f"{sim_fn} not implemented.")

        frames = nn.functional.normalize(frames, p=2, dim=-1)

        return frames, n_frames, ids

    def forward_query(self, model, batch, sim_fn):
        return self.forward_batch(model, batch, sim_fn)

    def forward_labeled(self, model, batch, sim_fn):
        return self.forward_batch(model, batch, sim_fn)

    def forward_unlabeled(self, model, batch, sim_fn):
        return self.forward_batch(model, batch, sim_fn)

    def cal_sim(self, feats, lens, sim_fn):
        if sim_fn == "fme" or sim_fn == "fmx":
            return self.qfeats.mm(feats.t())

        # Q x C x F x F
        sims = self.qfeats.unsqueeze(1).matmul(feats.transpose(1, 2))
        mask = torch.ones_like(sims, dtype=torch.bool)
        for i in range(mask.size(0)):
            for j in range(mask.size(1)):
                mask[i, j, self.qlens[i]:, :] = False
                mask[i, j, :, lens[j]:] = False

        if sim_fn == "cf":
            sims = self.cal_sim_chamfer(sims, mask)
        elif sim_fn == "sme":
            sims = self.cal_sim_mean(sims, mask)
        elif sim_fn == "smx":
            sims = self.cal_sim_max(sims, mask)
        else:
            raise NotImplementedError(f"{sim_fn} is not implemented.")

        return sims

    def cal_sim_chamfer(self, sim_mat, mask):
        sim_mat.masked_fill_(mask.logical_not(), float("-inf"))
        sim_mat = sim_mat.max(dim=-1)[0]
        is_inf = sim_mat.isinf()
        sim_mat = sim_mat.masked_fill_(is_inf, 0)
        sim_mat = sim_mat.sum(dim=-1).div(is_inf.logical_not().sum(dim=-1))

        return sim_mat

    def cal_sim_max(self, sim_mat, mask):
        sim_mat.masked_fill_(mask.logical_not(), float("-inf"))
        sim_mat = sim_mat.flatten(2).max(dim=-1)[0]
        return sim_mat

    def cal_sim_mean(self, sim_mat, mask):
        sim_mat.masked_fill_(mask.logical_not(), 0)
        sim_mat = sim_mat.sum(dim=(2,3)).div(mask.sum(dim=(2,3)))
        return sim_mat

    def gather_query(self, feats, lens, ids):
        num_queries_gather = [torch.tensor(0, dtype=torch.long, device=self.device) for _ in range(self.world_size)]
        dist.all_gather(num_queries_gather, torch.tensor(feats.size(0), dtype=torch.long, device=self.device), async_op=False)
        
        self.qfeats = [
            torch.zeros([num_queries_gather[i],*feats.size()[1:]], dtype=feats.dtype, device=self.device)
            for i in range(self.world_size)
        ]
        self.qfeats[self.rank] = feats
        for i in range(self.world_size):
            dist.broadcast(self.qfeats[i], src=i, async_op=False)
        self.qfeats = torch.cat(self.qfeats, dim=0)

        self.qlens = [
            torch.zeros(num_queries_gather[i], dtype=lens.dtype, device=self.device)
            for i in range(self.world_size)
        ]
        self.qlens[self.rank] = lens
        for i in range(self.world_size):
            dist.broadcast(self.qlens[i], src=i, async_op=False)
        self.qlens = torch.cat(self.qlens, dim=0)

        ids_gather = [None for _ in range(self.world_size)]
        dist.all_gather_object(ids_gather, ids)
        self.qids = sum(ids_gather, [])

    def sync_ranks(self):
        for qid in self.qids:
            ranks_gather = [None for _ in range(self.world_size)]
            if self.rank == 0:
                dist.gather_object(self.ranks[qid], object_gather_list=ranks_gather, dst=0)
                self.ranks[qid] = sum(ranks_gather, [])
            else:
                dist.gather_object(self.ranks[qid], object_gather_list=None, dst=0)
            dist.barrier()

    def handle_labeled(self, sims, ids):
        sims = sims.cpu().tolist()
        for i, qid in enumerate(self.qids):
            sim = []
            cids = []
            for j, cid in enumerate(ids):
                if cid in self.gdtruth[qid]:
                    sim.append(sims[i][j])
                    cids.append(cid)
            self.ranks[qid] += list(zip(sim, cids, [self.gdtruth[qid][cid] for cid in cids]))

    def handle_unlabeled(self, sims, ids):
        sims = sims.cpu().tolist()
        for i, qid in enumerate(self.qids):
            self.ranks[qid] += list(zip(sims[i], ids, [0]*len(ids)))

    def score(self):
        self.aps = defaultdict(OrderedDict)
        for qid in self.qids:
            self.ranks[qid].sort(key=lambda x: x[0], reverse=True)
            for k in self.topk:
                sorted_labels = []
                for i in self.ranks[qid][:k]:
                    sorted_labels.append(i[2])
                ap = self.scorer.score(sorted_labels)
                self.aps[qid]["top-"+str(k)] = ap
                self.res["top-"+str(k)].update(ap)

            sorted_labels = []
            for i in self.ranks[qid]:
                sorted_labels.append(i[2])
            ap = self.scorer.score(sorted_labels)
            self.aps[qid]["top-inf"] = ap
            self.res["top-inf"].update(ap)

        return {k: v.avg for k, v in self.res.items()}

    def dump(self, dump_to, topk=100):
        record = list()
        for qid in self.qids:
            item = {'qid': qid, 'ap': self.aps[qid], 'ranking': [], 'positive': []}
            for score, vid, label in self.ranks[qid][:topk]:
                d = {
                    'score': score,
                    'id': vid,
                    'label': label
                }
                item['ranking'].append(d)
            
            pos = [cid for cid, ispos in self.gdtruth[qid].items() if ispos]
            for i, (score, vid, _) in enumerate(self.ranks[qid]):
                if vid in pos:
                    item['positive'].append(
                        {
                            "id": vid, 
                            "rank": i,
                            "score": score
                        }
                    )
            record.append(item)
        record.sort(key=lambda x: x['ap']['top-inf'])
        os.makedirs(dump_to, exist_ok=True)
        with open(os.path.join(dump_to, "ret_res.json"), 'w') as f:
            f.write(json.dumps(record, sort_keys=True, indent=4))


class MultiTeacherEvaluator(DistEvaluator):
    def forward_batch(self, model, batch, *args, **kwargs):
        frames, lens, ids = batch
        frames = frames.to(self.device)
        lens = lens.to(self.device)

        frames = model(frames, *args, **kwargs)
        for i in range(len(frames)):
            _frames= []
            s = 0
            for l in lens:
                _frames.append(frames[i][s:s+l].mean(dim=0))
                # _frames.append(frames[i][s:s+l].max(dim=0)[0])
                s += l
            frames[i] = torch.stack(_frames, dim=0)
            frames[i] = nn.functional.normalize(frames[i], p=2, dim=-1)

        frames = torch.cat(frames, dim=-1)

        return frames, lens, ids


class FinetuneEvaluator(DistEvaluator):
    def forward_batch(self, model, batch, sim_fn):
        x, n_frames, ids = batch
        x = x.to(self.device)
        n_frames = n_frames.to(self.device)

        x = model(x, n_frames)

        if sim_fn == "fme" or sim_fn == "fmx":
            _x = []
            for i in range(x.size(0)):
                if sim_fn == "fme":
                    _x.append(x[i][:n_frames[i]].mean(dim=0))
                elif sim_fn == "fmx":
                    _x.append(x[i][:n_frames[i]].max(dim=0)[0])
            x = torch.stack(_x, dim=0)

        x = nn.functional.normalize(x, p=2, dim=-1)

        return x, n_frames, ids