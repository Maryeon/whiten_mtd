#!/usr/bin/env python
# coding=utf-8

import torch
import numpy as np
import torch.nn as nn
import torch.distributed as dist

from utils import get_world_size, get_rank


class MultiTeacherDistillLoss(nn.Module):
    def __init__(self, st=0.05, tt=0.05, s=None, teachers=None):
        """
            st: student temperature
            tt: teacher temperature
            s: strategy
            teachers: list of teacher models
        """
        super().__init__()
        self.st = st
        self.tt = tt
        assert s is not None and teachers is not None
        self.s = s
        self.teachers = teachers
        self.distill_loss_fn = nn.KLDivLoss(reduction="batchmean")
        self.register_buffer("pos_win_count", torch.zeros(len(self.teachers), dtype=torch.long))

    def forward(self, stu1, stu2, tch1, tch2):
        """
            stu1: B x D, representations of one view
            stu2: B x D, representations of another view
            tch1: list of B x D, representatioons of one view of all teachers
            tch2: list of B x D, representatioons of another view of all teachers
        """
        stu1 = nn.functional.normalize(stu1, p=2, dim=-1)
        stu2 = nn.functional.normalize(stu2, p=2, dim=-1)
        tch1 = [nn.functional.normalize(t1, p=2, dim=-1) for t1 in tch1]
        tch2 = [nn.functional.normalize(t2, p=2, dim=-1) for t2 in tch2]

        # gather features from other devices without loss of gradients
        stu1 = self.gather_with_grad(stu1)
        stu2 = self.gather_with_grad(stu2)
        tch1 = [self.gather_with_grad(t1.contiguous()) for t1 in tch1]
        tch2 = [self.gather_with_grad(t2.contiguous()) for t2 in tch2]

        stu_sim_mat = stu1.mm(stu2.t())

        tch_sim_mats = [t1.mm(t2.t()) for t1, t2 in zip(tch1, tch2)]

        distill_loss = \
            self.get_distill_loss(
                stu_sim_mat,
                tch_sim_mats,
                tt=self.tt,
                st=self.st
            ) + \
            self.get_distill_loss(
                stu_sim_mat.t(),
                [tch_sim_mat.t() for tch_sim_mat in tch_sim_mats],
                tt=self.tt,
                st=self.st
            )
        
        # scale loss value due to the gradient mean reduction mechanism in distributed training
        # see https://pytorch.org/docs/master/notes/ddp.html
        distill_loss *= get_world_size()
        
        return {"distill loss": distill_loss}

    def get_distill_loss(self, stu_sim, tch_sims, tt=0.05, st=0.05):
        # T x N
        tch_sims = torch.stack(tch_sims, dim=0)

        if self.s == "mean":
            tch_sim = self.s_mean(tch_sims)
        elif self.s == "maxmin":
            tch_sim = self.s_maxmin(tch_sims)
        elif self.s == "maxmean":
            tch_sim = self.s_maxmean(tch_sims)
        elif self.s == "maxrand":
            tch_sim = self.s_maxrand(tch_sims)
        elif self.s == "rand":
            tch_sim = self.s_rand(tch_sims)
        
        t = nn.functional.softmax(tch_sim.div(tt), dim=-1)

        s = nn.functional.log_softmax(stu_sim.div(st), dim=-1)

        return self.distill_loss_fn(s, t)

    def s_mean(self, sims):
        return sims.mean(dim=0)

    def s_maxmin(self, sims):
        sim_diag, max_indices = sims.max(dim=0)
        max_indices = max_indices.diagonal()
        for i in range(self.pos_win_count.size(0)):
            self.pos_win_count[i] += (max_indices == i).sum()

        sim_off_diag = sims.min(dim=0)[0]
        mask = torch.eye(sims.size(1), m=sims.size(2), dtype=torch.bool, device=sims.device)
        fusion_sim = sim_diag * mask + sim_off_diag * mask.logical_not()
        return fusion_sim

    def s_maxmean(self, sims):
        sim_diag, max_indices = sims.max(dim=0)
        max_indices = max_indices.diagonal()
        for i in range(self.pos_win_count.size(0)):
            self.pos_win_count[i] += (max_indices == i).sum()

        sim_off_diag = sims.mean(dim=0)
        mask = torch.eye(sims.size(1), m=sims.size(2), dtype=torch.bool, device=sims.device)
        fusion_sim = sim_diag * mask + sim_off_diag * mask.logical_not()
        return fusion_sim

    def s_maxfix(self, sims):
        sim_diag, max_indices = sims.max(dim=0)
        max_indices = max_indices.diagonal()
        for i in range(self.pos_win_count.size(0)):
            self.pos_win_count[i] += (max_indices == i).sum()

        sim_off_diag = sims[-1]
        mask = torch.eye(sims.size(1), m=sims.size(2), dtype=torch.bool, device=sims.device)
        fusion_sim = sim_diag * mask + sim_off_diag * mask.logical_not()
        return fusion_sim

    def s_maxrand(self, sims):
        sim_diag, max_indices = sims.max(dim=0)
        max_indices = max_indices.diagonal()
        for i in range(self.pos_win_count.size(0)):
            self.pos_win_count[i] += (max_indices == i).sum()

        mask = torch.randint(0, len(sims), sims[0].size(), device=sims.device)
        mask = nn.functional.one_hot(mask, num_classes=len(sims)).permute(2, 0, 1)
        sim_off_diag = (sims * mask).sum(dim=0)
        mask = torch.eye(sims.size(1), m=sims.size(2), dtype=torch.bool, device=sims.device)
        fusion_sim = sim_diag * mask + sim_off_diag * mask.logical_not()
        return fusion_sim

    def s_rand(self, sims):
        mask = torch.randint(0, len(sims), sims[0].size(), device=sims.device)
        mask = nn.functional.one_hot(mask, num_classes=len(sims)).permute(2, 0, 1)
        fusion_sim = (sims * mask).sum(dim=0)
        return fusion_sim

    def get_ctr_loss(self, sim):
        target = torch.arange(sim.size(0), dtype=torch.long, device=sim.device)

        return self.ctr_loss_fn(sim.div(self.ct), target)

    def gather_with_grad(self, x):
        world_size = get_world_size()
        rank = get_rank()

        gather_x = [torch.zeros_like(x) for _ in range(world_size)]
        dist.all_gather(gather_x, x)
        # tensor returned by all_gather does not have gradients
        # reassign x to recover gradients
        gather_x[rank] = x

        gather_x = torch.cat(gather_x, dim=0)

        return gather_x

    def __repr__(self):
        return "positive count: " + \
        " | ".join([f"{self.teachers[i]}: {self.pos_win_count[i]/self.pos_win_count.sum()*100:.2f}%" for i in range(len(self.teachers))])