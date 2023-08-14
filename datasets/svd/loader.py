import os
import math
import yaml
import random
import logging
from functools import reduce

import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader

from utils import get_rank, get_world_size


class DistributedDistillLoader(object):
    def __init__(
        self, dataset, train_ids, frm_cnt, batch_size=64, num_workers=4):
        
        self._rank = get_rank()
        self._world_size = get_world_size()
        self._dataset = dataset

        self._train_ids = train_ids
        self._frm_cnt = frm_cnt

        # Samplers
        self._sampler = DistributedFrameSampler(
            self._rank, self._world_size,
            batch_size, self._train_ids, self._frm_cnt,
            seed=0, shuffle=True, drop_last=True)
        
        # DataLoaders
        self._train_loader = DataLoader(
            self._dataset,
            batch_sampler=self._sampler,
            collate_fn=self._collate_fn,
            num_workers=num_workers,
            pin_memory=True
        )

    def _collate_fn(self, batch):
        frames1, frames2, video_ids = zip(*batch)
        
        frames1 = torch.stack(frames1, dim=0)
        frames2 = torch.stack(frames2, dim=0)

        return frames1, frames2, video_ids

    @property
    def batch_sampler(self):
        return self._train_loader.batch_sampler

    def __len__(self):
        return len(self._train_loader)

    def __iter__(self):
        return iter(self._train_loader)

    @classmethod
    def build(cls, dataset, meta, batch_size=64, num_workers=4):
        # train_ids = meta.train_ids + meta.unlabeled_ids
        train_ids = meta.train_ids

        if get_rank() == 0:
            objects = [train_ids]
        else:
            objects = [None]
        dist.broadcast_object_list(
            objects, src=0, 
            device=torch.device("cuda:"+str(torch.cuda.current_device()))
        )
        train_ids = objects[0]
        
        return cls(dataset, train_ids, meta.frm_cnt, batch_size=batch_size, num_workers=num_workers)


class DistributedTestLoader(object):
    def __init__(self, dataset, 
        query_ids, labeled_ids, unlabeled_ids, 
        batch_size=64, num_workers=4):

        self._rank = get_rank()
        self._world_size = get_world_size()
        self._dataset = dataset

        self._test_query_sampler = DistributedTestQuerySampler(
            self._rank, self._world_size, batch_size, query_ids
        )
        self._test_labeled_sampler = DistributedTestLabeledSampler(
            self._rank, self._world_size, batch_size, labeled_ids
        )
        self._test_unlabeled_sampler = DistributedUnLabeledSampler(
            self._rank, self._world_size, batch_size, unlabeled_ids
        )

        self._query_loader = DataLoader(
            self._dataset,
            batch_sampler=self._test_query_sampler,
            collate_fn=self._collate_fn,
            num_workers=num_workers,
            pin_memory=True
        )
        self._labeled_loader = DataLoader(
            self._dataset,
            batch_sampler=self._test_labeled_sampler,
            collate_fn=self._collate_fn,
            num_workers=num_workers,
            pin_memory=True
        )
        self._unlabeled_loader = DataLoader(
            self._dataset,
            batch_sampler=self._test_unlabeled_sampler,
            collate_fn=self._collate_fn,
            num_workers=num_workers,
            pin_memory=True
        )

    def _collate_fn(self, batch):
        """
        Args:
            batch (tuple): list of frames (Tensor(3 x h x w)), list of video ids (str)
        Returns:
            tuple:
                concatenated frame tensors (Tensor)
                number of frames per video (Tensor)
                video ids(str)
        """
        data, video_ids, num_of_frames = zip(*batch)
        
        num_of_frames = torch.tensor(num_of_frames, dtype=torch.long)

        data = torch.cat(data, dim=0)

        return data, num_of_frames, video_ids

    @property
    def query_loader(self):
        return self._query_loader

    @property
    def labeled_loader(self):
        return self._labeled_loader

    @property
    def unlabeled_loader(self):
        return self._unlabeled_loader

    @classmethod
    def build(cls, dataset, meta, batch_size=64, num_workers=4):
        test_groundtruth = meta.test_groundtruth
        query_ids = meta.test_query_ids
        labeled_ids = meta.test_labeled_ids
        unlabeled_ids = meta.unlabeled_ids

        if get_rank() == 0:
            objects = [query_ids, labeled_ids, unlabeled_ids]
        else:
            objects = [None, None, None]
        dist.broadcast_object_list(
            objects, src=0, 
            device=torch.device("cuda:"+str(torch.cuda.current_device()))
        )
        query_ids, labeled_ids, unlabeled_ids = objects
        
        return cls(dataset, query_ids, labeled_ids, unlabeled_ids, batch_size, num_workers)


class DistributedFinetuneLoader(object):
    def __init__(self, dataset, 
        groups, negative_ids, 
        batch_size=64, negative_batch_size=1024, num_workers=4):

        self._rank = get_rank()
        self._world_size = get_world_size()
        self._dataset = dataset
        self._batch_size = batch_size
        self._negative_batch_size = negative_batch_size

        self._sampler = DistributedFinetuneSampler(
            self._rank, self._world_size, self._batch_size, self._negative_batch_size,
            groups, negative_ids
        )

        self._data_loader = DataLoader(
            self._dataset,
            batch_sampler=self._sampler,
            collate_fn=self._collate_fn,
            num_workers=num_workers,
            pin_memory=True
        )

    def _collate_fn(self, batch):
        """
        Args:
            batch (tuple): list of frames (Tensor(3 x h x w)), list of video ids (str)
        Returns:
            tuple:
                concatenated frame tensors (Tensor)
                number of frames per video (Tensor)
                video ids(str)
        """
        data, video_ids, num_of_frames = zip(*batch)
        
        num_of_frames = torch.tensor(num_of_frames, dtype=torch.long)

        data = torch.cat(data, dim=0)

        return data, num_of_frames, video_ids

    def __iter__(self):
        return iter(self._data_loader)

    def __len__(self):
        return len(self._data_loader)

    @property
    def batch_sampler(self):
        return self._data_loader.batch_sampler

    @classmethod
    def build(cls, dataset, meta, batch_size=64, negative_batch_size=1024, num_workers=4):
        train_ids = meta.train_ids
        groups = meta.train_groups
        negative_ids = list(reduce(lambda x,y: x-y, [set(train_ids)]+groups))
        negative_ids += meta.unlabeled_ids
        groups = list(map(list, groups))

        if get_rank() == 0:
            objects = [groups, negative_ids]
        else:
            objects = [None, None]
        dist.broadcast_object_list(
            objects, src=0, 
            device=torch.device("cuda:"+str(torch.cuda.current_device()))
        )
        groups, negative_ids = objects
        
        return cls(dataset, groups, negative_ids, batch_size, negative_batch_size, num_workers)


class DistributedSampler(Sampler):
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


class DistributedFrameSampler(DistributedSampler):
    def __init__(self, rank, num_replicas, batch_size, ids, frm_cnt, seed=0, shuffle=True, drop_last=True):
        super(DistributedFrameSampler, self).__init__(
            rank, num_replicas, batch_size, ids,
            seed=seed, shuffle=shuffle, drop_last=drop_last
        )
        random.seed(rank)
        self.frm_cnt = frm_cnt

        logger = logging.getLogger("datasets.svd.dist_frame_sampler."+str(rank))
        logger.info(self)

    def __str__(self):
        return f"| Distributed Frame Sampler | {self._num_samples} samples | iters {len(self)} | {self._batch_size} per batch"

    def __iter__(self):
        subset = self._get_subset()
        for i in range(0, len(subset), self._batch_size):
            video_ids = subset[i:i+self._batch_size]
            n_frames = [self.frm_cnt[i] for i in video_ids]
            frame_ids = [random.randint(0, n_frame-1) for n_frame in n_frames]
            yield list(zip(video_ids, frame_ids))


class DistributedSeqSampler(Sampler):
    def __init__(self, rank, num_replicas, batch_size, ids):
        self._rank = rank
        self._num_replicas = num_replicas
        self._batch_size =  batch_size
        self._id_list = ids[self._rank:len(ids):self._num_replicas]

        logger = logging.getLogger("datasets.svd.dist_loader.dist_seq_sampler."+str(self._rank))
        logger.info(self)

    def __len__(self):
        return math.ceil(len(self._id_list) / self._batch_size)

    def __iter__(self):
        for i in range(0, len(self._id_list), self._batch_size):
            x = self._id_list[i:i+self._batch_size]
            yield x


class DistributedTestQuerySampler(DistributedSeqSampler):
    def __str__(self):
        return f"| Test Query Sampler | {len(self._id_list)} queries | iters {self.__len__()} | batch size {self._batch_size}"


class DistributedTestLabeledSampler(DistributedSeqSampler):
    def __str__(self):
        return f"| Test Labeled Sampler | {len(self._id_list)} labeled videos | iters {self.__len__()} | batch size {self._batch_size}"


class DistributedUnLabeledSampler(DistributedSeqSampler):
    def __str__(self):
        return f"| Test UnLabeled Sampler | {len(self._id_list)} unlabeled videos | iters {self.__len__()} | batch size {self._batch_size}"


class DistributedFinetuneSampler(DistributedSampler):
    def __init__(self, rank, num_replicas, batch_size, negative_batch_size, 
        train_ids, negative_ids, seed=0, shuffle=True, drop_last=True):
        super().__init__(
            rank, num_replicas, batch_size, train_ids,
            seed=seed, shuffle=shuffle, drop_last=drop_last
        )
        self._negative_ids = negative_ids
        self._negative_batch_size = negative_batch_size

        logger = logging.getLogger("datasets.svd.dist_finetune_sampler."+str(rank))
        logger.info(self)

    def __str__(self):
        return f"| Distributed Finetune Sampler | {self._num_samples} samples | iters {len(self)} | {self._batch_size} per batch"

    def __len__(self):
        return self._num_samples//self._batch_size

    def _get_negative_subset(self):
        if self._drop_last and len(self._negative_ids) % self._num_replicas != 0:
            num_samples = math.ceil(
                (len(self._negative_ids) - self._num_replicas) / self._num_replicas
            )
        else:
            num_samples = math.ceil(len(self._negative_ids) / self._num_replicas)
        total_size = num_samples * self._num_replicas

        if self._shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self._seed + self._epoch * 2)
            indices = torch.randperm(len(self._negative_ids), generator=g).tolist()
        else:
            indices = list(range(len(self._negative_ids)))

        if not self._drop_last:
            # add extra samples to make it evenly divisible
            padding_size = total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:total_size]
        assert len(indices) == total_size

        # subsample
        indices = indices[self._rank:total_size:self._num_replicas]
        assert len(indices) == num_samples

        return [self._negative_ids[i] for i in indices]

    def __iter__(self):
        groups = self._get_subset()
        negative_ids = self._get_negative_subset()
        n_ptr = 0
        for i in range(len(self)):
            positive_pairs = [random.sample(g, 2) for g in groups[i*self._batch_size:(i+1)*self._batch_size]]
            negative_samples = negative_ids[n_ptr:n_ptr+self._negative_batch_size]
            n_samples = self._negative_batch_size - len(negative_samples)
            if n_samples > 0:
                negative_samples += negative_ids[:n_samples]
            n_ptr = (n_ptr + self._negative_batch_size) % len(negative_ids)
            yield \
                [pair[0] for pair in positive_pairs] + \
                [pair[1] for pair in positive_pairs] + \
                negative_samples