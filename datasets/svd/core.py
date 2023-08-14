import os
import torch
import random
import logging

from PIL import Image
from collections import defaultdict
from torch.utils.data.dataset import Dataset


__all__ = [
    "Frame",
    "Video",
    "MetaData"
]


class Frame(Dataset):
    """
    Args:
        root_path (str): root_path of frame directory
        frame_count (dict): dict that maps video to its frame numbers
        max_frames (int): maximum frame numbers sampled for a video
        stride (int): frame sample stride
    """
    def __init__(self, root_path, frm_cnt, transform):
        super().__init__()
        self._root_path = root_path
        self._frm_cnt = frm_cnt

        self.t = transform

    def __getitem__(self, k):
        """
        Args:
            k (tuple): video id, frame id.

        Returns:
            tuple: preprocessed frame (Tensor), video id (str)
        """
        video_id, frame_id = k
        num_of_frame = self._frm_cnt[video_id]
        assert frame_id < num_of_frame

        frame_file = f"{frame_id:04d}.jpg"
        frame_path = os.path.join(self._root_path, video_id, frame_file)
        
        frame = Image.open(frame_path)
        frame = frame.convert("RGB")
        frame = self.t(frame)
        
        return *frame, f"{video_id}/{frame_file}"


class Video(Dataset):
    """
    Args:
        root_path (str): root_path of frame directory
        frame_count (dict): dict that maps video to its frame numbers
        max_frames (int): maximum frame numbers sampled for a video
        stride (int): frame sample stride
    """
    def __init__(self, root_path, frm_cnt, transform, max_frames=60, stride=1):
        super().__init__()
        self._root_path = root_path
        self._frm_cnt = frm_cnt
        self._max_frames = max_frames
        self._stride = stride

        self.t = transform

    def __getitem__(self, k):
        """
        Args:
            k (str): video id.

        Returns:
            tuple: preprocessed frames (Tensor), video id (str)
        """
        video_id = k
        num_of_frame = self._frm_cnt[video_id]

        frame_ids = [f"{i:04d}.jpg" for i in range(0, min(num_of_frame, self._max_frames), self._stride)]

        frame_paths = []
        for frame_id in frame_ids:
            frame_path = os.path.join(self._root_path, video_id, frame_id)
            frame_paths.append(frame_path)
        
        frames = []
        for frame_path in frame_paths:
            frame = Image.open(frame_path)
            frame = frame.convert("RGB")
            frame = self.t(frame)
            frames.append(frame)
        frames = torch.stack(frames, dim=0)
        
        num_frame = frames.size(0)

        return frames, video_id, num_frame

    @property
    def max_frames(self):
        return self._max_frames


class MetaData(object):
    def __init__(self, cfg):
        self._cfg = cfg
        self._query_ids = None
        self._labeled_ids = None
        self._unlabeled_ids = None
        self._train_groundtruth = None
        self._test_groundtruth = None
        self._test_query_ids = None
        self._test_labeled_ids = None
        self._train_groups = None
        self._train_pairs = None
        self._train_ids = None
        self._frm_cnt = None

    def _load_ids(self, id_file):
        ids = list()
        assert os.path.exists(id_file), f"file {id_file} does not exist!"
        with open(id_file, 'r') as f:
            for l in f:
                l = l.strip().replace('.mp4', '')
                ids.append(l)
        return ids

    def _load_groundtruth(self, groundtruth_file):
        queries = list()
        labeled = list()
        gdtruth = defaultdict(dict)
        with open(groundtruth_file, 'r') as f:
            for l in f:
                l = l.strip().split(' ')
                qid = l[0].replace('.mp4', '')
                cid = l[1].replace('.mp4', '')
                gt = int(l[2])
                gdtruth[qid][cid] = gt

                if qid not in queries:
                    queries.append(qid)
                if cid not in labeled:
                    labeled.append(cid)
        return queries, labeled, gdtruth

    def _load_frame_cnts(self, frame_count_file):
        frame_cnts = dict()
        with open(frame_count_file, "r") as f:
            for l in f:
                l = l.strip().split(" ")
                frame_cnts[l[0]] = int(l[1])

        return frame_cnts

    @property
    def query_ids(self):
        if self._query_ids is None:
            self._query_ids = self._load_ids(self._cfg['query_id'])
        return self._query_ids

    @property
    def labeled_ids(self):
        if self._labeled_ids is None:
            self._labeled_ids = self._load_ids(self._cfg['labeled_id'])
        return self._labeled_ids

    @property
    def unlabeled_ids(self):
        if self._unlabeled_ids is None:
            self._unlabeled_ids = self._load_ids(self._cfg['unlabeled_id'])
        return self._unlabeled_ids

    @property
    def all_video_ids(self):
        if self._query_ids is None:
            self._query_ids = self._load_ids(self._cfg['query_id'])
        if self._labeled_ids is None:
            self._labeled_ids = self._load_ids(self._cfg['labeled_id'])
        if self._unlabeled_ids is None:
            self._unlabeled_ids = self._load_ids(self._cfg['unlabeled_id'])
        return (self._query_ids + self._labeled_ids + self._unlabeled_ids)

    @property
    def test_groundtruth(self):
        if self._test_groundtruth is None:
            self._test_query_ids, self._test_labeled_ids, self._test_groundtruth = \
                self._load_groundtruth(self._cfg['test_groundtruth'])
        return self._test_groundtruth

    @property
    def test_query_ids(self):
        if self._test_query_ids is None:
            self._test_query_ids, self._test_labeled_ids, self._test_groundtruth = \
                self._load_groundtruth(self._cfg['test_groundtruth'])
        return self._test_query_ids

    @property
    def test_labeled_ids(self):
        if self._test_labeled_ids is None:
            self._test_query_ids, self._test_labeled_ids, self._test_groundtruth = \
                self._load_groundtruth(self._cfg['test_groundtruth'])
        return self._test_labeled_ids

    @property
    def train_groundtruth(self):
        if self._train_groundtruth is None:
            _, _, self._train_groundtruth = self._load_groundtruth(self._cfg['train_groundtruth'])
        return self._train_groundtruth

    @property
    def train_groups(self):
        """
        train_groups should be deprecated in the future: pair matching has no transitivity.
        """
        if self._train_groups is not None:
            return self._train_groups

        groundtruth = self.train_groundtruth
        self._train_groups = list()
        for qid, cdict in groundtruth.items():
            group = None
            for g in self._train_groups:
                if qid in g:
                    group = g
                    break
            for cid, isp in cdict.items():
                if isp:
                    for g in self._train_groups:
                        if cid in g:
                            group = g
                            break
            if group is None:
                group = set()
                self._train_groups.append(group)
            group.add(qid)
            for cid, isp in cdict.items():
                if isp:
                    group.add(cid)

        return self._train_groups

    @property
    def train_pairs(self):
        if self._train_pairs is not None:
            return self._train_pairs

        groundtruth = self.train_groundtruth
        self._train_pairs = list()
        for qid, cdict in groundtruth.items():
            for cid, isp in cdict.items():
                if isp:
                    self._train_pairs.append((qid, cid))

        return self._train_pairs

    @property
    def train_ids(self):
        if self._train_ids is not None:
            return self._train_ids

        self._train_ids = list()
        for qid, cdict in self.train_groundtruth.items():
            if qid not in self._train_ids:
                self._train_ids.append(qid)
            for cid, _ in cdict.items():
                if cid not in self._train_ids:
                    self._train_ids.append(cid)

        return self._train_ids

    @property
    def frm_cnt(self):
        if self._frm_cnt is None:
            self._frm_cnt = self._load_frame_cnts(self._cfg["frame_count_file"])
        return self._frm_cnt

    @property
    def frm_root_path(self):
        return self._cfg["frame_root_path"]
