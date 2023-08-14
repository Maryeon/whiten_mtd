# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration file (powered by YACS)."""

import argparse
import os
import sys

from yacs.config import CfgNode as CfgNode


# Global config object
_C = CfgNode()

# Example usage:
#   from core.config import cfg
cfg = _C


# ------------------------------------------------------------------------------------ #
# Model options
# ------------------------------------------------------------------------------------ #
_C.MODEL = CfgNode()

# Model type
_C.MODEL.TYPE = ""

# Number of weight layers
_C.MODEL.DEPTH = 0

# Number of classes
_C.MODEL.NUM_CLASSES = 10

# Loss function (see pycls/models/loss.py for options)
_C.MODEL.LOSSES = CfgNode()
_C.MODEL.LOSSES.NAME = "cross_entropy"

# ------------------------------------------------------------------------------------ #
# Heads options
# ------------------------------------------------------------------------------------
_C.MODEL.HEADS = CfgNode()
_C.MODEL.HEADS.NAME = "LinearHead"
# Normalization method for the convolution layers.
# Number of identity
_C.MODEL.HEADS.NUM_CLASSES = 1000
# Input feature dimension
_C.MODEL.HEADS.IN_FEAT = 2048
# Reduction dimension in head
_C.MODEL.HEADS.REDUCTION_DIM = 512
# Pooling layer type
_C.MODEL.HEADS.POOL_LAYER = "avgpool"
# Classification layer type
_C.MODEL.HEADS.CLS_LAYER = "linear"  
# Margin and Scale for margin-based classification layer
_C.MODEL.HEADS.MARGIN = 0.15
_C.MODEL.HEADS.SCALE = 128


# ------------------------------------------------------------------------------------ #
# ResNet options
# ------------------------------------------------------------------------------------ #
_C.RESNET = CfgNode()

# Transformation function (see pycls/models/resnet.py for options)
_C.RESNET.TRANS_FUN = "basic_transform"

# Number of groups to use (1 -> ResNet; > 1 -> ResNeXt)
_C.RESNET.NUM_GROUPS = 1

# Width of each group (64 -> ResNet; 4 -> ResNeXt)
_C.RESNET.WIDTH_PER_GROUP = 64

# Apply stride to 1x1 conv (True -> MSRA; False -> fb.torch)
_C.RESNET.STRIDE_1X1 = True


# ------------------------------------------------------------------------------------ #
# Batch norm options
# ------------------------------------------------------------------------------------ #
_C.BN = CfgNode()

# BN epsilon
_C.BN.EPS = 1e-5

# BN momentum (BN momentum in PyTorch = 1 - BN momentum in Caffe2)
_C.BN.MOM = 0.1

# Precise BN stats
_C.BN.USE_PRECISE_STATS = False
_C.BN.NUM_SAMPLES_PRECISE = 1024

# Initialize the gamma of the final BN of each block to zero
_C.BN.ZERO_INIT_FINAL_GAMMA = False

# Use a different weight decay for BN layers
_C.BN.USE_CUSTOM_WEIGHT_DECAY = False
_C.BN.CUSTOM_WEIGHT_DECAY = 0.0


# ------------------------------------------------------------------------------------ #
# Memory options
# ------------------------------------------------------------------------------------ #
_C.MEM = CfgNode()

# Perform ReLU inplace
_C.MEM.RELU_INPLACE = True