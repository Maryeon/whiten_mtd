import os
import torch.nn as nn
from .config import cfg
from .resnet import ResNet


class R101_DELG(nn.Module):
    def __init__(self):
        super().__init__()
        cfg.merge_from_file(os.path.join(os.path.dirname(__file__), "r101_delg_config.yaml"))
        cfg.freeze()
        self.globalmodel = ResNet()
        self.embed_dim = cfg.MODEL.HEADS.REDUCTION_DIM
    
    def forward(self, x):
        global_feature, _ = self.globalmodel(x)
        
        return global_feature


class R50_DELG(nn.Module):
    def __init__(self):
        super().__init__()
        cfg.merge_from_file(os.path.join(os.path.dirname(__file__), "r50_delg_config.yaml"))
        cfg.freeze()
        self.globalmodel = ResNet()
        self.embed_dim = cfg.MODEL.HEADS.REDUCTION_DIM
    
    def forward(self, x):
        global_feature, _ = self.globalmodel(x)
        
        return global_feature