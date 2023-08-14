#!/usr/bin/env python
# coding=utf-8

import os
import torch
import torch.nn as nn
from .resnet import resnet50, resnet101
from .solar.networks import ResNetSOAs
from .delg.model import R101_DELG
from .dolg.model import DOLG

from .gem_pooling import GeneralizedMeanPooling


def resnet101_gem(path_to_pretrained_weights, gem_p=3., **kwargs):

    class ResNet101_GeM(nn.Module):
        def __init__(self, backbone, fc):
            super().__init__()
            self.backbone = backbone
            self.fc = fc

        def forward(self, x):
            x = self.backbone(x)
            x = nn.functional.normalize(x, p=2, dim=-1)
            x = self.fc(x)
            return x

    pretrained_weights = os.path.join(path_to_pretrained_weights, "gl18-tl-resnet101-gem-w-a4d43db.pt")
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    backbone = resnet101(pretrained=False, num_classes=0)
    backbone.avgpool = GeneralizedMeanPooling(gem_p)
    model = ResNet101_GeM(
        backbone,
        nn.Linear(backbone.embed_dim, checkpoint["meta"]["outputdim"])
    )
    model.embed_dim = checkpoint["meta"]["outputdim"]

    state_dict = checkpoint["state_dict"]
    fc_weight = state_dict.pop("fc.weight")
    fc_bias = state_dict.pop("fc.bias")

    model.backbone.load_state_dict(state_dict, strict=True)
    model.fc.load_state_dict({"weight": fc_weight, "bias": fc_bias}, strict=True)

    return model


def resnet101_ap_gem(path_to_pretrained_weights, gem_p=3., **kwargs):
    pretrained_weights = os.path.join(path_to_pretrained_weights, "Resnet101-AP-GeM-LM18.pt")
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    model = resnet101(pretrained=False, num_classes=checkpoint["model_options"]["out_dim"])
    model.avgpool = GeneralizedMeanPooling(gem_p)
    
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith("module"):
            if k != "module.adpool.p":
                state_dict[k[len("module."):]] = state_dict[k]
        del state_dict[k]

    model.load_state_dict(state_dict, strict=True)
    model.embed_dim = checkpoint["model_options"]["out_dim"]

    return model


def resnet101_solar(path_to_pretrained_weights, gem_p=3., **kwargs):

    class ResNet101_SOLAR(nn.Module):
        def __init__(self, backbone, fc, gem_p):
            super().__init__()
            self.backbone = backbone
            self.fc = fc
            self.pool = GeneralizedMeanPooling(gem_p)

        def forward(self, x):
            x = self.backbone(x)
            x = self.pool(x)
            x = nn.functional.normalize(x, p=2, dim=-1)
            x = self.fc(x)
            return x

    pretrained_weights = os.path.join(path_to_pretrained_weights, "resnet101-solar-best.pth")
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    backbone = ResNetSOAs()
    model = ResNet101_SOLAR(
        backbone,
        nn.Linear(backbone.embed_dim, checkpoint["meta"]["outputdim"]),
        gem_p
    )
    model.embed_dim = checkpoint["meta"]["outputdim"]

    state_dict = checkpoint["state_dict"]
    state_dict.pop("pool.p")
    fc_weight = state_dict.pop("whiten.weight")
    fc_bias = state_dict.pop("whiten.bias")

    for k in list(state_dict.keys()):
        state_dict[k[len("features."):]] = state_dict.pop(k)

    model.backbone.load_state_dict(state_dict, strict=True)
    model.fc.load_state_dict({"weight": fc_weight, "bias": fc_bias}, strict=True)

    return model


def resnet101_delg(path_to_pretrained_weights, pretrained=True, gem_p=3.):
    pretrained_weights = os.path.join(path_to_pretrained_weights, "r101_delg_s512.pyth")
    model = R101_DELG()

    if pretrained:
        checkpoint = torch.load(pretrained_weights, map_location="cpu")
        state_dict = checkpoint['model_state']
        state_dict["globalmodel.head.pool.p"] = torch.tensor([gem_p], dtype=torch.float32)
        for k in list(state_dict.keys()):
            if not k.startswith("globalmodel"):
                state_dict.pop(k)
        model.load_state_dict(state_dict, strict=True)

    return model


def resnet101_dolg(path_to_pretrained_weights, pretrained=True, gem_p=3.):
    pretrained_weights = os.path.join(path_to_pretrained_weights, "r101_dolg_512.pyth")
    model = DOLG()

    if pretrained:
        checkpoint = torch.load(pretrained_weights, map_location="cpu")
        state_dict = checkpoint['model_state']
        state_dict.pop("pool1.p")
        state_dict["pool_g.p"] = torch.tensor([gem_p], dtype=torch.float32)
        model.load_state_dict(state_dict, strict=True)

    return model


def mocov3(path_to_pretrained_weights, pretrained=True, with_head=False, gem_p=3.):
    pretrained_moco = os.path.join(path_to_pretrained_weights, 'mocov3_1000ep.pth.tar')
    
    model = resnet50(pretrained=False, num_classes=0)
    model.avgpool = GeneralizedMeanPooling(gem_p)

    if with_head:
        model.fc = nn.Sequential(
            # projector
            nn.Linear(2048, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 256, bias=False),
            nn.BatchNorm1d(256, affine=False),
            # predictor
            nn.Linear(256, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 256, bias=False)
        )
        model.embed_dim = 256

    if pretrained:
        checkpoint = torch.load(pretrained_moco, map_location="cpu")
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith("module.base_encoder"):
                if k.startswith("module.base_encoder.fc") and not with_head:
                    state_dict.pop(k)
                    continue
                state_dict[k[len("module.base_encoder."):]] = state_dict.pop(k)
            elif k.startswith("module.predictor") and with_head:
                i = int(k[len("module.predictor."):][0])
                new_k = "fc."+str(i + 5)+k[len("module.predictor.0"):]
                state_dict[new_k] = state_dict.pop(k)
            else:
                state_dict.pop(k) 
        model.load_state_dict(state_dict, strict=True)

    return model


def barlowtwins(path_to_pretrained_weights, pretrained=True, with_head=False, gem_p=3.):
    model = resnet50(pretrained=False, num_classes=0)
    model.avgpool = GeneralizedMeanPooling(gem_p)

    if with_head:
        model.fc = nn.Sequential(
            nn.Linear(2048, 8192, bias=False),
            nn.BatchNorm1d(8192),
            nn.ReLU(inplace=True),
            nn.Linear(8192, 8192, bias=False),
            nn.BatchNorm1d(8192),
            nn.ReLU(inplace=True),
            nn.Linear(8192, 8192, bias=False),
            nn.BatchNorm1d(8192, affine=False)
        )
        model.embed_dim = 8192

    if pretrained:
        checkpoint_file = os.path.join(path_to_pretrained_weights, 'barlowtwins_full.pth.tar')
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        state_dict = checkpoint["model"]
        for k in list(state_dict.keys()):
            if k.startswith("module.backbone"):
                state_dict[k[len("module.backbone."):]] = state_dict.pop(k)
            elif k.startswith("module.projector") and with_head:
                state_dict["fc."+k[len("module.projector."):]] = state_dict.pop(k)
            elif k.startswith("module.bn") and with_head:
                state_dict["fc.7."+k[len("module.bn."):]] = state_dict.pop(k)
            else:
                state_dict.pop(k)
        model.load_state_dict(state_dict, strict=True)

    return model


teacher_models = {
    "mocov3": mocov3,
    "barlowtwins": barlowtwins,
    "resnet101_gem": resnet101_gem,
    "resnet101_ap_gem": resnet101_ap_gem,
    "resnet101_solar": resnet101_solar,
    "resnet101_delg": resnet101_delg,
    "resnet101_dolg": resnet101_dolg,
}


if __name__ == "__main__":
    model = resnet101_delg()