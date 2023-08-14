#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
from collections import OrderedDict

from .teachers import teacher_models
from .pca_layer import pca_layers
from .resnet import resnet18, resnet34
from .gem_pooling import GeneralizedMeanPooling


__all__ = [
    "DistillModel",
    "MultiTeacher"
]

archs = {
    "resnet18": resnet18,
    "resnet34": resnet34
}


class MultiTeacher(nn.Module):
    def __init__(self, path_to_pretrained_weights, *teachers, p=3., embed_dim=512):
        super().__init__()
        self.teachers = teachers
        encoders = list()
        for teacher in self.teachers:
            encoders.append(teacher_models[teacher](path_to_pretrained_weights, pretrained=True, gem_p=p))
        self.encoders = nn.ModuleList(encoders)
        self.embed_dims = [encoder.embed_dim for encoder in self.encoders]
        
        norm_layers = list()
        for teacher in self.teachers:
            norm_layers.append(pca_layers[teacher](path_to_pretrained_weights, embed_dim=embed_dim))
        self.norm_layers = nn.ModuleList(norm_layers)
        self.embed_dims = [norm_layer.dim2 for norm_layer in self.norm_layers]
    
    def forward(self, x):
        out = list()
        for teacher, encoder in zip(self.teachers, self.encoders):
            if teacher.endswith("delg") or teacher.endswith("dolg"):
                out.append(encoder(x[:, (2,1,0)]))
            else:
                out.append(encoder(x))
        
        out = [nn.functional.normalize(o, p=2, dim=-1) for o in out]
        out = [norm_layer(o) for norm_layer, o in zip(self.norm_layers, out)]

        return out


class MultiTeacherDistillModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.base_encoder = archs[args.arch](pretrained=True, num_classes=args.embed_dim)
        self.base_encoder.avgpool = GeneralizedMeanPooling(args.p)
        self.base_encoder.fc = nn.Linear(self.base_encoder.embed_dim, args.embed_dim)
        self.embed_dim = self.base_encoder.embed_dim
        self.teacher_encoders = MultiTeacher(args.path_to_pretrained_weights, *args.teachers, p=args.p, embed_dim=args.embed_dim)
        for param in self.teacher_encoders.parameters():
            param.requires_grad = False

    def forward(self, x1, x2):
        stu1 = self.base_encoder(x1)
        stu2 = self.base_encoder(x2)

        with torch.no_grad():
            tch1 = self.teacher_encoders(x1)
            tch2 = self.teacher_encoders(x2)

        return stu1, stu2, tch1, tch2

    # teacher models's weights are never trained
    def train(self, mode=True):
        self.teacher_encoders.train(False)
        self.base_encoder.train(mode)
        return self

    # overwrite of built-in function
    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        if len(args) > 0:
            if destination is None:
                destination = args[0]
            if len(args) > 1 and prefix == '':
                prefix = args[1]
            if len(args) > 2 and keep_vars is False:
                keep_vars = args[2]

        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()

        local_metadata = dict(version=self._version)
        if hasattr(destination, "_metadata"):
            destination._metadata[prefix[:-1]] = local_metadata

        self._save_to_state_dict(destination, prefix, keep_vars)
        for name, module in self._modules.items():
            # only save weights of student model
            if module is not None and name not in ["teacher_encoders"]:
                module.state_dict(destination=destination, prefix=prefix + name + '.', keep_vars=keep_vars)
        for hook in self._state_dict_hooks.values():
            hook_result = hook(self, destination, prefix, local_metadata)
            if hook_result is not None:
                destination = hook_result

        return destination


if __name__ == "__main__":
    pass
