import os
import copy
import time
import numpy as np

from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import models


####################################################################################################
########################################## Functions ###############################################
####################################################################################################


## Kaiming weight initialisation
def weights_init(module):
    if isinstance(module, nn.ReLU):
        pass
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, nn.BatchNorm2d):
        pass
        #nn.init.kaiming_normal_(module.weight.data)
        #nn.init.constant_(module.bias.data, 0.0)

def constant_init(module):
    if isinstance(module, nn.ReLU):
        pass
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        nn.init.constant_(module.weight.data, 0.0)
        nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, nn.BatchNorm2d):
        pass
        #nn.init.kaiming_normal_(module.weight.data)



####################################################################################################
########################################## Networks ###############################################
####################################################################################################

class SOABlock(nn.Module):
    def __init__(self, in_ch, k):
        super(SOABlock, self).__init__()

        self.in_ch = in_ch
        self.out_ch = in_ch
        self.mid_ch = in_ch // k

        self.f = nn.Sequential(
                nn.Conv2d(self.in_ch, self.mid_ch, (1, 1), (1, 1)),
                nn.BatchNorm2d(self.mid_ch),
                nn.ReLU())
        self.g = nn.Sequential(
                nn.Conv2d(self.in_ch, self.mid_ch, (1, 1), (1, 1)),
                nn.BatchNorm2d(self.mid_ch),
                nn.ReLU())
        self.h = nn.Conv2d(self.in_ch, self.mid_ch, (1, 1), (1, 1))
        self.v =nn.Conv2d(self.mid_ch, self.out_ch, (1, 1), (1, 1))

        self.softmax = nn.Softmax(dim=-1)

        for conv in [self.f, self.g, self.h]:    #, self.v]:
            conv.apply(weights_init)

        self.v.apply(constant_init)


    def forward(self, x, vis_mode=False):
        B, C, H, W = x.shape

        f_x = self.f(x).view(B, self.mid_ch, H * W) # B * mid_ch * N, where N = H*W
        g_x = self.g(x).view(B, self.mid_ch, H * W) # B * mid_ch * N, where N = H*W
        h_x = self.h(x).view(B, self.mid_ch, H * W) # B * mid_ch * N, where N = H*W

        z = torch.bmm(f_x.permute(0, 2, 1), g_x) # B * N * N, where N = H*W

        if vis_mode:
            # for visualisation only
            attn = self.softmax((self.mid_ch ** -.75) * z)
        else:
            attn = self.softmax((self.mid_ch ** -.50) * z)

        z = torch.bmm(attn, h_x.permute(0, 2, 1)) # B * N * mid_ch, where N = H*W
        z = z.permute(0, 2, 1).view(B, self.mid_ch, H, W) # B * mid_ch * H * W

        z = self.v(z)
        z = z + x

        return z, attn


class ResNetSOAs(nn.Module):
    def __init__(self, architecture='resnet101', soa_layers='45'):
        super(ResNetSOAs, self).__init__()

        base_model = vars(models)[architecture](pretrained=False)
        last_feat_in = base_model.inplanes
        base_model = nn.Sequential(*list(base_model.children())[:-2])

        res_blocks = list(base_model.children())

        self.conv1 = nn.Sequential(*res_blocks[0:2])
        self.conv2_x = nn.Sequential(*res_blocks[2:5])
        self.conv3_x = res_blocks[5]
        self.conv4_x = res_blocks[6]
        self.conv5_x = res_blocks[7]

        self.soa_layers = soa_layers
        if '4' in self.soa_layers:
            self.soa4 = SOABlock(in_ch=last_feat_in // 2, k=4)
        if '5' in self.soa_layers:
            self.soa5 = SOABlock(in_ch=last_feat_in, k=2)

        self.embed_dim = last_feat_in

    def forward(self, x, mode='test'):
        with torch.no_grad():
            x = self.conv1(x)
            x = self.conv2_x(x)
            x = self.conv3_x(x)
            x = self.conv4_x(x)

        # start SOA blocks
        if '4' in self.soa_layers:
            x, soa_m2 = self.soa4(x, mode == 'draw')
        
        x = self.conv5_x(x)
        if '5' in self.soa_layers:
            x, soa_m1 = self.soa5(x, mode == 'draw')

        if mode == 'draw':
            return x, soa_m2, soa_m1

        return x
