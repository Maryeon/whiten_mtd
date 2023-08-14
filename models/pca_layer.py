import torch
import torch.nn as nn
from utils import PCA


def mocov3_pca_layer(
    pretrained_weights="/home/zju/jieche.mz/pretrained_models/pca_weights/mocov3_pca_512d_svd_224x224_p1.pt"
):
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    pca_layer = PCA(dim1=checkpoint["dim1"].item(), dim2=checkpoint["dim2"].item())
    pca_layer.load_state_dict(checkpoint)

    return pca_layer


def barlowtwins_pca_layer(
    pretrained_weights="/home/zju/jieche.mz/pretrained_models/pca_weights/barlowtwins_pca_512d_svd_224x224_p1.pt"
):
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    pca_layer = PCA(dim1=checkpoint["dim1"].item(), dim2=checkpoint["dim2"].item())
    pca_layer.load_state_dict(checkpoint)

    return pca_layer


def resnet101_gem_pca_layer(embed_dim=512):
    pretrained_weights=f"/home/zju/jieche.mz/pretrained_models/pca_weights/resnet101_gem_pca_{embed_dim}d_gldv2_512x512_p3_randcrop.pt"
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    pca_layer = PCA(dim1=checkpoint["dim1"].item(), dim2=checkpoint["dim2"].item())
    pca_layer.load_state_dict(checkpoint)

    return pca_layer


def resnet101_gem_relu_pca_layer(
    pretrained_weights="/home/zju/jieche.mz/pretrained_models/pca_weights/resnet101_gem_relu_pca_512d_gldv2_512x512_p3.pt"
):
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    pca_layer = PCA(dim1=checkpoint["dim1"].item(), dim2=checkpoint["dim2"].item())
    pca_layer.load_state_dict(checkpoint)

    return pca_layer


def resnet101_ap_gem_pca_layer(embed_dim=512):
    pretrained_weights=f"/home/zju/jieche.mz/pretrained_models/pca_weights/resnet101_ap_gem_pca_{embed_dim}d_gldv2_512x512_p3_randcrop.pt"
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    pca_layer = PCA(dim1=checkpoint["dim1"].item(), dim2=checkpoint["dim2"].item())
    pca_layer.load_state_dict(checkpoint)

    return pca_layer


def resnet101_solar_pca_layer(embed_dim=512):
    pretrained_weights=f"/home/zju/jieche.mz/pretrained_models/pca_weights/resnet101_solar_pca_{embed_dim}d_gldv2_512x512_p3_randcrop.pt"
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    pca_layer = PCA(dim1=checkpoint["dim1"].item(), dim2=checkpoint["dim2"].item())
    pca_layer.load_state_dict(checkpoint)

    return pca_layer


def resnet50_delg_pca_layer(
    pretrained_weights="/home/zju/jieche.mz/pretrained_models/pca_weights/resnet50_delg_pca_512d_gldv2_512x512_p3.pt"
):
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    pca_layer = PCA(dim1=checkpoint["dim1"].item(), dim2=checkpoint["dim2"].item())
    pca_layer.load_state_dict(checkpoint)

    return pca_layer


def resnet101_delg_pca_layer(embed_dim=512):
    pretrained_weights=f"/home/zju/jieche.mz/pretrained_models/pca_weights/resnet101_delg_pca_{embed_dim}d_gldv2_512x512_p3_randcrop.pt"
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    pca_layer = PCA(dim1=checkpoint["dim1"].item(), dim2=checkpoint["dim2"].item())
    pca_layer.load_state_dict(checkpoint)

    return pca_layer


def resnet101_dolg_pca_layer(embed_dim=512):
    pretrained_weights=f"/home/zju/jieche.mz/pretrained_models/pca_weights/resnet101_dolg_pca_{embed_dim}d_gldv2_512x512_p3_global_randcrop.pt"
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    pca_layer = PCA(dim1=checkpoint["dim1"].item(), dim2=checkpoint["dim2"].item())
    pca_layer.load_state_dict(checkpoint)

    return pca_layer


def resnet101_random_pca_layer(
    pretrained_weights="/home/zju/jieche.mz/pretrained_models/pca_weights/resnet101_random_pca_512d_gldv2_512x512_p3.pt"
):
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    pca_layer = PCA(dim1=checkpoint["dim1"].item(), dim2=checkpoint["dim2"].item())
    pca_layer.load_state_dict(checkpoint)

    return pca_layer


pca_layers = {
    "mocov3": mocov3_pca_layer,
    "barlowtwins": barlowtwins_pca_layer,
    "resnet101_gem": resnet101_gem_pca_layer,
    "resnet101_gem_relu": resnet101_gem_relu_pca_layer,
    "resnet101_ap_gem": resnet101_ap_gem_pca_layer,
    "resnet101_solar": resnet101_solar_pca_layer,
    "resnet50_delg": resnet50_delg_pca_layer,
    "resnet101_delg": resnet101_delg_pca_layer,
    "resnet101_dolg": resnet101_dolg_pca_layer,
    "resnet101_random": resnet101_random_pca_layer
}