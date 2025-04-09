import copy
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
from io import open
import torch.nn.functional as F

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, SmoothL1Loss
import numpy as np
from functools import partial
from .vit import VisionTransformer


def resize_pos_embed(posemb, posemb_new):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]
    if True:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    # _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    orig = posemb_grid.dtype
    posemb_grid = F.interpolate(posemb_grid.float(), size=(gs_new, gs_new), mode='bilinear')
    posemb_grid = posemb_grid.to(orig)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def initialize_clip(config):
    """
    初始化CLIP视觉编码器。
    """
    vision_width = config['vision_width']
    vision_model = VisionTransformer(
        img_size=config['image_res'], 
        patch_size=config['patch_size'], 
        embed_dim=vision_width, 
        depth=12, 
        num_heads=12, 
        mlp_ratio=4, 
        qkv_bias=True,
        drop_path_rate=0.
    )
    
    return vision_model, vision_width


def initialize_vit(VISUAL_CONFIG, model_type="ViT-B_32", pretrained_dir="data/ViT-B_32.npz", img_size=(384, 640),
                   num_patches=240):
    # 使用本地的VisionTransformer实现
    model = VisionTransformer(
        img_size=224, 
        patch_size=16, 
        embed_dim=768, 
        depth=12, 
        num_heads=12, 
        mlp_ratio=4, 
        qkv_bias=True,
        zero_head=True,
        num_classes=1
    )
    
    # 可以通过自定义方式加载预训练权重
    if os.path.exists(pretrained_dir):
        print(f"加载预训练权重: {pretrained_dir}")
        state_dict = torch.load(pretrained_dir) if pretrained_dir.endswith('.pth') else np.load(pretrained_dir)
        if isinstance(state_dict, np.ndarray):
            # 处理numpy权重文件
            pass
        elif isinstance(state_dict, dict):
            # 处理torch权重文件
            model.load_state_dict(state_dict, strict=False)
    
    pos_embed = nn.Parameter(torch.zeros(num_patches + 1, 768).float())
    if hasattr(model, 'pos_embed'):
        pos_embed.weight = resize_pos_embed(model.pos_embed, pos_embed.unsqueeze(0))
    
    if VISUAL_CONFIG.freeze_clip:
        for parameter in model.parameters():
            parameter.requires_grad = False
    return model


def initialize_optimizer(visual_model, lr, momentum, weight_decay):
    optimizer = torch.optim.SGD(visual_model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)
    return optimizer


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.sgd_lr

    for milestone in args.schedule.split(","):
        lr *= 0.1 if epoch >= float(milestone) else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


from torch.optim import Optimizer


class FusedOptimizer(Optimizer):
    def __init__(self, optimizers):
        self.optimizers = optimizers
        param_groups = []
        for optimizer in self.optimizers:
            param_groups += optimizer.param_groups
        # super(FusedOptimizer, self).__init__([], {})
        self.param_groups = param_groups

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()
