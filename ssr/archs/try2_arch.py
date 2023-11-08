"""
Adapted from: https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/archs/rrdbnet_arch.py
Authors: XPixelGroup
"""
import time
import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import default_init_weights, make_layer, pixel_unshuffle


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Empirically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Empirically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x


@ARCH_REGISTRY.register()
class Try2(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(Try2, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        if self.scale == 8 or self.scale == 16:
            self.conv_up3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            if self.scale == 16:
                self.conv_up4 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.initial_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, num_feat, kernel_size=3, padding='same'),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(num_feat, num_feat, kernel_size=3, padding='same'),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(num_feat, num_feat, kernel_size=1, padding='same'),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(num_feat, num_feat, kernel_size=1, padding='same'),
            torch.nn.ReLU(inplace=True),
        )
        self.num_feat = num_feat
        self.collapse1 = torch.nn.Sequential(
            torch.nn.Conv2d(num_feat*2, num_feat, kernel_size=1, padding='same'),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(num_feat, num_feat, kernel_size=1, padding='same'),
            torch.nn.ReLU(inplace=True),
        )
        self.collapse2 = torch.nn.Sequential(
            torch.nn.Conv2d(num_feat*2, num_feat, kernel_size=1, padding='same'),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(num_feat, num_feat, kernel_size=1, padding='same'),
            torch.nn.ReLU(inplace=True),
        )
        self.collapse3 = torch.nn.Sequential(
            torch.nn.Conv2d(num_feat*2, num_feat, kernel_size=1, padding='same'),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(num_feat, num_feat, kernel_size=1, padding='same'),
            torch.nn.ReLU(inplace=True),
        )
        self.collapse4 = torch.nn.Sequential(
            torch.nn.Conv2d(num_feat*2, num_feat, kernel_size=1, padding='same'),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(num_feat, num_feat, kernel_size=1, padding='same'),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        feat = x
        n_b, n_c, n_h, n_w = feat.shape
        n_len = n_c//3
        print("feat shape:", feat.shape)
        print("n_len:", n_len)
        assert n_len==8
        feat = feat.reshape(n_b*16, 3, n_h, n_w)
        feat = self.initial_layers(feat)
        feat = feat.reshape(n_b*8, self.num_feat*2, n_h, n_w)
        feat = self.collapse1(feat)
        feat = feat.reshape(n_b*4, self.num_feat*2, n_h, n_w)
        feat = self.collapse2(feat)
        feat = feat.reshape(n_b*2, self.num_feat*2, n_h, n_w)
        feat = self.collapse3(feat)
        feat = feat.reshape(n_b, self.num_feat*2, n_h, n_w)
        feat = self.collapse4(feat)

        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))

        # Additional upsampling if doing x8 or x16.
        if self.scale == 8 or self.scale == 16:
            feat = self.lrelu(self.conv_up3(F.interpolate(feat, scale_factor=2, mode='nearest')))
            if self.scale == 16:
                feat = self.lrelu(self.conv_up4(F.interpolate(feat, scale_factor=2, mode='nearest')))

        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out