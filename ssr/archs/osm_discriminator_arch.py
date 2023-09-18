import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class OSMDiscriminator(nn.Module):
    """
    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(OSMDiscriminator, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm

        """
        ### First trial - same layers as regular discriminator
        # osm layers
        self.o_conv0 = nn.Conv2d(3, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.o_conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.o_conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.o_conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.o_conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.o_conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.o_conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.o_conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.o_conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.o_conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)
        """
        ### Second trial - far smaller & simpler
        self.o_conv0 = nn.Conv2d(3, 32, kernel_size=1, stride=3, padding=1)
        self.o_conv1 = nn.Conv2d(32, 64, kernel_size=1, stride=3, padding=1)
        self.o_conv2 = nn.Conv2d(64, 128, kernel_size=1, stride=3, padding=1)
        self.o_conv3 = nn.Conv2d(128, 1, 1, 3, 0)

        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x, osm_objs):

        """
        # osm convolutions
        o0 = F.leaky_relu(self.o_conv0(osm_objs), negative_slope=0.2, inplace=True)
        o1 = F.leaky_relu(self.o_conv1(o0), negative_slope=0.2, inplace=True)
        o2 = F.leaky_relu(self.o_conv2(o1), negative_slope=0.2, inplace=True)
        o3 = F.leaky_relu(self.o_conv3(o2), negative_slope=0.2, inplace=True)
        o3 = F.interpolate(o3, scale_factor=2, mode='bilinear', align_corners=False)
        o4 = F.leaky_relu(self.o_conv4(o3), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            o4 = o4 + o2
        o4 = F.interpolate(o4, scale_factor=2, mode='bilinear', align_corners=False)
        o5 = F.leaky_relu(self.o_conv5(o4), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            o5 = o5 + o1
        o5 = F.interpolate(o5, scale_factor=2, mode='bilinear', align_corners=False)
        o6 = F.leaky_relu(self.o_conv6(o5), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            o6 = o6 + o0
        o7 = F.interpolate(o6, scale_factor=4, mode='bilinear', align_corners=False)
        o_out = F.leaky_relu(self.o_conv7(o7), negative_slope=0.2, inplace=True)
        o_out = F.leaky_relu(self.o_conv8(o_out), negative_slope=0.2, inplace=True)
        o_out = self.o_conv9(o_out)
        """

        # osm layers
        o0 = F.leaky_relu(self.o_conv0(osm_objs), negative_slope=0.2, inplace=True)
        o1 = F.leaky_relu(self.o_conv1(o0), negative_slope=0.2, inplace=True)
        o2 = F.leaky_relu(self.o_conv2(o1), negative_slope=0.2, inplace=True)
        o_out = F.leaky_relu(self.o_conv3(o2), negative_slope=0.2, inplace=True)

        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)
        return out, o_out
