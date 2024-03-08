import torch
import math
import numpy as np
import cv2
from pywt import dwt2, wavedec2
from torch import nn as nn
from torch.nn import functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from thop import profile  # 计算参数量和运算量
from fvcore.nn import FlopCountAnalysis, parameter_count_table    # 计算参数量和运算量
from torchkeras import summary
from .arch_util import default_init_weights, make_layer, pixel_unshuffle
from pywt import dwt2, idwt2
from torchstat import stat
# from pytorch_wavelets import DWTForward, DWTInverse # (or import DWT, IDWT)


class MSDB(nn.Module):    # 3-5-7-9
    def __init__(self, channels, mid_channels):
        super(MSDB, self).__init__()
        self.conv_first = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels * 3, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels * 5, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(mid_channels * 7, channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=9, stride=1, padding=4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        default_init_weights(
            [self.conv_first, self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv7, self.conv9],
            0.1)

    def forward(self, x0):
        x = self.conv_first(x0)
        x31 = self.conv1(x)
        x51 = self.conv5(x)
        x32 = self.conv2(torch.cat([x, x31, x51], 1))
        x52 = self.conv5(x51)
        x71 = self.conv7(x51)
        x33 = self.conv3(torch.cat([x, x31, x32, x52, x71], 1))
        x53 = self.conv5(x52)
        x72 = self.conv7(x71)
        x91 = self.conv9(x71)
        out = self.conv4(torch.cat([x, x31, x32, x33, x53, x72, x91], 1))
        out = torch.add(out, x0)
        return out


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class SMSDB(nn.Module):
    def __init__(self, channels, mid_channels):
        super(SMSDB, self).__init__()
        self.layers = make_layer(basic_block=MSDB, num_basic_block=4, channels=channels, mid_channels=mid_channels)

    def forward(self, x):
        x1 = self.layers(x)
        out = torch.add(x1, x)
        return out


class upsampling(nn.Module):
    def __init__(self, channels):
        super(upsampling, self).__init__()

        self.up = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.Conv2d(16, channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


class DWT1(nn.Module):
    def __init__(self, channels, mid_channels):
        super(DWT1, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels * 4, channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.DWT = DWT()
        self.SMSDB = SMSDB(channels, mid_channels)
        self.upsampling = upsampling(channels)

    def forward(self, x):
        x = self.upsampling(x)
        x = self.DWT(x)
        x = self.conv1(x)
        x = self.SMSDB(x)
        x = self.conv(x)
        return x


class DWT2(nn.Module):
    def __init__(self, channels, mid_channels):
        super(DWT2, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels * 4, channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.DWT = DWT()
        self.SMSDB = SMSDB(channels, mid_channels)

    def forward(self, x):
        x = self.DWT(x)
        x = self.conv1(x)
        x = self.SMSDB(x)
        x = self.conv(x)
        return x


class Stage1(nn.Module):
    def __init__(self, channels, mid_channels):
        super(Stage1, self).__init__()
        self.SMSDB = SMSDB(channels, mid_channels)
        self.upsampling = upsampling(channels)
        self.DWT1 = DWT1(channels, mid_channels)
        self.IWT = IWT()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        D11 = self.DWT1(x)
        M11 = self.SMSDB(x)
        D12 = torch.add(D11, self.DWT1(M11))
        M12 = self.SMSDB(M11)
        D13 = torch.add(D12, self.DWT1(M12))
        M13 = self.SMSDB(M12)
        D14 = torch.add(D13, self.DWT1(M13))
        D14 = self.conv1(self.IWT(self.conv2(D14)))
        M14 = self.upsampling(torch.add(self.SMSDB(M13), x))
        out = torch.add(D14, M14)
        return out


class Stage2(nn.Module):
    def __init__(self, channels, mid_channels):
        super(Stage2, self).__init__()
        self.SMSDB = SMSDB(channels, mid_channels)
        self.upsampling = upsampling(channels)
        self.DWT2 = DWT2(channels, mid_channels)
        self.IWT = IWT()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        D21 = self.DWT2(x)
        M21 = self.SMSDB(x)
        D22 = torch.add(D21, self.DWT2(M21))
        M22 = self.SMSDB(M21)
        D23 = torch.add(D22, self.DWT2(M22))
        M23 = self.SMSDB(M22)
        D24 = torch.add(D23, self.DWT2(M23))
        D24 = self.conv1(self.IWT(self.conv2(D24)))
        out = torch.add(D24, torch.add(self.SMSDB(M23), x))
        out = self.upsampling(out)
        return out


@ARCH_REGISTRY.register()
class WFFGAN(nn.Module):
    def __init__(self, channels=64, mid_channels=32):
        super(WFFGAN, self).__init__()

        self.conv_first = nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1)
        self.stage1 = Stage1(channels, mid_channels)
        self.stage2 = Stage2(channels, mid_channels)
        self.upsampling = upsampling(channels)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv_last = nn.Sequential(
            nn.Conv2d(channels, 3, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.conv_first(x)
        x = self.stage1(x)
        x = self.conv(x)
        x = self.stage2(x)
        x = self.conv_last(x)
        return x

