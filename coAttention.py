###########################################################################
# Created by: CASIA IVA
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################

import numpy as np
import torch
import math
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable
torch_ver = torch.__version__[:3]

import numpy as np
import torch
from torch import nn
from torch.nn import init

__all__ = ['PCAM_Module', 'CCAM_Module', 'FusionBAMBlock']

class PCAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PCAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x, y):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1) ## (m_batchsize H*W C)
        proj_key = self.key_conv(y).view(m_batchsize, -1, width*height) ## (m_batchsize C H*W)
        energy = torch.bmm(proj_query, proj_key) ## bmm 批量矩阵乘法 (m_batchsize H*W H*W)
        attention = self.softmax(energy)    ## 归一化获得注意力
        proj_value = self.value_conv(y).view(m_batchsize, -1, width*height) ## (m_batchsize C1 H*W)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))## (m_batchsize C1 H*W)
        out = out.view(m_batchsize, C, height, width)

        '''
        这里要保证最后输出的维度和输入的 图像特征维度 一样，就需要保证
        输入depth特征图维度 和 输入的图像特征图 保持一致
        所以 depth特征图 进来之后就利用卷积层进行 维度上保持一致
        '''
        out = self.gamma*out + x ## gamma是个可学习参数
        return out ## (m_batchsize C H*W)

class CCAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CCAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self, x, y):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = y.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)      ## (m_batchsize C C)

        '''
        energy_new 是通过提取energy中每一列最大值并扩张为与energy同样
        '''

        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy ## (m_batchsize C C)
        attention = self.softmax(energy_new)
        proj_value = y.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16, num_layers=3):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        gate_channels = [channel]
        gate_channels += [channel // reduction] * num_layers
        gate_channels += [channel]

        self.ca = nn.Sequential()
        self.ca.add_module('flatten', Flatten())
        for i in range(len(gate_channels) - 2):
            self.ca.add_module('fc%d' % i, nn.Linear(gate_channels[i], gate_channels[i + 1]))
            self.ca.add_module('bn%d' % i, nn.BatchNorm1d(gate_channels[i + 1]))
            self.ca.add_module('relu%d' % i, nn.ReLU())
        self.ca.add_module('last_fc', nn.Linear(gate_channels[-2], gate_channels[-1]))

    def forward(self, x):
        res = self.avgpool(x)
        res = self.ca(res)
        # res=res.unsqueeze(-1).unsqueeze(-1).expand_as(x)
        return res


class FusionChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16, num_layers=1):
        super().__init__()
        self.caDepth = ChannelAttention(channel=channel, num_layers=num_layers)
        self.caImage = ChannelAttention(channel=channel, num_layers=num_layers)

        '''
        分别得到权重之后，进行一层MLP进行融合
        '''
        gate_channels = [channel * 2, (channel * 2) // reduction, channel]
        self.Fusion = nn.Sequential()
        for i in range(0, 2, 1):
            self.Fusion.add_module('fc{}'.format(i), nn.Linear(gate_channels[i], gate_channels[i + 1]))
            self.Fusion.add_module('bn{}'.format(i), nn.BatchNorm1d(gate_channels[i + 1]))
            self.Fusion.add_module('relu{}'.format(i), nn.ReLU())

        ## todo: whether need batchnorm and relu

    def forward(self, depth: torch.Tensor, Image: torch.Tensor):
        assert depth.ndim == Image.ndim, f'FusionCA input diff dim of tensor: depth dim = {depth.ndim}\tImage dim = {Image.ndim}'
        weightDepth = self.caDepth(depth)
        weightImage = self.caImage(Image)
        weight = torch.cat([weightDepth, weightImage], dim=1)
        res = self.Fusion(weight)
        res = res.unsqueeze(-1).unsqueeze(-1).expand_as(depth)
        return res


class SpatialAttention(nn.Module):
    def __init__(self, channel, reduction=16, num_layers=3, dia_val=2):
        super().__init__()
        self.sa = nn.Sequential()
        self.sa.add_module('conv_reduce1',
                           nn.Conv2d(kernel_size=1, in_channels=channel, out_channels=channel // reduction))
        self.sa.add_module('bn_reduce1', nn.BatchNorm2d(channel // reduction))
        self.sa.add_module('relu_reduce1', nn.ReLU())
        for i in range(num_layers):
            self.sa.add_module('conv_%d' % i, nn.Conv2d(kernel_size=3, in_channels=channel // reduction,
                                                        out_channels=channel // reduction, padding=1, dilation=dia_val))
            self.sa.add_module('bn_%d' % i, nn.BatchNorm2d(channel // reduction))
            self.sa.add_module('relu_%d' % i, nn.ReLU())
        # self.sa.add_module('last_conv',nn.Conv2d(channel//reduction,1,kernel_size=1))

    def forward(self, x):
        res = self.sa(x)
        # res=res.expand_as(x)
        return res


class FusionSpatialAttention(nn.Module):
    def __init__(self, channel, reduction=16, num_layers=3, dia_val=2):
        super().__init__()
        self.saDepth = SpatialAttention(channel=channel, reduction=reduction)
        self.saImage = SpatialAttention(channel=channel, reduction=reduction)

        self.Fusion = nn.Sequential()
        self.Fusion.add_module('cov', nn.Conv2d((channel * 2) // reduction, 1, kernel_size=1))
        self.Fusion.add_module('bn', nn.BatchNorm2d(1))
        # self.Fusion.add_module('relu', nn.ReLU())

    def forward(self, depth: torch.Tensor, Image: torch.Tensor):
        assert depth.ndim == Image.ndim, f'FusionSA input diff dim of tensor: depth dim = {depth.ndim}\tImage dim = {Image.ndim}'
        weightDepth = self.saDepth(depth)
        weightImage = self.saImage(Image)
        weightFusion = torch.cat([weightDepth, weightImage], dim=1)

        res = self.Fusion(weightFusion)
        res = res.expand_as(depth)
        return res


class BAMBlock(nn.Module):

    def __init__(self, channel=512, reduction=16, dia_val=2):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(channel=channel, reduction=reduction, dia_val=dia_val)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        sa_out = self.sa(x)
        ca_out = self.ca(x)
        weight = self.sigmoid(sa_out + ca_out)
        out = (1 + weight) * x
        return out


class FusionBAMBlock(nn.Module):

    def __init__(self, channel, reduction=16, dia_val=2):
        super().__init__()
        self.ca = FusionChannelAttention(channel=channel, reduction=reduction)
        self.sa = FusionSpatialAttention(channel=channel, reduction=reduction, dia_val=dia_val)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, depth: torch.Tensor, Image: torch.Tensor):
        assert depth.ndim == Image.ndim, f'FusionBAMBlock input diff dim of tensor: depth dim = {depth.ndim}\tImage dim = {Image.ndim}'
        b, c, _, _ = depth.size()
        sa_out = self.sa(depth, Image)
        ca_out = self.ca(depth, Image)
        weight = self.sigmoid(sa_out + ca_out)
        depth_image = depth + Image
        out = (1 + weight) * depth_image
        return out


if __name__ == '__main__':
    inputDepth = torch.randn(50, 512, 7, 7)
    inputImage = torch.randn(50, 512, 7, 7)
    Fusion = FusionBAMBlock(channel=512, reduction=16)
    output = Fusion(inputDepth, inputImage)
    print(output.shape)

