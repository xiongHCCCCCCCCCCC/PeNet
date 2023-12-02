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

__all__ = ['PCAM_Module', 'CCAM_Module']

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

# class PCCMA_Moudle(Module):
#     def __init__(self, indimRGB, indimDepth):
#         super(PCCMA_Moudle, self).__init__()
#
#         self.CCAM = CCAM_Module(indimDepth)
#         self.PCAM = PCAM_Module(indimRGB)