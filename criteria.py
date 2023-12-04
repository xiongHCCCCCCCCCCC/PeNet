import torch
import torch.nn as nn
import numpy as np

loss_names = ['l1', 'l2']

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach() # valid_mask 是元素值为 True or False 的矩阵
        diff = target - pred
        diff = diff[valid_mask] #取出valid_mask为True的位置的值
        self.loss = (diff**2).mean()#求均值，得出误差
        return self.loss


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target, weight=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss


class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack((edge_kx, edge_ky))

        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.edge_conv(x)
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))

        return out


class gradLoss():
    def __init__(self) -> None:
        self.getGrad = Sobel().cuda()
        self.cos = nn.CosineSimilarity(dim=1, eps=0)

    def forward(self, depth, output, lambda1, lambda2, lambda3):
        assert depth.dim() == output.dim(), "inconsistent dimensions"
        maskSignleChannel = (depth > 0).detach()

        depth_grad = self.getGrad(depth)
        output_grad = self.getGrad(output)
        depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth)
        depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth)
        output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth)
        output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth)

        ones = torch.ones(depth.size(0), 1, depth.size(2), depth.size(3)).float().cuda()
        ones = torch.autograd.Variable(ones)

        depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
        output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)

        loss_depth = torch.log(torch.abs(output[maskSignleChannel] - depth[maskSignleChannel]) + 0.5).mean()
        loss_dx = torch.log(torch.abs(output_grad_dx[maskSignleChannel] - depth_grad_dx[maskSignleChannel]) + 0.5).mean()
        loss_dy = torch.log(torch.abs(output_grad_dy[maskSignleChannel] - depth_grad_dy[maskSignleChannel]) + 0.5).mean()
        diffNorm = self.cos(output_normal, depth_normal).contiguous().view(output.size())
        loss_normal = torch.abs(1 - diffNorm[maskSignleChannel]).mean()

        loss = lambda1 * loss_depth + lambda2 * (loss_dx + loss_dy) + lambda3 * loss_normal
        return loss
