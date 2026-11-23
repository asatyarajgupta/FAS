import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class CDC(nn.Module):
    """Central difference convolution. Idea from Yu et al., CVPR 2020 [CREDIT: Architecture adapted from https://github.com/ZitongYu/CDCN (CVPR2020_paper_codes/models/CDCNs.py), MIT licence, research use only.]."""
    def __init__(self, IC, OC, K=3, P=1, theta=0.7):
        super().__init__()
        self.vani = nn.Conv2d(IC, OC, kernel_size=K, padding=P)
        self.theta = theta
    def forward(self, x):
        k = self.vani.weight.sum(2).sum(2)[:, :, None, None]
        return self.vani(x) - self.theta * F.conv2d(x, k, padding=0)

class CDCNBlock(nn.Module):
    def __init__(self, i, o, theta=0.7):
        super().__init__()
        self.cdc = CDC(i, o, theta=theta); self.bn = nn.BatchNorm2d(o); self.relu = nn.ReLU(inplace=False)
    def forward(self, x): return self.relu(self.bn(self.cdc(x)))

def _stage(theta):
    return nn.Sequential(CDCNBlock(128, 128, theta), CDCNBlock(128, 196, theta), CDCNBlock(196, 128, theta),
                         nn.MaxPool2d(3, 2, 1))

class CDCN(nn.Module):
    def __init__(self, theta=0.7):
        super().__init__()
        self.stem = CDCNBlock(3, 64, theta)
        self.low_block = nn.Sequential(CDCNBlock(64, 128, theta), CDCNBlock(128, 196, theta),
                                       CDCNBlock(196, 128, theta), nn.MaxPool2d(3, 2, 1))
        self.mid_block, self.high_block = _stage(theta), _stage(theta)
        self.head = nn.Sequential(CDCNBlock(384, 128, theta), CDCNBlock(128, 64, theta), CDC(64, 1, theta=theta))
    def forward(self, x):
        h = self.stem(x)
        if self.training:      # gradient checkpointing: saves GPU memory, no change to results
            lo = checkpoint(self.low_block, h, use_reentrant=False)
            mi = checkpoint(self.mid_block, lo, use_reentrant=False)
            hi = checkpoint(self.high_block, mi, use_reentrant=False)
        else:
            lo = self.low_block(h); mi = self.mid_block(lo); hi = self.high_block(mi)
        f = [F.interpolate(t, size=(32, 32), mode='bilinear', align_corners=False) for t in (lo, mi, hi)]
        return self.head(torch.cat(f, dim=1))        # [B, 1, 32, 32]

class SobelGradientLoss(nn.Module):
    """L1 on Sobel gradients. NOT the eight-neighbour Contrastive Depth Loss of the CDCN paper."""
    def __init__(self):
        super().__init__()
        self.register_buffer('wx', torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3))
        self.register_buffer('wy', torch.tensor([[-1., -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3))
        self.l1 = nn.L1Loss()
    def forward(self, p, t):
        return (self.l1(F.conv2d(p, self.wx, padding=1), F.conv2d(t, self.wx, padding=1)) +
                self.l1(F.conv2d(p, self.wy, padding=1), F.conv2d(t, self.wy, padding=1)))

class CDCNLoss(nn.Module):
    def __init__(self, use_grad=True):
        super().__init__()
        self.mse, self.grad, self.use_grad = nn.MSELoss(), SobelGradientLoss(), use_grad
    def forward(self, p, t):
        return self.mse(p, t) + (self.grad(p, t) if self.use_grad else 0.0)