"""
Loss Functions for DSLT v3 — L1 + Perceptual + SSIM + LAB Color Loss
MPS-compatible, highly optimized for vintage image restoration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models


class SSIMLoss(nn.Module):
    """Differentiable SSIM Loss (1 - SSIM)."""
    def __init__(self, window_size=11, channels=3, device="cpu"):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = (g.unsqueeze(1) @ g.unsqueeze(0)).unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
        self.register_buffer('window', window)

    def forward(self, output, target):
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        pad = self.window_size // 2
        mu1 = F.conv2d(output, self.window, padding=pad, groups=self.channels)
        mu2 = F.conv2d(target, self.window, padding=pad, groups=self.channels)
        sigma1_sq = F.conv2d(output ** 2, self.window, padding=pad, groups=self.channels) - mu1 ** 2
        sigma2_sq = F.conv2d(target ** 2, self.window, padding=pad, groups=self.channels) - mu2 ** 2
        sigma12 = F.conv2d(output * target, self.window, padding=pad, groups=self.channels) - mu1 * mu2
        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1.0 - ssim_map.mean()


class PerceptualLoss(nn.Module):
    """VGG16 perceptual loss (conv3_3)."""
    def __init__(self, device):
        super().__init__()
        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.DEFAULT)
        self.features = nn.Sequential(*list(vgg.features[:16])).to(device)
        self.features.eval()
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, output, target):
        return F.l1_loss(self.features(output), self.features(target))


class LABColorLoss(nn.Module):
    """
    Computes L1 loss on the continuous A and B color channels of LAB space.
    Approximated cleanly in tensor operations to fight pink/magenta artifacts.
    Assumes inputs are RGB [0, 1].
    """
    def __init__(self):
        super().__init__()
        # RGB to XYZ matrix
        M = torch.tensor([
            [0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227]
        ]).view(3, 3, 1, 1)
        self.register_buffer('M', M)

    def rgb_to_xyz(self, rgb):
        # Inverse sRGB gamma safely
        rgb = torch.clamp(rgb, 0.0, 1.0)
        mask = (rgb > 0.04045).float()
        val = torch.clamp((rgb + 0.055) / 1.055, min=1e-6)
        rgb_linear = mask * torch.pow(val, 2.4) + (1 - mask) * (rgb / 12.92)
        xyz = F.conv2d(rgb_linear, self.M)
        return xyz

    def xyz_to_lab(self, xyz):
        # D65 white point
        xyz_norm = xyz / torch.tensor([0.950456, 1.0, 1.088754], device=xyz.device).view(1, 3, 1, 1)
        xyz_norm = torch.clamp(xyz_norm, min=1e-6)
        mask = (xyz_norm > 0.008856).float()
        f_xyz = mask * torch.pow(xyz_norm, 1/3) + (1 - mask) * (7.787 * xyz_norm + 16/116)
        
        L = 116 * f_xyz[:, 1:2] - 16
        a = 500 * (f_xyz[:, 0:1] - f_xyz[:, 1:2])
        b = 200 * (f_xyz[:, 1:2] - f_xyz[:, 2:3])
        return torch.cat([L, a, b], dim=1)

    def forward(self, output, target):
        lab_out = self.xyz_to_lab(self.rgb_to_xyz(output))
        lab_tgt = self.xyz_to_lab(self.rgb_to_xyz(target))
        # We only care about color channels (a, b) to penalize color bleed
        return F.l1_loss(lab_out[:, 1:], lab_tgt[:, 1:])


class CombinedLoss(nn.Module):
    """
    L1 + 0.1 * Perceptual + 0.2 * SSIM + 0.05 * LAB Color Consistency
    Optimized for highest visual fidelity on vintage photos.
    """
    def __init__(self, device):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.perceptual = PerceptualLoss(device)
        self.ssim_loss = SSIMLoss(device=device)
        self.lab_loss = LABColorLoss().to(device)
        
        # Danger level weights: SSIM heavily favored to maintain details
        self.w_perceptual = 0.1
        self.w_ssim = 0.20
        self.w_lab = 0.05

    def forward(self, output, target):
        loss_l1 = self.l1(output, target)
        loss_p = self.perceptual(output, target)
        loss_ssim = self.ssim_loss(output, target)
        loss_lab = self.lab_loss(output, target)
        
        total = loss_l1 + (self.w_perceptual * loss_p) + (self.w_ssim * loss_ssim) + (self.w_lab * loss_lab)
        return total, loss_l1, loss_p, loss_ssim
