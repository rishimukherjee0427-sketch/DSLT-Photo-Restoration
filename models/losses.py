"""
Loss Functions for DSLT v5 — Charbonnier + FFT Frequency + Multi-Scale Perceptual + SSIM + LAB
Designed for maximum PSNR/SSIM while maintaining perceptual quality.
MPS-compatible, optimized for vintage image restoration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models


class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss — robust variant of L1 that avoids over-smoothing.
    More stable gradients near zero compared to L1.
    """
    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.epsilon_sq = epsilon ** 2

    def forward(self, output, target):
        diff = output - target
        return torch.mean(torch.sqrt(diff * diff + self.epsilon_sq))


class FFTFrequencyLoss(nn.Module):
    """
    Frequency-domain loss using FFT.
    Penalizes differences in frequency space to preserve edges, textures,
    and high-frequency details that pixel-space losses tend to miss.
    """
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        # 2D FFT on spatial dimensions
        fft_output = torch.fft.rfft2(output, norm='ortho')
        fft_target = torch.fft.rfft2(target, norm='ortho')

        # Loss on magnitude spectrum (amplitude)
        mag_loss = F.l1_loss(fft_output.abs(), fft_target.abs())

        # Loss on phase (important for structure)
        phase_loss = F.l1_loss(
            torch.angle(fft_output),
            torch.angle(fft_target)
        )

        return mag_loss + 0.1 * phase_loss


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


class MultiScalePerceptualLoss(nn.Module):
    """
    Multi-scale VGG16 perceptual loss.
    Extracts features from conv1_2, conv2_2, conv3_3, conv4_3 for multi-scale matching.
    Much more effective than single-layer perceptual loss.
    """
    def __init__(self, device):
        super().__init__()
        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.DEFAULT)
        features = list(vgg.features)

        # Feature extraction at multiple scales
        # conv1_2: index 3, conv2_2: index 8, conv3_3: index 15, conv4_3: index 22
        self.slice1 = nn.Sequential(*features[:4]).to(device)   # conv1_2
        self.slice2 = nn.Sequential(*features[4:9]).to(device)  # conv2_2
        self.slice3 = nn.Sequential(*features[9:16]).to(device) # conv3_3
        self.slice4 = nn.Sequential(*features[16:23]).to(device)# conv4_3

        for module in [self.slice1, self.slice2, self.slice3, self.slice4]:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False

        # Weights for each scale (deeper features weighted less — they're more abstract)
        self.weights = [1.0, 0.8, 0.6, 0.4]

    def forward(self, output, target):
        loss = 0.0
        x_out = output
        x_tgt = target

        for i, (slice_module, w) in enumerate(zip(
            [self.slice1, self.slice2, self.slice3, self.slice4],
            self.weights
        )):
            x_out = slice_module(x_out)
            x_tgt = slice_module(x_tgt)
            loss += w * F.l1_loss(x_out, x_tgt)

        return loss


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


class EdgeLoss(nn.Module):
    """
    Edge-aware loss using Laplacian operator.
    Ensures the model preserves sharp edges and fine structures.
    """
    def __init__(self):
        super().__init__()
        # Laplacian kernel for edge detection
        kernel = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # Repeat for 3 channels
        self.register_buffer('kernel', kernel.repeat(3, 1, 1, 1))

    def forward(self, output, target):
        edge_out = F.conv2d(output, self.kernel, padding=1, groups=3)
        edge_tgt = F.conv2d(target, self.kernel, padding=1, groups=3)
        return F.l1_loss(edge_out, edge_tgt)


class CombinedLoss(nn.Module):
    """
    v5 Combined Loss:
    Charbonnier + Multi-Scale Perceptual + SSIM + LAB + FFT Frequency + Edge
    Balanced for maximum PSNR while maintaining perceptual quality.
    """
    def __init__(self, device):
        super().__init__()
        self.charbonnier = CharbonnierLoss(epsilon=1e-3)
        self.perceptual = MultiScalePerceptualLoss(device)
        self.ssim_loss = SSIMLoss(device=device)
        self.lab_loss = LABColorLoss().to(device)
        self.fft_loss = FFTFrequencyLoss().to(device)
        self.edge_loss = EdgeLoss().to(device)

        # Tuned weights for PSNR optimization
        self.w_perceptual = 0.05      # Lower than before — PSNR focus
        self.w_ssim = 0.15
        self.w_lab = 0.05
        self.w_fft = 0.10             # NEW: frequency domain supervision
        self.w_edge = 0.05            # NEW: edge preservation

    def forward(self, output, target):
        loss_char = self.charbonnier(output, target)
        loss_p = self.perceptual(output, target)
        loss_ssim = self.ssim_loss(output, target)
        loss_lab = self.lab_loss(output, target)
        loss_fft = self.fft_loss(output, target)
        loss_edge = self.edge_loss(output, target)

        total = (loss_char
                 + self.w_perceptual * loss_p
                 + self.w_ssim * loss_ssim
                 + self.w_lab * loss_lab
                 + self.w_fft * loss_fft
                 + self.w_edge * loss_edge)

        return total, loss_char, loss_p, loss_ssim
