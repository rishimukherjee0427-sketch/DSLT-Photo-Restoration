"""
Multi-Scale DSLT U-Net — v3 (CPU-Optimized)
3-level U-Net (32→64→128→256 bottleneck) with CBAM attention, bounded corrections.
"""

import torch
import torch.nn as nn


# ─── CBAM: Channel + Spatial Attention ────────────────────

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        r = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(r, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        mask = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return mask * x


class CBAM(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention()

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


# ─── Conv Blocks ──────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_cbam=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.cbam = CBAM(out_ch) if use_cbam else nn.Identity()

    def forward(self, x):
        return self.cbam(self.conv(x))


# ─── Encoder ─────────────────────────────────────────────

class Encoder(nn.Module):
    """3-level U-Net encoder: 3→32→64→128→256 bottleneck."""
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(3, 32, use_cbam=False)   # 256×256 (no CBAM at first level for speed)
        self.enc2 = ConvBlock(32, 64)                   # 128×128
        self.enc3 = ConvBlock(64, 128)                  # 64×64
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(128, 256)            # 32×32

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        return b, [e1, e2, e3]


# ─── Decoder ─────────────────────────────────────────────

class Decoder(nn.Module):
    """3-level decoder with skip connections + bounded output."""
    def __init__(self):
        super().__init__()
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32, use_cbam=False)

        # Bounded correction: tanh limits output to [-1, 1], scaled to [-0.5, 0.5]
        self.final = nn.Sequential(
            nn.Conv2d(32, 3, 1),
            nn.Tanh()
        )

    def forward(self, bottleneck, skips):
        e1, e2, e3 = skips

        d3 = self.up3(bottleneck)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.final(d1) * 0.5
