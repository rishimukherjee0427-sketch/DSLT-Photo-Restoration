"""
DSLT v5 U-Net — Deep Residual Encoder-Decoder with PixelShuffle
4-level encoder (64→128→256→512) with strided conv downsampling,
RRDB bottleneck, PixelShuffle upsampling, and global skip connection.
Designed for 30+ dB PSNR restoration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── CBAM: Channel + Spatial Attention ────────────────────

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        r = max(channels // reduction, 8)
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
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention()

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


# ─── Residual Dense Block (RDB) ─────────────────────────

class ResidualDenseBlock(nn.Module):
    """Dense block with residual scaling for stable deep networks."""
    def __init__(self, channels, growth_rate=16, num_layers=3, residual_scale=0.2):
        super().__init__()
        self.residual_scale = residual_scale
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_ch = channels + i * growth_rate
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_ch, growth_rate, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ))
        # Final 1x1 conv to squeeze back to original channels
        self.squeeze = nn.Conv2d(channels + num_layers * growth_rate, channels, 1)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        out = self.squeeze(torch.cat(features, dim=1))
        return x + out * self.residual_scale


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block — the core of ESRGAN/Real-ESRGAN."""
    def __init__(self, channels, growth_rate=16, residual_scale=0.2):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(channels, growth_rate, residual_scale=residual_scale)
        self.rdb2 = ResidualDenseBlock(channels, growth_rate, residual_scale=residual_scale)
        self.rdb3 = ResidualDenseBlock(channels, growth_rate, residual_scale=residual_scale)
        self.residual_scale = residual_scale

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + out * self.residual_scale


# ─── Conv Blocks ──────────────────────────────────────────

class ConvBlock(nn.Module):
    """Double conv with optional CBAM and residual connection."""
    def __init__(self, in_ch, out_ch, use_cbam=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),       # InstanceNorm > BatchNorm for restoration
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.cbam = CBAM(out_ch) if use_cbam else nn.Identity()
        # Residual projection if channel count changes
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.cbam(self.conv(x))
        return out + identity


class DownBlock(nn.Module):
    """Strided convolution downsample (replaces MaxPool — preserves more info)."""
    def __init__(self, in_ch, out_ch, use_cbam=True):
        super().__init__()
        self.down = nn.Conv2d(in_ch, in_ch, 4, stride=2, padding=1)  # Strided downsample
        self.conv = ConvBlock(in_ch, out_ch, use_cbam=use_cbam)

    def forward(self, x):
        return self.conv(self.down(x))


class UpBlock(nn.Module):
    """PixelShuffle upsample (replaces ConvTranspose2d — no checkerboard artifacts)."""
    def __init__(self, in_ch, skip_ch, out_ch, use_cbam=True):
        super().__init__()
        # PixelShuffle: in_ch → in_ch * 4 channels → shuffle to 2x spatial → in_ch channels
        self.up = nn.Sequential(
            nn.Conv2d(in_ch, in_ch * 4, 3, padding=1),
            nn.PixelShuffle(2),  # in_ch * 4 → in_ch at 2x resolution
        )
        self.conv = ConvBlock(in_ch + skip_ch, out_ch, use_cbam=use_cbam)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatches from odd dimensions
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ─── Encoder ─────────────────────────────────────────────

class Encoder(nn.Module):
    """
    4-level encoder: 3→48→96→192→384 bottleneck.
    Uses strided conv downsampling and CBAM at all levels.
    Optimized channel widths for ~30M total params.
    """
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(3, 48, use_cbam=True)           # Full resolution
        self.enc2 = DownBlock(48, 96, use_cbam=True)          # 1/2
        self.enc3 = DownBlock(96, 192, use_cbam=True)         # 1/4
        self.enc4 = DownBlock(192, 384, use_cbam=True)        # 1/8
        # RRDB bottleneck for deep feature extraction
        self.bottleneck = nn.Sequential(
            nn.Conv2d(384, 384, 3, stride=2, padding=1),      # 1/16
            *[RRDB(384, growth_rate=16) for _ in range(4)],   # 4 RRDB blocks
            CBAM(384),
        )

    def forward(self, x):
        e1 = self.enc1(x)        # 48ch @ full res
        e2 = self.enc2(e1)       # 96ch @ 1/2
        e3 = self.enc3(e2)       # 192ch @ 1/4
        e4 = self.enc4(e3)       # 384ch @ 1/8
        b = self.bottleneck(e4)  # 384ch @ 1/16
        return b, [e1, e2, e3, e4]


# ─── Decoder ─────────────────────────────────────────────

class Decoder(nn.Module):
    """
    4-level decoder with PixelShuffle upsampling, skip connections,
    and a learnable global residual connection.
    """
    def __init__(self):
        super().__init__()
        self.up4 = UpBlock(384, 384, 256, use_cbam=True)    # 1/16 → 1/8 (squeeze)
        self.up3 = UpBlock(256, 192, 192, use_cbam=True)    # 1/8 → 1/4
        self.up2 = UpBlock(192, 96, 96, use_cbam=True)      # 1/4 → 1/2
        self.up1 = UpBlock(96, 48, 48, use_cbam=True)       # 1/2 → full

        # Residual output head
        self.output_conv = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(48, 3, 3, padding=1),
        )

        # Learnable residual scaling — starts small, grows during training
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, bottleneck, skips):
        e1, e2, e3, e4 = skips

        d4 = self.up4(bottleneck, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        correction = self.output_conv(d1) * self.residual_scale
        return correction
