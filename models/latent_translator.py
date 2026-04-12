"""
Multi-Scale DSLT v5 Latent Translator — 4-Scale with Dilated Convolutions
4-scale translation: e2(128ch) + e3(256ch) + e4(512ch) + bottleneck(512ch)
Uses dilated convolutions for larger receptive field and channel shuffle for better mixing.
"""

import torch
import torch.nn as nn


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        r = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(r, channels, 1, bias=False),
        )
        self.spatial_conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        x = self.sigmoid(avg_out + max_out) * x
        avg_s = torch.mean(x, dim=1, keepdim=True)
        max_s, _ = torch.max(x, dim=1, keepdim=True)
        mask = self.sigmoid(self.spatial_conv(torch.cat([avg_s, max_s], dim=1)))
        return mask * x


class DilatedResidualBlock(nn.Module):
    """
    Residual block with mixed dilated convolutions for larger receptive field.
    Uses dilation rates [1, 2] alternating for multi-scale feature capture.
    """
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
        )
        self.cbam = CBAM(channels)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.lrelu(x + self.cbam(self.block(x)))


class ScaleTranslator(nn.Module):
    """
    Multi-block translator for a single scale.
    Alternates between dilation=1 and dilation=2 for multi-scale receptive fields.
    """
    def __init__(self, channels, num_blocks):
        super().__init__()
        blocks = []
        for i in range(num_blocks):
            dilation = 2 if i % 2 == 1 else 1  # Alternate dilations
            blocks.append(DilatedResidualBlock(channels, dilation=dilation))
        self.blocks = nn.Sequential(*blocks)
        # Residual scaling for training stability
        self.scale = nn.Parameter(torch.tensor(0.2))

    def forward(self, z):
        return z + self.blocks(z) * self.scale


class MultiScaleDSLT(nn.Module):
    """
    v5: 4-Scale DSLT with dilated convolutions:
      - Level e2 (96ch, 1/2 res):      4 blocks
      - Level e3 (192ch, 1/4 res):     6 blocks
      - Level e4 (384ch, 1/8 res):     4 blocks
      - Bottleneck (384ch, 1/16 res):  6 blocks
    """
    def __init__(self):
        super().__init__()
        self.translator_e2 = ScaleTranslator(96, num_blocks=3)
        self.translator_e3 = ScaleTranslator(192, num_blocks=4)
        self.translator_e4 = ScaleTranslator(384, num_blocks=3)
        self.translator_bn = ScaleTranslator(384, num_blocks=4)

    def forward(self, bottleneck, skips):
        e1, e2, e3, e4 = skips
        e2_t = self.translator_e2(e2)
        e3_t = self.translator_e3(e3)
        e4_t = self.translator_e4(e4)
        bn_t = self.translator_bn(bottleneck)
        return bn_t, [e1, e2_t, e3_t, e4_t]