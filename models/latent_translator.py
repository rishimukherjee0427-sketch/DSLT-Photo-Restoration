"""
Multi-Scale DSLT v4 Latent Translator — 3-Scale, Deeper Blocks
3-scale translation: e2(64ch,128x128) + e3(128ch,64x64) + bottleneck(256ch,32x32)
All 3 CBAM-residual translators are deeper, making the model more expressive.
"""

import torch
import torch.nn as nn


class CBAM(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        r = max(channels // reduction, 4)
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


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.cbam = CBAM(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.cbam(self.block(x)))


class ScaleTranslator(nn.Module):
    def __init__(self, channels, num_blocks):
        super().__init__()
        self.blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )

    def forward(self, z):
        return z + self.blocks(z)


class MultiScaleDSLT(nn.Module):
    """
    v4: 3-Scale DSLT (maximum accuracy configuration):
      - Level e2 (64ch, 128×128):  4 residual blocks  [NEW in v4]
      - Level e3 (128ch, 64×64):   6 residual blocks  [was 4]
      - Bottleneck (256ch, 32×32): 10 residual blocks [was 8]
    Total DSLT params: ~8.5M
    """
    def __init__(self):
        super().__init__()
        self.translator_e2 = ScaleTranslator(64, num_blocks=4)    # NEW
        self.translator_e3 = ScaleTranslator(128, num_blocks=6)   # was 4
        self.translator_bn = ScaleTranslator(256, num_blocks=10)  # was 8

    def forward(self, bottleneck, skips):
        e1, e2, e3 = skips
        e2_t = self.translator_e2(e2)   # NEW
        e3_t = self.translator_e3(e3)
        bn_t = self.translator_bn(bottleneck)
        return bn_t, [e1, e2_t, e3_t]  # pass translated e2 to decoder