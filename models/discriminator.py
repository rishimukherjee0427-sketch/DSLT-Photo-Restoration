"""
PatchGAN Discriminator for DSLT v5 — Optional GAN Sharpening
Multi-scale 70×70 PatchGAN with spectral normalization.
~800K parameters — lightweight relative to the 30M generator.
Training-only module: NEVER used at inference time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralNormConv(nn.Module):
    """Conv2d with spectral normalization for stable GAN training."""
    def __init__(self, in_ch, out_ch, kernel_size=4, stride=2, padding=1, use_norm=True):
        super().__init__()
        conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        self.conv = nn.utils.spectral_norm(conv)
        self.norm = nn.InstanceNorm2d(out_ch) if use_norm else nn.Identity()
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class PatchGANDiscriminator(nn.Module):
    """
    70×70 PatchGAN discriminator.
    Classifies overlapping 70×70 patches as real/fake rather than the whole image.
    This encourages high-frequency sharpness without affecting global structure.

    Architecture:
        3 → 64 (stride 2) → 128 (stride 2) → 256 (stride 2) → 512 (stride 1) → 1 (stride 1)

    Output: [B, 1, H', W'] map of patch-level real/fake scores.
    ~800K parameters.
    """
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()

        self.model = nn.Sequential(
            # Layer 1: No normalization on first layer (standard PatchGAN practice)
            SpectralNormConv(in_channels, base_channels, 4, stride=2, padding=1, use_norm=False),

            # Layer 2
            SpectralNormConv(base_channels, base_channels * 2, 4, stride=2, padding=1),

            # Layer 3
            SpectralNormConv(base_channels * 2, base_channels * 4, 4, stride=2, padding=1),

            # Layer 4: stride 1 for finer spatial discrimination
            SpectralNormConv(base_channels * 4, base_channels * 8, 4, stride=1, padding=1),

            # Output: 1-channel score map (no activation — use BCEWithLogitsLoss)
            nn.Conv2d(base_channels * 8, 1, 4, stride=1, padding=1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable GAN start."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] image tensor in [0, 1]
        Returns:
            patch_scores: [B, 1, H', W'] real/fake logits per patch
        """
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator: runs PatchGAN at 2 scales (original + 0.5×).
    Captures both fine details and coarser structure artifacts.
    Total ~1.6M params (2 × 800K), but can be used with single scale too.
    """
    def __init__(self, in_channels=3, base_channels=64, num_scales=2):
        super().__init__()
        self.num_scales = num_scales
        self.discriminators = nn.ModuleList([
            PatchGANDiscriminator(in_channels, base_channels)
            for _ in range(num_scales)
        ])
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1)

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] image tensor
        Returns:
            outputs: list of patch score maps at each scale
        """
        outputs = []
        current = x
        for i, disc in enumerate(self.discriminators):
            outputs.append(disc(current))
            if i < self.num_scales - 1:
                current = self.downsample(current)
        return outputs


def get_discriminator(multi_scale=False, device='cpu'):
    """
    Factory function to create a discriminator.

    Args:
        multi_scale: If True, use 2-scale discriminator (~1.6M params).
                     If False, use single-scale (~800K params).
        device: Target device.

    Returns:
        discriminator module on the specified device.
    """
    if multi_scale:
        disc = MultiScaleDiscriminator().to(device)
    else:
        disc = PatchGANDiscriminator().to(device)

    n_params = sum(p.numel() for p in disc.parameters())
    print(f"{'Multi-scale' if multi_scale else 'Single-scale'} PatchGAN: {n_params:,} params")
    return disc
