"""
models/damage_estimator.py
Lightweight CNN encoder-decoder predicting:
  (a) 4-channel severity map (B, 4, H, W) in [0, 1] [blur, noise, scratch, fade]
  (b) 4 scalar gate values (B, 4) in [0, 1] for [e2, e3, e4, bottleneck]

Gate head (fixed in the gating-diagnosis pass, see experiments/diagnose_gates.py):
  gate_head='legacy' : original head. MLP on global-AVG-pooled e4 only, sigmoid with
                       default init (gates start ~0.5), no supervision path. Kept so the
                       "before" behaviour can be reproduced / diagnosed.
  gate_head='rich'   : (default) MLP on concat of
                         avg+max pooled e3 (2*128), avg+max pooled e4 (2*256) and
                         avg+max pooled *detached* predicted severity map (2*4)
                       with a final-layer bias initialised so gates start at
                       sigmoid(gate_bias_init) ~= 0.73 (not stuck at 0.5).
                       Gates are meant to be supervised with
                       CombinedLoss.compute_gate_loss (models/losses.py).

forward() signature / return shapes are unchanged: (severity_map (B,4,H,W), gates (B,4)).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DamageEstimator(nn.Module):
    def __init__(self, in_channels=3, base_ch=32, gate_head='rich', gate_bias_init=1.0):
        super().__init__()
        assert gate_head in ('legacy', 'rich')
        self.gate_head = gate_head

        # --- Lightweight Encoder ---
        self.enc1 = ConvBlock(in_channels, base_ch, stride=1)      # (B, 32, H, W)
        self.enc2 = ConvBlock(base_ch, base_ch * 2, stride=2)     # (B, 64, H/2, W/2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4, stride=2) # (B, 128, H/4, W/4)
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8, stride=2) # (B, 256, H/8, W/8)

        # --- Global Latent Gate Predictor (MLP head) ---
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        if gate_head == 'legacy':
            self.gate_mlp = nn.Sequential(
                nn.Linear(base_ch * 8, base_ch * 4),
                nn.ReLU(inplace=True),
                nn.Linear(base_ch * 4, 4),
                nn.Sigmoid()  # Gates in range [0, 1]
            )
        else:
            self.global_max = nn.AdaptiveMaxPool2d(1)
            in_dim = 2 * base_ch * 4 + 2 * base_ch * 8 + 2 * 4
            self.gate_mlp = nn.Sequential(
                nn.Linear(in_dim, base_ch * 4),
                nn.ReLU(inplace=True),
                nn.Linear(base_ch * 4, 4),
            )
            # Gate logits start at gate_bias_init (sigmoid(1.0)=0.73) instead of ~0.5
            nn.init.constant_(self.gate_mlp[-1].bias, gate_bias_init)

        # --- Lightweight Decoder (for Pixel-level 4-channel Severity Map) ---
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=4, stride=2, padding=1)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=4, stride=2, padding=1)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=4, stride=2, padding=1)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)

        # Final output layer: 4 channels (Blur, Noise, Scratch, Fade)
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_ch, 4, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Severity targets in [0, 1]
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Degraded input image (B, 3, H, W)
        Returns:
            severity_map (Tensor): Predicted damage map (B, 4, H, W)
            scale_gates (Tensor): (B, 4) gates for [e2, e3, e4, bottleneck]
        """
        # Encoder
        e1 = self.enc1(x)   # (B, 32, H, W)
        e2 = self.enc2(e1)  # (B, 64, H/2, W/2)
        e3 = self.enc3(e2)  # (B, 128, H/4, W/4)
        e4 = self.enc4(e3)  # (B, 256, H/8, W/8)

        # Branch 2: Pixel-level severity map decoding
        d3 = self.up3(e4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        severity_map = self.out_conv(d1)  # (B, 4, H, W)

        # Branch 1: Scale gates prediction
        B = e4.size(0)
        if self.gate_head == 'legacy':
            pooled = self.global_pool(e4).view(B, -1)
            scale_gates = self.gate_mlp(pooled)  # (B, 4)
        else:
            sev = severity_map.detach()  # gate input only; gate loss must not distort severity head
            feats = torch.cat([
                self.global_pool(e3).view(B, -1), self.global_max(e3).view(B, -1),
                self.global_pool(e4).view(B, -1), self.global_max(e4).view(B, -1),
                self.global_pool(sev).view(B, -1), self.global_max(sev).view(B, -1),
            ], dim=1)
            scale_gates = torch.sigmoid(self.gate_mlp(feats))  # (B, 4)

        return severity_map, scale_gates
