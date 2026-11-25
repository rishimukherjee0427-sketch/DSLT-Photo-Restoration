"""
Plain U-Net (Ronneberger, Fischer, Brox, MICCAI 2015) adapted for restoration: 4 down-sampling levels,
two 3x3 conv + ReLU per level (no normalisation, as in the original), max-pool down, bilinear up-sampling
followed by concatenation with the skip (the common 'bilinear' variant, bottleneck = 8 x base channels),
global residual and clamp to [0,1] like DSLT. base=64 -> ~13M parameters.

Deliberately minimal: no attention, no dense blocks, no latent translators, no damage estimator -- the
'how far does a vanilla encoder-decoder get' reference point.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def double_conv(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, base=64):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 8]
        self.inc = double_conv(3, c[0])
        self.downs = nn.ModuleList([double_conv(c[i], c[i + 1]) for i in range(4)])
        # up path: concat(upsampled, skip) -> double_conv
        self.ups = nn.ModuleList([double_conv(c[4] + c[3], c[2]), double_conv(c[2] + c[2], c[1]),
                                  double_conv(c[1] + c[1], c[0]), double_conv(c[0] + c[0], c[0])])
        self.outc = nn.Conv2d(c[0], 3, 1)

    def forward(self, inp):
        H, W = inp.shape[-2:]
        ph, pw = (-H) % 16, (-W) % 16
        x0 = F.pad(inp, (0, pw, 0, ph), mode="reflect") if (ph or pw) else inp
        skips = [self.inc(x0)]
        for d in self.downs:
            skips.append(d(F.max_pool2d(skips[-1], 2)))
        x = skips.pop()
        for u in self.ups:
            s = skips.pop()
            x = u(torch.cat([F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False), s], 1))
        x = self.outc(x) + x0
        return torch.clamp(x[..., :H, :W], 0.0, 1.0)


def build_unet(size="b64", **kw):
    cfg = {"b64": dict(base=64), "b32": dict(base=32)}[size]
    cfg.update(kw)
    return UNet(**cfg)
