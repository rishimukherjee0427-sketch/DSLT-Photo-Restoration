"""
NAFNet (Chen, Chu, Zhang, Sun, "Simple Baselines for Image Restoration", ECCV 2022) -- from-scratch
re-implementation of the published architecture (NAFBlock = LayerNorm -> 1x1 -> 3x3 depthwise -> SimpleGate ->
simplified channel attention -> 1x1, then LayerNorm -> 1x1 -> SimpleGate -> 1x1, each branch with a learnable
zero-initialised residual scale). Default = the paper's width-32 configuration (enc [1,1,1,28], middle 1,
dec [1,1,1,1]; ~17M parameters). Not the official checkpoint; trained from scratch on this project's data.

forward(x): x in [0,1] (B,3,H,W) -> restored in [0,1] (global residual, clamped like DSLT). Any H, W (padded
internally to a multiple of 16).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, c, dw_expand=2, ffn_expand=2):
        super().__init__()
        dw = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.conv3 = nn.Conv2d(dw // 2, c, 1)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw // 2, dw // 2, 1))
        self.sg = SimpleGate()
        ffn = c * ffn_expand
        self.conv4 = nn.Conv2d(c, ffn, 1)
        self.conv5 = nn.Conv2d(ffn // 2, c, 1)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp):
        x = self.sg(self.conv2(self.conv1(self.norm1(inp))))
        x = self.conv3(x * self.sca(x))
        y = inp + x * self.beta
        x = self.conv5(self.sg(self.conv4(self.norm2(y))))
        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(self, width=32, middle_blk_num=1, enc_blk_nums=(1, 1, 1, 28), dec_blk_nums=(1, 1, 1, 1)):
        super().__init__()
        self.intro = nn.Conv2d(3, width, 3, padding=1)
        self.ending = nn.Conv2d(width, 3, 3, padding=1)
        self.encoders, self.downs, self.ups, self.decoders = (nn.ModuleList() for _ in range(4))
        ch = width
        for n in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(n)]))
            self.downs.append(nn.Conv2d(ch, 2 * ch, 2, 2))
            ch *= 2
        self.middle = nn.Sequential(*[NAFBlock(ch) for _ in range(middle_blk_num)])
        for n in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(ch, ch * 2, 1, bias=False), nn.PixelShuffle(2)))
            ch //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(n)]))
        self.mult = 2 ** len(enc_blk_nums)

    def forward(self, inp):
        H, W = inp.shape[-2:]
        ph, pw = (-H) % self.mult, (-W) % self.mult
        x0 = F.pad(inp, (0, pw, 0, ph), mode="reflect") if (ph or pw) else inp
        x = self.intro(x0)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)
        x = self.middle(x)
        for dec, up, s in zip(self.decoders, self.ups, skips[::-1]):
            x = dec(up(x) + s)
        x = self.ending(x) + x0
        return torch.clamp(x[..., :H, :W], 0.0, 1.0)


def build_nafnet(size="w32", **kw):
    cfg = {"w32": dict(width=32), "w16": dict(width=16)}[size]
    cfg.update(kw)
    return NAFNet(**cfg)
